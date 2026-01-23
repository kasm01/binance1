from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


@dataclass
class RedisBusConfig:
    host: str = _env("REDIS_HOST", "127.0.0.1") or "127.0.0.1"
    port: int = _env_int("REDIS_PORT", 6379)
    db: int = _env_int("REDIS_DB", 0)
    password: Optional[str] = _env("REDIS_PASSWORD", None)


class RedisBus:
    """
    Redis Streams based bus.

    Default streams:
      - signals_stream: workers -> aggregator
      - candidates_stream: aggregator -> top selector (or executor)

    Env overrides:
      - SIGNALS_STREAM
      - CANDIDATES_STREAM
    """

    def __init__(
        self,
        cfg: Optional[RedisBusConfig] = None,
        signals_stream: Optional[str] = None,
        candidates_stream: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or RedisBusConfig()

        self.signals_stream = signals_stream or _env("SIGNALS_STREAM", "signals_stream") or "signals_stream"
        self.candidates_stream = candidates_stream or _env("CANDIDATES_STREAM", "candidates_stream") or "candidates_stream"

        self.r = redis.Redis(
            host=self.cfg.host,
            port=self.cfg.port,
            db=self.cfg.db,
            password=self.cfg.password,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        """
        Ensure consumer group exists; create stream if missing (MKSTREAM).
        Safe to call multiple times.
        """
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise
        except Exception:
            # fail-open
            return

    def xadd_json(self, stream: str, payload: Dict[str, Any], maxlen: int = 20000) -> str:
        data = {"json": json.dumps(payload, ensure_ascii=False)}
        try:
            return self.r.xadd(stream, data, maxlen=maxlen, approximate=True)
        except Exception:
            # fail-open: best effort, return empty id to avoid crashing
            return ""

    def publish_signal(self, payload: Dict[str, Any]) -> str:
        return self.xadd_json(self.signals_stream, payload)

    def publish_candidate(self, payload: Dict[str, Any]) -> str:
        return self.xadd_json(self.candidates_stream, payload)

    def xreadgroup_json(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 50,
        block_ms: int = 1000,
        start_id: str = ">",
        create_group: bool = True,
        group_start_id: str = "$",
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns list of (message_id, payload_dict)

        start_id:
          ">" => new messages
          "0" => pending (PEL)
        """
        if create_group:
            try:
                self.ensure_group(stream, group, start_id=group_start_id)
            except Exception:
                pass

        try:
            resp = self.r.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: start_id},
                count=count,
                block=block_ms,
            )
        except redis.exceptions.ResponseError as e:
            msg = str(e)
            # Stream key deleted while blocking
            if "UNBLOCKED" in msg and "no longer exists" in msg:
                try:
                    self.ensure_group(stream, group, start_id=group_start_id)
                except Exception:
                    pass
                return []
            return []
        except Exception:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        if not resp:
            return out

        for _stream_name, msgs in resp:
            for mid, fields in msgs:
                raw = fields.get("json")
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        out.append((mid, obj))
                except Exception:
                    continue
        return out

    def xack(self, stream: str, group: str, message_ids: Iterable[str]) -> int:
        ids = list(message_ids)
        if not ids:
            return 0
        try:
            return int(self.r.xack(stream, group, *ids))
        except Exception:
            return 0
