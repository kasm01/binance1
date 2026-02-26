from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_id_to_epoch_ms(stream_id: str) -> int:
    try:
        return int(str(stream_id).split("-", 1)[0])
    except Exception:
        return int(time.time() * 1000)


def _epoch_ms_to_iso(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return _now_utc_iso()


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
      - candidates_stream: aggregator -> top selector

    Env overrides:
      - SIGNALS_STREAM
      - CANDIDATES_STREAM

    Extra env:
      - BUS_XADD_MAXLEN (default 20000)
        maxlen <= 0 => no trimming
    """

    def __init__(
        self,
        cfg: Optional[RedisBusConfig] = None,
        signals_stream: Optional[str] = None,
        candidates_stream: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or RedisBusConfig()

        self.signals_stream = signals_stream or _env("SIGNALS_STREAM", "signals_stream") or "signals_stream"
        self.candidates_stream = (
            candidates_stream or _env("CANDIDATES_STREAM", "candidates_stream") or "candidates_stream"
        )

        self.xadd_maxlen_default = _env_int("BUS_XADD_MAXLEN", 20000)

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

    def xadd_json(self, stream: str, payload: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        """
        Best-effort XADD with JSON payload.

        maxlen:
          - None => BUS_XADD_MAXLEN default
          - <=0  => no trimming
        """
        data = {"json": json.dumps(payload, ensure_ascii=False)}

        ml = self.xadd_maxlen_default if maxlen is None else int(maxlen)
        try:
            if ml <= 0:
                return self.r.xadd(stream, data)
            return self.r.xadd(stream, data, maxlen=ml, approximate=True)
        except Exception:
            # fail-open: best effort, return empty id to avoid crashing
            return ""

    def publish_signal(self, payload: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        return self.xadd_json(self.signals_stream, payload, maxlen=maxlen)

    def publish_candidate(self, payload: Dict[str, Any], maxlen: Optional[int] = None) -> str:
        return self.xadd_json(self.candidates_stream, payload, maxlen=maxlen)

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
        add_ts_utc_if_missing: bool = True,
        add_source_stream_id: bool = True,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns list of (message_id, payload_dict)

        start_id:
          ">" => new messages
          "0" => pending (PEL)

        add_ts_utc_if_missing:
          if payload lacks ts_utc, fill from stream-id timestamp

        add_source_stream_id:
          add _source_stream_id=mid (useful for tracing)
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
                    if not isinstance(obj, dict):
                        continue

                    if add_source_stream_id and ("_source_stream_id" not in obj):
                        obj["_source_stream_id"] = mid

                    if add_ts_utc_if_missing and (not obj.get("ts_utc")):
                        ms = _stream_id_to_epoch_ms(mid)
                        obj["ts_utc"] = _epoch_ms_to_iso(ms)

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
    def xrevrange_json(
        self,
        stream: str,
        count: int = 1,
        require_json_field: bool = True,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read last N messages from a stream, parse {"json": "..."} payloads.
        Returns [(mid, obj), ...] newest-first.
        """
        try:
            rows = self.r.xrevrange(stream, max="+", min="-", count=int(count))
        except Exception:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        for mid, fields in rows:
            if not isinstance(fields, dict):
                continue
            raw = fields.get("json")
            if not raw:
                if require_json_field:
                    continue
                out.append((mid, dict(fields)))
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    out.append((mid, obj))
            except Exception:
                continue
        return out

    def xlen_safe(self, stream: str) -> int:
        try:
            return int(self.r.xlen(stream))
        except Exception:
            return 0

    def delete_keys(self, *keys: str) -> int:
        """
        Best-effort DEL; returns deleted count (0 on error).
        """
        ks = [k for k in keys if k]
        if not ks:
            return 0
        try:
            return int(self.r.delete(*ks))
        except Exception:
            return 0
