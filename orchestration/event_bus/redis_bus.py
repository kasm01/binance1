from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import redis


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v


@dataclass
class RedisBusConfig:
    host: str = _env("REDIS_HOST", "127.0.0.1") or "127.0.0.1"
    port: int = int(_env("REDIS_PORT", "6379") or "6379")
    db: int = int(_env("REDIS_DB", "0") or "0")
    password: Optional[str] = _env("REDIS_PASSWORD", None)


class RedisBus:
    """
    Redis Streams based bus:
      - signals stream: workers -> aggregator
      - candidates stream: aggregator -> main executor
    """
    def __init__(
        self,
        cfg: Optional[RedisBusConfig] = None,
        signals_stream: str = "signals_stream",
        candidates_stream: str = "candidates_stream",
    ) -> None:
        self.cfg = cfg or RedisBusConfig()
        self.signals_stream = signals_stream
        self.candidates_stream = candidates_stream

        self.r = redis.Redis(
            host=self.cfg.host,
            port=self.cfg.port,
            db=self.cfg.db,
            password=self.cfg.password,
            decode_responses=True,  # strings
            socket_connect_timeout=2,
            socket_timeout=5,
        )

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def xadd_json(self, stream: str, payload: Dict[str, Any], maxlen: int = 20000) -> str:
        data = {"json": json.dumps(payload, ensure_ascii=False)}
        # approximate trimming for speed
        return self.r.xadd(stream, data, maxlen=maxlen, approximate=True)

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
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns list of (message_id, payload_dict)
        """
        if create_group:
            try:
                self.r.xgroup_create(stream, group, id="0-0", mkstream=True)
            except Exception:
                pass  # group exists

        resp = self.r.xreadgroup(group, consumer, {stream: start_id}, count=count, block=block_ms)
        out: List[Tuple[str, Dict[str, Any]]] = []
        if not resp:
            return out

        # resp: [(stream, [(id, {field:val})...])]
        for _stream, msgs in resp:
            for mid, fields in msgs:
                raw = fields.get("json")
                if not raw:
                    continue
                try:
                    out.append((mid, json.loads(raw)))
                except Exception:
                    continue
        return out

    def xack(self, stream: str, group: str, message_ids: Iterable[str]) -> int:
        ids = list(message_ids)
        if not ids:
            return 0
        return int(self.r.xack(stream, group, *ids))
