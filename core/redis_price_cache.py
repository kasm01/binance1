from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class RedisPriceCache:
    """
    Redis backed simple price cache.

    Key format:
      pricecache:<SYMBOL>

    Value JSON:
      {
        "symbol": "BTCUSDT",
        "bid": 71000.1,
        "ask": 71000.3,
        "mid": 71000.2,
        "ts": 1770000000.123
      }
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_prefix: str = "pricecache",
        ttl_sec: int = 15,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.key_prefix = str(key_prefix or "pricecache").strip() or "pricecache"
        self.ttl_sec = int(ttl_sec or 15)
        self._client = None

        if redis is not None:
            try:
                self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            except Exception:
                self._client = None

    def _key(self, symbol: str) -> str:
        return f"{self.key_prefix}:{str(symbol).upper()}"

    def is_available(self) -> bool:
        return self._client is not None

    def set_bid_ask(
        self,
        symbol: str,
        bid: float,
        ask: float,
        ts: Optional[float] = None,
    ) -> None:
        if self._client is None:
            return

        try:
            bid_f = float(bid)
            ask_f = float(ask)
            if bid_f <= 0 or ask_f <= 0:
                return

            if ask_f < bid_f:
                bid_f, ask_f = min(bid_f, ask_f), max(bid_f, ask_f)

            now = float(ts) if ts is not None else time.time()
            mid = (bid_f + ask_f) / 2.0

            payload = {
                "symbol": str(symbol).upper(),
                "bid": bid_f,
                "ask": ask_f,
                "mid": mid,
                "ts": now,
            }

            self._client.set(self._key(symbol), json.dumps(payload), ex=self.ttl_sec)
        except Exception:
            return

    def get_mid(self, symbol: str, max_age_sec: float = 2.0) -> Optional[float]:
        if self._client is None:
            return None

        try:
            raw = self._client.get(self._key(symbol))
            if not raw:
                return None

            data: Any = json.loads(raw)
            if not isinstance(data, dict):
                return None

            mid = float(data.get("mid", 0.0) or 0.0)
            ts = float(data.get("ts", 0.0) or 0.0)

            if mid <= 0:
                return None

            if max_age_sec > 0 and ts > 0 and (time.time() - ts) > float(max_age_sec):
                return None

            return mid
        except Exception:
            return None

