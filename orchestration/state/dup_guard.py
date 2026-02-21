from __future__ import annotations

import time
from typing import Optional

from orchestration.event_bus.redis_bus import RedisBus


class DupGuard:
    """
    Simple cooldown guard using Redis SETNX + TTL.
    If key exists -> duplicate within cooldown.

    Notes:
      - cooldown_sec <= 0 => dedup disabled (always allow)
      - Redis error => fail-open (avoid freezing system)
    """

    def __init__(self, bus: RedisBus, prefix: str = "dup", default_cooldown_sec: int = 20) -> None:
        self.bus = bus
        self.prefix = prefix
        self.default_cooldown_sec = int(default_cooldown_sec)

    def allow(self, key: str, cooldown_sec: Optional[int] = None) -> bool:
        cd_raw = self.default_cooldown_sec if cooldown_sec is None else int(cooldown_sec)
        cd = int(cd_raw)

        # disable dedup if misconfigured
        if cd <= 0:
            return True

        rk = f"{self.prefix}:{key}"
        try:
            # SET key value NX EX cd
            ok = self.bus.r.set(rk, str(int(time.time())), nx=True, ex=cd)
            return bool(ok)
        except Exception:
            # if redis fails, fail-open (avoid freezing system)
            return True
