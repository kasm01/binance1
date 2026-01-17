from __future__ import annotations

import time
from typing import Optional

from orchestration.event_bus.redis_bus import RedisBus


class DupGuard:
    """
    Simple cooldown guard using Redis SETNX + TTL.
    If key exists -> duplicate within cooldown.
    """
    def __init__(self, bus: RedisBus, prefix: str = "dup", default_cooldown_sec: int = 20) -> None:
        self.bus = bus
        self.prefix = prefix
        self.default_cooldown_sec = int(default_cooldown_sec)

    def allow(self, key: str, cooldown_sec: Optional[int] = None) -> bool:
        cd = int(cooldown_sec or self.default_cooldown_sec)
        rk = f"{self.prefix}:{key}"
        try:
            # SET key value NX EX cd
            ok = self.bus.r.set(rk, str(int(time.time())), nx=True, ex=cd)
            return bool(ok)
        except Exception:
            # if redis fails, fail-open (avoid freezing system)
            return True
