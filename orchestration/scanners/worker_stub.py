from __future__ import annotations

import os
import random
import time
import uuid
from typing import List

from orchestration.event_bus.redis_bus import RedisBus


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default


def main() -> None:
    bus = RedisBus()
    assert bus.ping(), "[WorkerStub] Redis ping failed."

    producer_id = os.getenv("WORKER_ID", "w0")
    symbols: List[str] = [s.strip().upper() for s in os.getenv("WORKER_SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT").split(",") if s.strip()]
    interval = os.getenv("WORKER_INTERVAL", "5m")

    # HOLD event basmayı kapat (spam azaltır)
    publish_hold = _env_bool("WORKER_PUBLISH_HOLD", False)

    print(f"[WorkerStub] started. publishing to {bus.signals_stream} ...")
    while True:
        sym = random.choice(symbols)
        p = random.random()
        side = "none"
        if p > 0.62:
            side = "long"
        elif p < 0.38:
            side = "short"

        if side == "none" and not publish_hold:
            time.sleep(0.5)
            continue

        evt = {
            "event_id": str(uuid.uuid4()),
            "producer_id": producer_id,
            "symbol": sym,
            "interval": interval,
            "side_candidate": side,
            "p_used": p,
            "p_single": max(0.0, min(1.0, p + random.uniform(-0.05, 0.05))),
            "confidence": random.uniform(0.3, 0.9),
            "source": "STUB",
            "whale_dir": "none",
            "whale_score": 0.0,
            "atr_pct": random.uniform(0.005, 0.03),
            "spread_pct": random.uniform(0.0001, 0.0009),
            "vol_score": random.uniform(0.0, 1.0),
            "cooldown_key": f"{sym}|{interval}|{side}",
            "meta": {"note": "stub_event"},
        }

        bus.publish_signal(evt)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
