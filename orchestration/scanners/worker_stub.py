from __future__ import annotations

import random
import time
import uuid
from typing import List

from orchestration.event_bus.redis_bus import RedisBus


def main() -> None:
    bus = RedisBus()
    assert bus.ping(), "[WorkerStub] Redis ping failed."

    producer_id = "w0"
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    interval = "5m"

    print("[WorkerStub] started. publishing to signals_stream ...")
    while True:
        sym = random.choice(symbols)
        p = random.random()
        side = "none"
        if p > 0.62:
            side = "long"
        elif p < 0.38:
            side = "short"

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
