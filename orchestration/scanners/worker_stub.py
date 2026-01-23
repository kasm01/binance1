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
    symbols: List[str] = [
        s.strip().upper()
        for s in os.getenv("WORKER_SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT").split(",")
        if s.strip()
    ]
    interval = os.getenv("WORKER_INTERVAL", "5m")

    publish_hold = _env_bool("WORKER_PUBLISH_HOLD", False)
    simulate_whale = _env_bool("WORKER_SIM_WHALE", True)

    print(f"[WorkerStub] started. publishing to {bus.signals_stream} ... worker={producer_id} syms={symbols} itv={interval}")
    while True:
        sym = random.choice(symbols)
        p = random.random()

        side = "none"
        if p > 0.62:
            side = "long"
        elif p < 0.38:
            side = "short"

        if side == "none" and not publish_hold:
            time.sleep(0.3)
            continue

        # whale sim: bazen align bazen contra
        whale_dir = "none"
        whale_score = 0.0
        if simulate_whale and random.random() < 0.35:
            whale_score = random.uniform(0.05, 0.60)
            if random.random() < 0.65:
                # align
                whale_dir = "buy" if side == "long" else "sell"
            else:
                # contra
                whale_dir = "sell" if side == "long" else "buy"

        evt = {
            "event_id": str(uuid.uuid4()),
            "producer_id": producer_id,
            "symbol": sym,
            "interval": interval,
            "side_candidate": side,
            "p_used": p,
            "p_single": max(0.0, min(1.0, p + random.uniform(-0.05, 0.05))),
            "confidence": random.uniform(0.25, 0.95),
            "source": "STUB",
            "whale_dir": whale_dir,
            "whale_score": whale_score,
            "atr_pct": random.uniform(0.004, 0.035),
            "spread_pct": random.uniform(0.0001, 0.0012),
            "vol_score": random.uniform(0.0, 1.0),
            "cooldown_key": f"{sym}|{interval}|{side}",
            "meta": {"note": "stub_event"},
        }

        bus.publish_signal(evt)
        time.sleep(0.8)


if __name__ == "__main__":
    main()

