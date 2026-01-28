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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


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
    scanner_mode = os.getenv("SCANNER_MODE", "fast").strip().lower()

    publish_hold = _env_bool("WORKER_PUBLISH_HOLD", False)
    simulate_whale = _env_bool("WORKER_SIM_WHALE", True)

    # weights (whale-first)
    w_whale = float(os.getenv("W_SCORE_WHALE", "0.60"))
    w_fast = float(os.getenv("W_SCORE_FAST", "0.25"))
    w_micro = float(os.getenv("W_SCORE_MICRO", "0.15"))

    print(
        f"[WorkerStub] started. publishing to {bus.signals_stream} ... "
        f"worker={producer_id} mode={scanner_mode} syms={symbols} itv={interval}"
    )

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
                whale_dir = "buy" if side == "long" else "sell"
            else:
                whale_dir = "sell" if side == "long" else "buy"

        atr_pct = random.uniform(0.004, 0.035)
        spread_pct = random.uniform(0.0001, 0.0012)
        vol_score = random.uniform(0.0, 1.0)

        # micro_score: spread düşük + vol iyi + atr makul => yüksek
        spr_component = _clamp(0.0006 / max(spread_pct, 1e-9), 0.0, 1.0)
        atr_component = _clamp(0.020 / max(atr_pct, 1e-9), 0.0, 1.0)
        micro_score = _clamp(0.40 * spr_component + 0.30 * atr_component + 0.30 * vol_score, 0.0, 1.0)

        # fast_model_score: stub için p_used'u kullan
        fast_model_score = _clamp(p, 0.0, 1.0)

        score_total = _clamp((w_whale * _clamp(whale_score)) + (w_fast * fast_model_score) + (w_micro * micro_score), 0.0, 1.0)

        evt = {
            "event_id": str(uuid.uuid4()),
            "producer_id": producer_id,
            "symbol": sym,
            "interval": interval,
            "interval_base": "1m",  # yeni alan (scalp planı)
            "side_candidate": side,

            # model-ish
            "p_used": p,
            "p_single": max(0.0, min(1.0, p + random.uniform(-0.05, 0.05))),
            "confidence": random.uniform(0.25, 0.95),

            "source": "STUB",

            # whale/micro
            "whale_dir": whale_dir,
            "whale_score": float(whale_score),

            "atr_pct": float(atr_pct),
            "spread_pct": float(spread_pct),
            "vol_score": float(vol_score),

            # NEW fields for pipeline
            "micro_score": float(micro_score),
            "fast_model_score": float(fast_model_score),
            "score_total": float(score_total),

            "cooldown_key": f"{sym}|{interval}|{side}",
            "meta": {"note": "stub_event", "stage": "fast", "scanner_mode": scanner_mode},
        }

        bus.publish_signal(evt)
        time.sleep(0.8)


if __name__ == "__main__":
    main()

