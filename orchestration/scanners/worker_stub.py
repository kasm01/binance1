from __future__ import annotations

import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

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


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    bus = RedisBus()
    assert bus.ping(), "[WorkerStub] Redis ping failed."

    # Identity / routing
    source = os.getenv("WORKER_ID", "w0").strip() or "w0"
    symbols: List[str] = [
        s.strip().upper()
        for s in os.getenv("WORKER_SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT").split(",")
        if s.strip()
    ]
    interval = os.getenv("WORKER_INTERVAL", "5m").strip() or "5m"
    scanner_mode = os.getenv("SCANNER_MODE", "fast").strip().lower() or "fast"

    # Publish controls
    publish_hold = _env_bool("WORKER_PUBLISH_HOLD", False)       # side=none bile bas (debug)
    simulate_whale = _env_bool("WORKER_SIM_WHALE", True)
    emit_meta = _env_bool("WORKER_EMIT_META", True)              # meta şişmesini istersen kapat

    # Pace
    sleep_sec = _env_float("WORKER_SLEEP_SEC", 0.80)
    hold_skip_sleep_sec = _env_float("WORKER_HOLD_SKIP_SLEEP_SEC", 0.30)

    # Anti-spam / quality gates (Worker layer)
    # spread_pct bir oran: 0.0006 = %0.06
    max_spread_pct = _env_float("WORKER_MAX_SPREAD_PCT", 0.0010)  # default %0.10
    max_atr_pct = _env_float("WORKER_MAX_ATR_PCT", 0.0300)        # default %3.0
    min_conf = _env_float("WORKER_MIN_CONF", 0.45)                # default 0.45

    # Scoring weights (stub only)
    w_whale = _env_float("W_SCORE_WHALE", 0.60)
    w_fast = _env_float("W_SCORE_FAST", 0.25)
    w_micro = _env_float("W_SCORE_MICRO", 0.15)

    # Optional deterministic seed (debug)
    seed = os.getenv("WORKER_SEED")
    if seed:
        try:
            random.seed(int(seed))
        except Exception:
            random.seed(seed)

    print(
        f"[WorkerStub] started. publishing to {bus.signals_stream} | "
        f"source={source} mode={scanner_mode} itv={interval} syms={symbols} | "
        f"gates(min_conf={min_conf} max_spread={max_spread_pct} max_atr={max_atr_pct}) | "
        f"sleep={sleep_sec}s meta={emit_meta}"
    )

    while True:
        sym = random.choice(symbols)

        # ----- side candidate (long/short/none) -----
        p = random.random()
        side_candidate = "none"
        if p > 0.62:
            side_candidate = "long"
        elif p < 0.38:
            side_candidate = "short"

        if side_candidate == "none" and not publish_hold:
            time.sleep(hold_skip_sleep_sec)
            continue

        # ----- simulate features (stub) -----
        whale_dir = "none"
        whale_score: Optional[float] = None
        if simulate_whale and random.random() < 0.35:
            ws = random.uniform(0.05, 0.60)
            whale_score = float(_clamp(ws))
            # sometimes align, sometimes contra
            if random.random() < 0.65:
                whale_dir = "buy" if side_candidate == "long" else "sell"
            else:
                whale_dir = "sell" if side_candidate == "long" else "buy"

        atr_pct = float(random.uniform(0.004, 0.035))          # 0.004 = %0.4
        spread_pct = float(random.uniform(0.0001, 0.0012))     # 0.0010 = %0.10

        # Liquidity proxy (stub)
        liq_base = random.uniform(0.0, 1.0)
        liq_pen_spread = _clamp(spread_pct / max(max_spread_pct, 1e-9), 0.0, 1.0)
        liq_pen_atr = _clamp(atr_pct / max(max_atr_pct, 1e-9), 0.0, 1.0)
        liq_score = float(
            _clamp(
                liq_base * (1.0 - 0.4 * liq_pen_spread) * (1.0 - 0.2 * liq_pen_atr),
                0.0,
                1.0,
            )
        )

        # micro_score: spread düşük + atr makul => yüksek
        spr_component = _clamp(0.0006 / max(spread_pct, 1e-9), 0.0, 1.0)
        atr_component = _clamp(0.020 / max(atr_pct, 1e-9), 0.0, 1.0)
        micro_score = float(_clamp(0.55 * spr_component + 0.45 * atr_component, 0.0, 1.0))

        # fast_model_score: stub için p
        fast_model_score = float(_clamp(p, 0.0, 1.0))

        # edge signal (0..1): Worker “trade açmaz”, sadece aday kalitesi üretir
        ws_for_edge = float(whale_score) if whale_score is not None else 0.0
        score_edge = float(
            _clamp(
                (w_whale * ws_for_edge) + (w_fast * fast_model_score) + (w_micro * micro_score),
                0.0,
                1.0,
            )
        )

        confidence = float(random.uniform(0.25, 0.95))

        # ----- Worker quality gate (spam kesme) -----
        # sadece side != none iken gate uygula (hold debug basılabilir)
        if side_candidate != "none":
            if spread_pct > max_spread_pct:
                time.sleep(min(0.25, sleep_sec))
                continue
            if atr_pct > max_atr_pct:
                time.sleep(min(0.25, sleep_sec))
                continue
            if confidence < min_conf:
                time.sleep(min(0.25, sleep_sec))
                continue

        # ----- SignalEvent (standard) -----
        ts_utc = _utc_now_iso()
        dedup_key = f"{sym}|{interval}|{side_candidate}"

        evt = {
            # helpful envelope
            "event_id": str(uuid.uuid4()),
            "ts_utc": ts_utc,
            "source": source,  # w1..w8

            # core
            "symbol": sym,
            "interval": interval,
            "side_candidate": side_candidate,  # long|short|none

            # scoring inputs (worker-level)
            "score_edge": float(score_edge),     # 0..1
            "confidence": float(confidence),     # 0..1

            # market / risk features
            "atr_pct": float(atr_pct),           # 0..1 (örn 0.012 = %1.2)
            "spread_pct": float(spread_pct),     # 0..1
            "liq_score": float(liq_score),       # 0..1

            # dedup
            "dedup_key": dedup_key,              # symbol|interval|side_candidate
        }

        if whale_score is not None:
            evt["whale_score"] = float(whale_score)

        if emit_meta:
            evt["meta"] = {
                "kind": "SignalEvent",
                "scanner_mode": scanner_mode,
                "stub": True,
                "whale_dir": whale_dir,
                "p_used": float(p),
                "fast_model_score": float(fast_model_score),
                "micro_score": float(micro_score),
                "weights": {"w_whale": w_whale, "w_fast": w_fast, "w_micro": w_micro},
                "gates": {
                    "min_conf": min_conf,
                    "max_spread_pct": max_spread_pct,
                    "max_atr_pct": max_atr_pct,
                },
            }

        bus.publish_signal(evt)
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
