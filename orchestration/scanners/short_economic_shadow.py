#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features
from models.short_model_registry import ShortModelRegistry
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.context.entry_context_snapshot import build_entry_context_from_rows

FAPI = "https://fapi.binance.com"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except Exception:
        return int(default)


def _symbols() -> List[str]:
    raw = os.getenv("STRATEGY_SYMBOLS", os.getenv("SHORT_SHADOW_SYMBOLS", os.getenv("SYMBOLS", "LINKUSDT,AVAXUSDT,INJUSDT")))
    out: List[str] = []
    for x in str(raw or "").split(","):
        s = x.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _fetch_klines(session: requests.Session, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    r = session.get(
        f"{FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": int(limit)},
        timeout=8,
    )
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        return pd.DataFrame(columns=KLINE_COLUMNS)
    return pd.DataFrame(rows, columns=KLINE_COLUMNS)


def _completed_only(df: pd.DataFrame, now_ms: Optional[int] = None) -> pd.DataFrame:
    if df.empty:
        return df
    now_ms = int(now_ms or time.time() * 1000)
    z = df.copy()
    if int(float(z.iloc[-1]["close_time"])) >= now_ms:
        z = z.iloc[:-1].copy()
    return z.reset_index(drop=True)


def _confirmation_state(rows_1m: pd.DataFrame, min_reversal: float) -> Tuple[str, Dict[str, float]]:
    if rows_1m is None or len(rows_1m) < 2:
        return "NO_1M_DATA", {}
    prev = rows_1m.iloc[-2]
    curr = rows_1m.iloc[-1]
    po, pc = float(prev.open), float(prev.close)
    co, cc = float(curr.open), float(curr.close)
    vals = {
        "prev_open": po, "prev_close": pc,
        "curr_open": co, "curr_close": cc,
        "reversal_pct": ((co - cc) / co) if co > 0 else 0.0,
        "current_open_time": float(curr.open_time),
    }
    if pc <= po:
        return "PREV_NOT_GREEN", vals
    if cc >= co:
        return "NOT_DOWN", vals
    if vals["reversal_pct"] < float(min_reversal):
        return "WEAK", vals
    return "CONFIRM", vals


def _entry_hazard_snapshot(rows_1m: pd.DataFrame, progress: float) -> Dict[str, Any]:
    """Live-causal entry snapshot from the same 1m response used by confirmation."""
    if rows_1m is None or len(rows_1m) < 16:
        return {"available": False, "reason": "insufficient_1m_rows"}
    try:
        z = rows_1m.copy()
        for c in ["open","high","low","close","quote_asset_volume","taker_buy_quote_volume"]:
            z[c] = pd.to_numeric(z[c], errors="coerce")
        curr = z.iloc[-1]
        prev = z.iloc[:-1].copy()
        if len(prev) < 15:
            return {"available": False, "reason": "insufficient_completed_rows"}
        q = prev["quote_asset_volume"]
        b = prev["taker_buy_quote_volume"]
        net = 2.0*b - q
        def flow(w: int) -> float:
            den = float(q.tail(w).sum())
            return float(net.tail(w).sum()/den) if den > 0 else 0.0
        co, ch, cl, cc = [float(curr[x]) for x in ["open","high","low","close"]]
        cq = float(curr["quote_asset_volume"] or 0.0)
        cb = float(curr["taker_buy_quote_volume"] or 0.0)
        crng = max(ch-cl, 1e-12)
        cflow = float((2.0*cb-cq)/cq) if cq > 0 else 0.0
        med20 = float(q.tail(20).median()) if len(q) >= 20 else float(q.median())
        pace_den = med20 * max(float(progress), 0.03)
        closes = prev["close"].astype(float)
        def ret(w: int) -> float:
            if len(closes) < w+1: return 0.0
            a=float(closes.iloc[-1]); b0=float(closes.iloc[-1-w]); return float(a/b0-1.0) if b0>0 else 0.0
        return {
            "available": True,
            "schema_version": "entry_hazard_snapshot.v1",
            "progress": float(progress),
            "curr_body_pct": float((cc-co)/co) if co>0 else 0.0,
            "curr_range_pct": float((ch-cl)/co) if co>0 else 0.0,
            "curr_close_loc": float((cc-cl)/crng),
            "curr_upper_wick_pct": float((ch-max(co,cc))/co) if co>0 else 0.0,
            "curr_lower_wick_pct": float((min(co,cc)-cl)/co) if co>0 else 0.0,
            "curr_flow_partial": cflow,
            "curr_buy_ratio_partial": float(cb/cq) if cq>0 else 0.0,
            "curr_quote_volume": cq,
            "curr_volume_pace": float(cq/pace_den) if pace_den>0 else 0.0,
            "flow1_prev": flow(1), "flow3_prev": flow(3), "flow5_prev": flow(5), "flow15_prev": flow(15),
            "flow_accel_partial_vs3": float(cflow-flow(3)),
            "flow_accel_partial_vs15": float(cflow-flow(15)),
            "ret1_prev": ret(1), "ret3_prev": ret(3), "ret5_prev": ret(5), "ret15_prev": ret(15),
        }
    except Exception as exc:
        return {"available": False, "reason": "snapshot_exception", "error": str(exc)[:180]}


def main() -> None:
    symbols = _symbols()
    if not symbols:
        raise SystemExit("SHORT_SHADOW: no symbols")

    model_dir = Path(os.getenv(
        "SHORT_SHADOW_MODEL_DIR",
        ROOT / "research/short/models",
    ))
    interval = str(os.getenv("SHORT_SHADOW_INTERVAL", "3m") or "3m").strip()
    registry = ShortModelRegistry(model_dir=model_dir)
    data_limit = max(260, _env_int("SHORT_SHADOW_DATA_LIMIT", 300))
    strategy_id = str(os.getenv("STRATEGY_ID", "SHORT1") or "SHORT1").strip().upper()
    strategy_ns = strategy_id.lower()
    stream = str(os.getenv("STRATEGY_MODEL_SHADOW_STREAM", os.getenv("SHORT_SHADOW_STREAM", f"telemetry:{strategy_ns}:model_shadow")) or f"telemetry:{strategy_ns}:model_shadow")
    maxlen = max(1000, _env_int("SHORT_SHADOW_MAXLEN", 20000))
    min_progress = max(0.0, _env_float("SNIPER_ENTRY_MIN_CANDLE_PROGRESS", 0.03))
    max_progress = min(1.0, _env_float("SNIPER_ENTRY_MAX_CANDLE_PROGRESS", 0.25))
    min_reversal = max(0.0, _env_float("SHORT_CURRENT_REVERSAL_MIN_PCT", 0.0001))
    poll_sec = max(0.25, _env_float("SHORT_SHADOW_POLL_SEC", 1.0))
    once = str(os.getenv("SHORT_SHADOW_ONCE", "0")).strip().lower() in ("1", "true", "yes", "on")

    models: Dict[str, Any] = {}
    for symbol in symbols:
        model = registry.get(symbol, interval, require_promoted=False, require_execution_contract=False)
        if list(model.features) != list(SHORT_ECON_MODEL_FEATURES_V1):
            raise SystemExit(f"SHORT_SHADOW: feature contract mismatch symbol={symbol}")
        models[symbol] = model

    bus = RedisBus()
    if not bus.ping():
        raise SystemExit("SHORT_SHADOW: Redis unavailable")
    session = requests.Session()
    last_close: Dict[str, int] = {}

    print(
        f"[SHORT-SHADOW] started strategy={strategy_id} symbols={symbols} interval={interval} stream={stream} "
        f"sniper={min_progress:.3f}..{max_progress:.3f} min_reversal={min_reversal:.6f} routing=OFF",
        flush=True,
    )

    while True:
        loop_start = time.time()

        # Cost-aware scheduling: only touch Binance near a fresh 3m boundary.
        # Outside the sniper window we sleep locally; no polling HTTP traffic.
        if not once:
            sec_in_3m = time.time() % 180.0
            if sec_in_3m > max_progress * 60.0:
                sleep_to_boundary = 180.0 - sec_in_3m + 0.15
                time.sleep(min(max(0.25, sleep_to_boundary), 30.0))
                continue

        for symbol in symbols:
            try:
                raw_all = _fetch_klines(session, symbol, interval, data_limit + 1)
                raw = _completed_only(raw_all)
                if len(raw) < 241:
                    continue
                close_id = int(float(raw.iloc[-1].close_time))
                if not once and last_close.get(symbol) == close_id:
                    continue
                last_close[symbol] = close_id

                model = models[symbol]
                score = model.predict_score(raw.tail(data_limit))
                if score is None:
                    continue
                feat_df = build_short_economic_features(raw.tail(data_limit))
                feat_row = feat_df.iloc[-1]
                features = {k: float(feat_row[k]) for k in SHORT_ECON_MODEL_FEATURES_V1}

                now = time.time()
                progress = (now % 60.0) / 60.0
                confirmation = "WINDOW_MISSED"
                confirm_data: Dict[str, float] = {}

                if progress < min_progress and not once:
                    sleep_for = min_progress * 60.0 - (time.time() % 60.0)
                    if sleep_for > 0:
                        time.sleep(min(sleep_for + 0.05, 20.0))
                    progress = (time.time() % 60.0) / 60.0

                entry_market_context: Dict[str, Any] = {"available": False, "reason": "confirmation_unavailable"}
                entry_hazard_snapshot: Dict[str, Any] = {"available": False, "reason": "confirmation_unavailable"}
                money_flow_60m = None
                if min_progress <= progress <= max_progress:
                    # Same single Binance request as before, but retain enough 1m rows
                    # for V30 60m flow + exact entry-time 15m/OI context.
                    # Request COUNT is unchanged; only response payload is slightly larger.
                    one = _fetch_klines(session, symbol, "1m", 65)
                    confirmation, confirm_data = _confirmation_state(one, min_reversal)
                    entry_hazard_snapshot = _entry_hazard_snapshot(one, progress)
                    try:
                        confirm_open_ms = int(float(confirm_data.get("current_open_time") or 0.0))
                        if confirm_open_ms > 0:
                            completed = one[pd.to_numeric(one["open_time"], errors="coerce") < confirm_open_ms].copy()
                            if len(completed) >= 60:
                                q = pd.to_numeric(completed["quote_asset_volume"], errors="coerce").tail(60)
                                b = pd.to_numeric(completed["taker_buy_quote_volume"], errors="coerce").tail(60)
                                den = float(q.sum())
                                if den > 0.0:
                                    money_flow_60m = float((2.0 * b.sum() - q.sum()) / den)
                            entry_market_context = build_entry_context_from_rows(
                                redis_client=bus.r,
                                symbol=symbol,
                                rows=one.values.tolist(),
                                confirmation_open_ms=confirm_open_ms,
                            )
                    except Exception as exc:
                        entry_market_context = {
                            "available": False,
                            "reason": "v30_context_error",
                            "error": str(exc)[:180],
                        }
                elif progress < min_progress:
                    confirmation = "WINDOW_EARLY"

                record: Dict[str, Any] = {
                    "event_id": str(uuid.uuid4()),
                    "strategy_id": strategy_id,
                    "ts": float(time.time()),
                    "symbol": symbol,
                    "interval": interval,
                    "completed_close_time": close_id,
                    "model_provider": str(model.meta.get("provider") or "short_economic_logistic_v1"),
                    "model_target": str(model.meta.get("target") or ""),
                    "model_feature_hash": str(model.meta.get("feature_hash") or ""),
                    "score_semantics": "short_economic_probability",
                    "economic_probability": float(score),
                    "reference_threshold": float(model.meta.get("reference_probability_threshold_full_train", 0.0) or 0.0),
                    "sniper_progress": float(progress),
                    "sniper_min": float(min_progress),
                    "sniper_max": float(max_progress),
                    "confirmation_state": confirmation,
                    "routing": "OFF",
                    "features": features,
                    "confirmation": confirm_data,
                    "money_flow_60m": money_flow_60m,
                    "entry_market_context": entry_market_context,
                    "entry_hazard_snapshot": entry_hazard_snapshot,
                }
                sid = bus.xadd_json(stream, record, maxlen=maxlen)
                print(
                    f"[SHORT-SHADOW] symbol={symbol} p={score:.5f} progress={progress:.3f} "
                    f"confirm={confirmation} stream_id={sid or '-'} routing=OFF",
                    flush=True,
                )
            except Exception as exc:
                print(f"[SHORT-SHADOW][WARN] symbol={symbol} err={str(exc)[:220]}", flush=True)

        if once:
            break
        elapsed = time.time() - loop_start
        time.sleep(max(0.25, poll_sec - elapsed))


if __name__ == "__main__":
    main()
