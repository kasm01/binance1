#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from app_paths import MODELS_DIR
from models.short_model_registry import ShortModelRegistry
from orchestration.event_bus.redis_bus import RedisBus

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
FAPI = "https://fapi.binance.com"


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
    raw = os.getenv("WORKER_SYMBOLS", os.getenv("SYMBOLS", ""))
    out = []
    for x in str(raw or "").split(","):
        s = x.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _money_flow_metrics(raw: pd.DataFrame) -> Dict[str, float]:
    """Compute causal taker-flow ratios from completed Binance futures klines.

    No extra market-data request is made. For a 3m model stream:
      - 15m = last 5 completed bars
      - 60m = last 20 completed bars
    The ratio is (aggressive_buy_quote - aggressive_sell_quote) / total_quote.
    """
    if not isinstance(raw, pd.DataFrame) or len(raw) < 20:
        return {"money_flow_15m": 0.0, "money_flow_60m": 0.0}
    try:
        q = pd.to_numeric(raw["quote_asset_volume"], errors="coerce").astype(float)
        buy = pd.to_numeric(raw["taker_buy_quote_volume"], errors="coerce").astype(float)
        net = (2.0 * buy) - q

        def ratio(n: int) -> float:
            qsum = float(q.tail(n).sum())
            if not (qsum > 0.0):
                return 0.0
            return float(net.tail(n).sum() / qsum)

        return {
            "money_flow_15m": ratio(5),
            "money_flow_60m": ratio(20),
        }
    except Exception:
        return {"money_flow_15m": 0.0, "money_flow_60m": 0.0}


def _price_response_metrics(raw: pd.DataFrame) -> Dict[str, float]:
    """Compute completed-bar price response with no additional API request.

    For the production 3m model stream, 15m = five completed 3m bars.
    The value is causal because _fetch_klines removes the forming candle.
    """
    if not isinstance(raw, pd.DataFrame) or len(raw) < 6:
        return {"price_return_15m": 0.0}
    try:
        close = pd.to_numeric(raw["close"], errors="coerce").astype(float)
        latest = float(close.iloc[-1])
        prior = float(close.iloc[-6])
        if not (latest > 0.0 and prior > 0.0):
            return {"price_return_15m": 0.0}
        return {"price_return_15m": float(latest / prior - 1.0)}
    except Exception:
        return {"price_return_15m": 0.0}


def _fetch_klines(session: requests.Session, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    r = session.get(
        f"{FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": int(limit)},
        timeout=8,
    )
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or len(rows) < 30:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    now_ms = int(time.time() * 1000)
    # Binance generally includes the currently-forming candle. MODEL operates on completed bars only.
    if len(df) and int(float(df.iloc[-1]["close_time"])) >= now_ms:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)




def _interval_to_seconds(interval: str) -> int:
    v = str(interval or "").strip().lower()
    if len(v) < 2:
        return 0
    try:
        n = int(v[:-1])
    except Exception:
        return 0
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(v[-1], 0)
    return max(0, n * mult)

def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    try:
        if len(df) < period + 2:
            return 0.0
        hi = pd.to_numeric(df["high"], errors="coerce").astype(float)
        lo = pd.to_numeric(df["low"], errors="coerce").astype(float)
        cl = pd.to_numeric(df["close"], errors="coerce").astype(float)
        pc = cl.shift(1)
        tr = pd.concat([(hi - lo), (hi - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        px = float(cl.iloc[-1])
        return max(0.0, atr / px) if px > 0 else 0.0
    except Exception:
        return 0.0


def main() -> None:
    worker_id = str(os.getenv("WORKER_ID", "w1") or "w1")
    symbols = _symbols()
    interval = str(os.getenv("WORKER_INTERVAL", os.getenv("INTERVAL", "3m")) or "3m").strip()
    poll_sec = max(2.0, _env_float("MODEL_WORKER_POLL_SEC", 10.0))
    data_limit = max(260, _env_int("MODEL_WORKER_DATA_LIMIT", 500))
    boundary_window_sec = max(1.0, _env_float("MODEL_WORKER_BOUNDARY_WINDOW_SEC", 5.0))
    interval_sec = _interval_to_seconds(interval)

    if not symbols:
        raise SystemExit("MODEL_WORKER: no symbols configured")

    bus = RedisBus()
    if not bus.ping():
        raise SystemExit("MODEL_WORKER: Redis unavailable")

    registry = ShortModelRegistry(model_dir=MODELS_DIR)
    session = requests.Session()
    last_completed_close: Dict[str, int] = {}
    warned_missing_model: set[str] = set()

    print(
        f"[ModelWorker] started id={worker_id} interval={interval} symbols={symbols} "
        f"poll={poll_sec}s provider=short_economic_coin promoted_required=1",
        flush=True,
    )

    while True:
        loop_started = time.time()

        # Cost-aware completed-bar scheduling: outside the small post-boundary
        # window, sleep locally and make no Binance kline HTTP request.
        if interval_sec > 0:
            sec_in_bar = time.time() % float(interval_sec)
            if sec_in_bar > boundary_window_sec:
                sleep_to_boundary = float(interval_sec) - sec_in_bar + 0.20
                time.sleep(min(max(0.5, sleep_to_boundary), 30.0))
                continue

        for symbol in symbols:
            try:
                # Validate the promoted coin-specific artifact before any market-data HTTP call.
                try:
                    model = registry.get(symbol, interval, require_promoted=True)
                except (FileNotFoundError, PermissionError, ValueError) as exc:
                    if symbol not in warned_missing_model:
                        warned_missing_model.add(symbol)
                        print(
                            f"[ModelWorker][NO-MODEL] symbol={symbol} interval={interval} "
                            f"reason={str(exc)[:180]} fail_closed=1",
                            flush=True,
                        )
                    continue

                raw = _fetch_klines(session, symbol, interval, data_limit + 1)
                if raw.empty or len(raw) < 241:
                    continue

                close_id = int(float(raw.iloc[-1]["close_time"]))
                if last_completed_close.get(symbol) == close_id:
                    continue

                short_score = model.predict_score(raw.tail(data_limit))
                last_completed_close[symbol] = close_id
                if short_score is None:
                    print(
                        f"[ModelWorker][NO-SCORE] symbol={symbol} interval={interval} fail_closed=1",
                        flush=True,
                    )
                    continue

                price = float(pd.to_numeric(raw.iloc[-1]["close"], errors="coerce") or 0.0)
                if price <= 0.0:
                    continue

                atr_pct = _atr_pct(raw)
                money_flow = _money_flow_metrics(raw)
                price_response = _price_response_metrics(raw)
                meta = dict(model.meta or {})
                evt: Dict[str, Any] = {
                    "event_id": str(uuid.uuid4()),
                    "producer_id": worker_id,
                    "source": "SHORT_ECONOMIC_MODEL",
                    "model_provider": str(meta.get("provider") or "short_economic_logistic_coin_v1"),
                    "symbol": symbol,
                    "interval": interval,
                    "side_candidate": "short",
                    "price": price,
                    "model_score": float(short_score),
                    "atr_pct": atr_pct,
                    "spread_pct": 0.0,
                    "liq_score": 0.0,
                    "money_flow_15m": float(money_flow.get("money_flow_15m", 0.0)),
                    "money_flow_60m": float(money_flow.get("money_flow_60m", 0.0)),
                    "price_return_15m": float(price_response.get("price_return_15m", 0.0)),
                    "whale_dir": "none",
                    "whale_score": 0.0,
                    "dedup_key": f"{symbol}|{interval}|short|{close_id}",
                    "meta": {
                        "kind": "SignalEvent",
                        "model_provider": str(meta.get("provider") or "short_economic_logistic_coin_v1"),
                        "model_score": float(short_score),
                        "score_semantics": str(meta.get("score_semantics") or "short_economic_probability"),
                        "model_symbol": str(meta.get("symbol") or symbol),
                        "model_version": str(meta.get("model_version") or meta.get("provider") or ""),
                        "feature_hash": str(meta.get("feature_hash") or ""),
                        "completed_close_time": close_id,
                        "money_flow_15m": float(money_flow.get("money_flow_15m", 0.0)),
                        "money_flow_60m": float(money_flow.get("money_flow_60m", 0.0)),
                        "price_return_15m": float(price_response.get("price_return_15m", 0.0)),
                    },
                }
                sid = bus.publish_signal(evt)
                if sid:
                    print(
                        f"[ModelWorker][SIGNAL] id={worker_id} symbol={symbol} "
                        f"short_score={short_score:.5f} provider={evt['model_provider']} stream_id={sid}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[ModelWorker][WARN] symbol={symbol} err={str(exc)[:220]}", flush=True)

        elapsed = time.time() - loop_started
        time.sleep(max(0.5, poll_sec - elapsed))


if __name__ == "__main__":
    main()
