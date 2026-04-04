import os
import math
import json
from typing import Dict, List, Tuple

import pandas as pd

try:
    from binance.client import Client
except Exception:
    Client = None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return int(default)


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty or len(df) < period + 2:
        return 0.0

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean().iloc[-1]
    last_close = close.iloc[-1]

    if not math.isfinite(_safe_float(atr)) or not math.isfinite(_safe_float(last_close)):
        return 0.0
    if last_close <= 0:
        return 0.0

    return float(atr / last_close)


def normalize(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    x = (float(val) - float(lo)) / (float(hi) - float(lo))
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def load_klines(client: Client, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_book_ticker(client: Client, symbol: str) -> Tuple[float, float]:
    try:
        x = client.futures_orderbook_ticker(symbol=symbol)
        bid = _safe_float(x.get("bidPrice"))
        ask = _safe_float(x.get("askPrice"))
        return bid, ask
    except Exception:
        return 0.0, 0.0


def spread_pct_from_bid_ask(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0 or ask < bid:
        return 1.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 1.0
    return float((ask - bid) / mid)


def build_client() -> Client:
    if Client is None:
        raise RuntimeError("python-binance Client import failed")

    key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_TESTNET_API_KEY") or ""
    sec = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_TESTNET_API_SECRET") or ""
    return Client(key, sec)


def rank_symbols() -> List[Dict]:
    symbols = [s.strip().upper() for s in (os.getenv("VOLSEL_SYMBOLS") or "").split(",") if s.strip()]
    interval = os.getenv("VOLSEL_INTERVAL", "5m")
    limit = _env_int("VOLSEL_KLINE_LIMIT", 120)

    w_atr = _env_float("VOLSEL_WEIGHT_ATR", 0.45)
    w_spread = _env_float("VOLSEL_WEIGHT_SPREAD", 0.20)
    w_liq = _env_float("VOLSEL_WEIGHT_LIQ", 0.20)
    w_whale = _env_float("VOLSEL_WEIGHT_WHALE", 0.15)

    min_atr_pct = _env_float("VOLSEL_MIN_ATR_PCT", 0.003)
    max_spread_pct = _env_float("VOLSEL_MAX_SPREAD_PCT", 0.0015)
    min_liq_score = _env_float("VOLSEL_MIN_LIQ_SCORE", 0.10)
    min_whale_score = _env_float("VOLSEL_MIN_WHALE_SCORE", 0.00)

    client = build_client()

    rows: List[Dict] = []
    for sym in symbols:
        try:
            df = load_klines(client, sym, interval, limit)
            if df.empty:
                continue

            atr_pct = compute_atr_pct(df)
            bid, ask = fetch_book_ticker(client, sym)
            spread_pct = spread_pct_from_bid_ask(bid, ask)

            last_quote_vol = _safe_float(df["quote_asset_volume"].tail(20).mean(), 0.0)
            liq_score = 0.0
            if last_quote_vol > 0:
                liq_score = min(1.0, last_quote_vol / 5_000_000.0)

            whale_score = 0.0

            if atr_pct < min_atr_pct:
                continue
            if spread_pct > max_spread_pct:
                continue
            if liq_score < min_liq_score:
                continue
            if whale_score < min_whale_score:
                whale_score = 0.0

            atr_norm = normalize(atr_pct, min_atr_pct, 0.03)
            spread_norm = 1.0 - normalize(spread_pct, 0.0, max_spread_pct)
            liq_norm = normalize(liq_score, min_liq_score, 1.0)
            whale_norm = normalize(whale_score, 0.0, 1.0)

            total = (
                w_atr * atr_norm
                + w_spread * spread_norm
                + w_liq * liq_norm
                + w_whale * whale_norm
            )

            rows.append({
                "symbol": sym,
                "atr_pct": round(atr_pct, 6),
                "spread_pct": round(spread_pct, 6),
                "liq_score": round(liq_score, 6),
                "whale_score": round(whale_score, 6),
                "score": round(float(total), 6),
            })
        except Exception:
            continue

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def main() -> None:
    rows = rank_symbols()
    pick_count = _env_int("VOLSEL_PICK_COUNT", 12)
    min_pick_count = _env_int("VOLSEL_MIN_PICK_COUNT", 8)
    out_file = os.getenv("VOLSEL_OUT_FILE", "/tmp/binance1_volsel_symbols.txt")

    picked = rows[:pick_count]
    if len(picked) < min_pick_count:
        fallback = [s.strip().upper() for s in (os.getenv("VOLSEL_SYMBOLS") or "").split(",") if s.strip()]
        picked_symbols = fallback[:max(min_pick_count, pick_count)]
    else:
        picked_symbols = [r["symbol"] for r in picked]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(",".join(picked_symbols) + "\n")

    print(json.dumps({
        "picked_symbols": picked_symbols,
        "rows": picked[:pick_count],
        "out_file": out_file,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
