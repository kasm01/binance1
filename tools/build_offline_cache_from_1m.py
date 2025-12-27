#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
SRC_INTERVAL = os.getenv("SRC_INTERVAL", "1m")
OUT_INTERVALS = [x.strip() for x in os.getenv("OUT_INTERVALS", "2h,4h").split(",") if x.strip()]

SRC = Path("data/offline_cache") / f"{SYMBOL}_{SRC_INTERVAL}_6m.csv"
OUT_DIR = Path("data/offline_cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _infer_timecol(df: pd.DataFrame) -> str:
    for c in ["open_time", "openTime", "timestamp", "datetime", "date"]:
        if c in df.columns:
            return c
    raise ValueError(f"time column not found in columns={list(df.columns)}")

def _to_datetime_index(s: pd.Series) -> pd.DatetimeIndex:
    """
    Accepts:
      - epoch ms / s
      - ISO datetime strings (with or without tz)
      - pandas datetime dtype
    Returns UTC DatetimeIndex
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError("datetime parse produced all NaT")
        return pd.DatetimeIndex(dt)

    # Try numeric (epoch)
    s_num = pd.to_numeric(s, errors="coerce")
    if not s_num.isna().all():
        v = float(s_num.dropna().iloc[0])
        unit = "ms" if v > 1e12 else "s"
        dt = pd.to_datetime(s_num.astype("int64"), unit=unit, utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError("epoch parse produced all NaT")
        return pd.DatetimeIndex(dt)

    # Fallback: parse strings
    dt = pd.to_datetime(s.astype(str), utc=True, errors="coerce")
    if dt.isna().all():
        raise ValueError("string datetime parse produced all NaT")
    return pd.DatetimeIndex(dt)

def _rule(interval: str) -> str:
    m = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
    }
    if interval not in m:
        raise ValueError(f"Unsupported interval={interval}")
    return m[interval]

def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source offline cache not found: {SRC}")

    df = pd.read_csv(SRC)
    timecol = _infer_timecol(df)

    dt = _to_datetime_index(df[timecol])
    df = df.copy()
    df["_dt"] = dt
    df = df.set_index("_dt").sort_index()

    # Coerce numeric columns (except the original time column)
    for c in df.columns:
        if c in (timecol,):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for itv in OUT_INTERVALS:
        out_path = OUT_DIR / f"{SYMBOL}_{itv}_6m.csv"
        rule = _rule(itv)

        agg = {}

        # OHLC
        if "open" in df.columns:  agg["open"] = "first"
        if "high" in df.columns:  agg["high"] = "max"
        if "low"  in df.columns:  agg["low"]  = "min"
        if "close" in df.columns: agg["close"] = "last"

        # Volume sums
        for c in [
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]:
            if c in df.columns:
                agg[c] = "sum"

        # ignore: keep last if exists
        if "ignore" in df.columns:
            agg["ignore"] = "last"

        rs = df.resample(rule, label="left", closed="left").agg(agg)

        # drop empty bars
        need = [c for c in ["open", "high", "low", "close"] if c in rs.columns]
        rs = rs.dropna(subset=need, how="any")

        # open_time & close_time as ISO string (to match your source style)
        idx = rs.index
        try:
            idx_naive = idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            try:
                idx_naive = idx.tz_localize(None)
            except Exception:
                idx_naive = pd.DatetimeIndex(pd.to_datetime(idx)).tz_localize(None)

        close_idx = idx + pd.Timedelta(rule) - pd.Timedelta(milliseconds=1)
        try:
            close_naive = close_idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            try:
                close_naive = close_idx.tz_localize(None)
            except Exception:
                close_naive = pd.DatetimeIndex(pd.to_datetime(close_idx)).tz_localize(None)

        rs.insert(0, "open_time", idx_naive.astype(str))
        rs.insert(1, "close_time", close_naive.astype(str))

        rs.to_csv(out_path, index=False)
        print(f"[OK] wrote {out_path} shape={rs.shape}")

if __name__ == "__main__":
    main()
