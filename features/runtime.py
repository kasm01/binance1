from __future__ import annotations

import pandas as pd


def build_runtime_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Single runtime feature authority used by live model inference."""
    df = raw_df.copy()

    raw_cols_12 = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for c in raw_cols_12:
        if c not in df.columns:
            df[c] = 0

    for col in ["open_time", "close_time"]:
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    dt = pd.to_datetime(df[col], unit="ms", utc=True)
                else:
                    dt = pd.to_datetime(df[col], utc=True)
                df[col] = (dt.astype("int64") / 1e9).astype(float)
            except Exception:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_like = ["number_of_trades", "ignore"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ignore" not in df.columns:
        df["ignore"] = 0.0

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    close = df["close"].astype(float)
    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)
    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    return df.ffill().bfill().fillna(0.0)
