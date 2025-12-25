from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd

# 5m meta şeman (22)
FEATURE_SCHEMA_22: List[str] = [
    "open_time","open","high","low","close","volume","close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore",
    "hl_range","oc_change","return_1","return_3","return_5","ma_5","ma_10","ma_20","vol_10","dummy_extra"
]

# --- SGD safe schema (NO timestamps) ---
SGD_SCHEMA_NO_TIME: List[str] = [
    c for c in FEATURE_SCHEMA_22 if c not in ('open_time','close_time')
]


ALIASES = {
    "taker_buy_base_asset_volume": "taker_buy_base_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
}

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # alias fix
    for src, dst in ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    # numeric cast
    for c in df.columns:
        if c in ("open_time","close_time"):
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    return df

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # basic engineered features
    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)

    close = df["close"].astype(float)
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    df["ma_5"]  = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()

    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    return df

def make_matrix(df: pd.DataFrame, schema: List[str] = FEATURE_SCHEMA_22) -> np.ndarray:
    df = _ensure_cols(df)
    df = _engineer(df)

    # ensure all schema columns exist
    for c in schema:
        if c not in df.columns:
            df[c] = 0.0

    X = df[schema].copy()

    # replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0.0)

    # === NUMERIC_COERCE_BEFORE_TONUMPY (auto) ===
    # X içinde string/datetime kalırsa to_numpy(float) patlar -> burada garantiye al
    try:
        import pandas as _pd
        for _c in list(X.columns):
            _s = X[_c]
            # datetime dtype -> epoch seconds
            if _pd.api.types.is_datetime64_any_dtype(_s):
                X[_c] = _s.astype("int64") / 1e9
                continue
            # object/string -> try datetime parse, else numeric coerce
            if _pd.api.types.is_object_dtype(_s) or _pd.api.types.is_string_dtype(_s):
                _dt = _pd.to_datetime(_s, errors='coerce', utc=True)
                if _dt.notna().any():
                    X[_c] = _dt.astype("int64") / 1e9
                else:
                    X[_c] = _pd.to_numeric(_s, errors='coerce')
            else:
                X[_c] = _pd.to_numeric(_s, errors='coerce')
        # NaN cleanup
        X = X.ffill().bfill().fillna(0.0)
    except Exception:
        pass

    return X.to_numpy(dtype=float, copy=False)


def make_matrix_sgd(df: pd.DataFrame) -> np.ndarray:
    """SGD için güvenli feature matrix: open_time/close_time yok."""
    return make_matrix(df, schema=SGD_SCHEMA_NO_TIME)
