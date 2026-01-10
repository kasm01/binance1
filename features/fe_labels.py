# features/fe_labels.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_features(df_raw: pd.DataFrame, interval: str | None = None) -> pd.DataFrame:
    """
    22-feature şeması (meta ile uyumlu):
    - 12 kline kolonu:
      open_time, open, high, low, close, volume, close_time,
      quote_asset_volume, number_of_trades,
      taker_buy_base_volume, taker_buy_quote_volume, ignore
    - +10 engineered:
      hl_range, oc_change, return_1, return_3, return_5,
      ma_5, ma_10, ma_20, vol_10, dummy_extra
    """
    df = df_raw.copy()

    # Beklenen kolonlar yoksa hata yerine en azından mevcutlarıyla ilerle
    # (offline cache CSV'lerinde genelde tam set var)
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    num_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume",
        "ignore", "open_time", "close_time",
    ]
    df = _to_numeric(df, num_cols)

    # ---- engineered features ----
    # high-low aralığı
    if {"high", "low"}.issubset(df.columns):
        df["hl_range"] = df["high"] - df["low"]
    else:
        df["hl_range"] = np.nan

    # open->close değişim
    if {"open", "close"}.issubset(df.columns):
        df["oc_change"] = (df["close"] - df["open"]) / (df["open"].replace(0, np.nan))
    else:
        df["oc_change"] = np.nan

    # returns
    if "close" in df.columns:
        close = df["close"].replace(0, np.nan)
        df["return_1"] = close.pct_change(1)
        df["return_3"] = close.pct_change(3)
        df["return_5"] = close.pct_change(5)
    else:
        df["return_1"] = np.nan
        df["return_3"] = np.nan
        df["return_5"] = np.nan

    # moving averages (close)
    if "close" in df.columns:
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
    else:
        df["ma_5"] = np.nan
        df["ma_10"] = np.nan
        df["ma_20"] = np.nan

    # volume rolling
    if "volume" in df.columns:
        df["vol_10"] = df["volume"].rolling(10).mean()
    else:
        df["vol_10"] = np.nan

    # meta'da görülen dummy_extra (schema sabitlemek için)
    df["dummy_extra"] = 0.0

    # NaN'ler: model tarafında genelde drop/fill yapılıyor ama burada minimal fill yapalım
    # (özellikle rolling başları)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(0.0)

    return df


def build_labels(feat_df: pd.DataFrame, horizon: int = 1, thr: float = 0.0) -> pd.Series:
    """
    Binary label:
      y[t] = 1 if (close[t+horizon] / close[t] - 1) > thr else 0
    thr: eşik (örn 0.0005)
    """
    if "close" not in feat_df.columns:
        raise ValueError("build_labels: 'close' column missing")

    close = pd.to_numeric(feat_df["close"], errors="coerce").replace(0, np.nan)
    fut = close.shift(-horizon)

    ret = (fut / close) - 1.0
    y = (ret > float(thr)).astype(int)

    # Son horizon satırın geleceği yok; yine de 0'a çekelim (caller zaten genelde kırpıyor)
    y = y.fillna(0).astype(int)
    return y
