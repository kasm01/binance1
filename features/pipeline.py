from __future__ import annotations

from typing import List, Dict
import numpy as np
import pandas as pd

# === RAW kline kolonları (Binance 12) ===
RAW_KLINE_COLS_12: List[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# === Model meta'sında sık kullanılan şema (20) ===
# Not: Senin logdaki meta örneğinde open_time/close_time yoktu ve dummy_extra vardı -> 20.
FEATURE_SCHEMA_20: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "hl_range",
    "oc_change",
    "return_1",
    "return_3",
    "return_5",
    "ma_5",
    "ma_10",
    "ma_20",
    "vol_10",
    "dummy_extra",
]

# Eğer bazı yerlerde timestamp'li 22’lik şema gerekiyorsa (opsiyonel)
FEATURE_SCHEMA_22: List[str] = [
    "open_time",
    *FEATURE_SCHEMA_20[:],  # open..dummy_extra
    "close_time",
]
# Yukarıdaki satır open_time + (20 şema) + close_time -> 22 eder ama sırası karışabilir.
# Eğer timestamp’li şemayı gerçekten kullanıyorsan aşağıdaki gibi net bir 22 tanımı daha güvenlidir:
FEATURE_SCHEMA_22 = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "hl_range",
    "oc_change",
    "return_1",
    "return_3",
    "return_5",
    "ma_5",
    "ma_10",
    "ma_20",
    "vol_10",
    "dummy_extra",
]

# --- SGD safe schema (NO timestamps) ---
SGD_SCHEMA_NO_TIME: List[str] = FEATURE_SCHEMA_20[:]  # SGD/LSTM genelde bunu bekliyor

# Binance REST bazı isimleri farklı verebiliyor (senin logda da bu vardı)
ALIASES: Dict[str, str] = {
    "taker_buy_base_asset_volume": "taker_buy_base_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
    # bazı kodlarda "taker_buy_base_volume" yerine farklı isimler çıkarsa buraya eklenir
}


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    - alias kolonlarını düzeltir
    - temel numeric cast yapar
    - LIVE'ta kritik kolonlar yoksa 0 ile tamamlar (özellikle taker/ignore)
    """
    df = df.copy()

    # Alias fix
    for src, dst in ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # RAW kline kolonlarını garantiye al (LIVE/PUBLIC her zaman 12 döndürür ama güvenlik)
    for c in RAW_KLINE_COLS_12:
        if c not in df.columns:
            df[c] = 0

    # open_time/close_time int64
    for tcol in ("open_time", "close_time"):
        if tcol in df.columns:
            df[tcol] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype("int64")

    # int kolonlar
    for icol in ("number_of_trades", "ignore"):
        if icol in df.columns:
            df[icol] = pd.to_numeric(df[icol], errors="coerce").fillna(0).astype("int64")

    # float kolonlar
    for fcol in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ):
        if fcol in df.columns:
            df[fcol] = pd.to_numeric(df[fcol], errors="coerce")

    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer edilen kolonlar:
    hl_range, oc_change, return_1/3/5, ma_5/10/20, vol_10, dummy_extra
    """
    df = df.copy()

    # close/open/high/low/volume numeric garanti
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    close = df["close"].astype(float)

    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)

    # returns
    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    # moving avgs
    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()

    # volume avg
    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    # model şeması bunu istiyor -> garanti
    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    return df


def _align_to_schema(df_feat: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
    """
    Model meta'daki schema neyse:
    - eksik kolonları 0 ile ekler
    - fazlaları atar
    - sıralamayı schema sırasına çeker
    - numeric'e zorlar
    """
    df_feat = df_feat.copy()

    for c in schema:
        if c not in df_feat.columns:
            df_feat[c] = 0.0

    X = df_feat[schema].copy()

    # her şeyi numeric'e zorla
    for c in schema:
        s = X[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            X[c] = (s.astype("int64") / 1e9).astype(float)
        else:
            X[c] = pd.to_numeric(s, errors="coerce")

    # inf/nan temizle
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill().fillna(0.0)

    return X


def make_matrix(df: pd.DataFrame, schema: List[str] | None = None) -> np.ndarray:
    """
    Default schema: FEATURE_SCHEMA_20
    (Model meta'nın beklediği şema çoğunlukla bu olduğu için mismatch'i çözer.)
    """
    if schema is None:
        schema = FEATURE_SCHEMA_20

    df = _ensure_cols(df)
    df = _engineer(df)

    X = _align_to_schema(df, schema)
    return X.to_numpy(dtype=float, copy=False)


def make_matrix_sgd(df: pd.DataFrame) -> np.ndarray:
    """SGD/LSTM için güvenli feature matrix: meta genelde 20 feature bekler."""
    return make_matrix(df, schema=SGD_SCHEMA_NO_TIME)

