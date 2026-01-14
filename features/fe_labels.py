# features/fe_labels.py
from __future__ import annotations
import os

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
    df = df.ffill().fillna(0.0)


    # --- feature normalization for SGD (train/preprocess match) ---
    # scaler istatistikleri: fiyat/MA log, hl_range relatif, time epoch seconds, volume RAW
    
    # 1) time: ms -> sec (ms geliyorsa)
    try:
        for _tc in ("open_time","close_time"):
            if _tc in df.columns:
                _t = pd.to_numeric(df[_tc], errors="coerce").astype(float)
                if float(np.nanmax(_t)) > 1.0e12:
                    _t = _t / 1000.0
                df[_tc] = _t
    except Exception:
        pass
    
    # 2) hl_range: absolute yerine relatif range
    try:
        if "high" in df.columns and "low" in df.columns:
            _den = pd.to_numeric(df["close"] if "close" in df.columns else df["low"], errors="coerce").astype(float)
            _den = np.clip(_den, 1e-12, None)
            df["hl_range"] = (pd.to_numeric(df["high"], errors="coerce").astype(float) - pd.to_numeric(df["low"], errors="coerce").astype(float)) / _den
    except Exception:
        pass
    
    # 3) price/log: OHLC + MA kolonlarını log ölçeğe al (BT_PRICE_LOG=1)
    try:
        _price_log = str(os.getenv("BT_PRICE_LOG","1")).lower() in ("1","true","yes","on")
    except Exception:
        _price_log = True
    if _price_log:
        for _c in ("open","high","low","close","ma_5","ma_10","ma_20"):
            if _c in df.columns:
                _v = pd.to_numeric(df[_c], errors="coerce").astype(float)
                df[_c] = np.log(np.clip(_v, 1e-12, None))

    # --- time normalization (prevent huge timestamp scale from saturating SGD) ---
    # BT_TIME_NORM=1 (default): open_time/close_time -> relative minutes from first bar
    try:
        _time_norm = str(os.getenv('BT_TIME_NORM', '1')).lower() in ('1','true','yes','on')
    except Exception:
        _time_norm = True
    
    if _time_norm:
        try:
            def _to_minutes(arr: pd.Series) -> pd.Series:
                a = pd.to_numeric(arr, errors="coerce").astype(float)
                mx = float(np.nanmax(a)) if len(a) else 0.0
                # ms gibi görünüyorsa: dakika = ms / 60000
                if mx > 1.0e12:
                    return a / 60000.0
                # saniye gibi görünüyorsa: dakika = sec / 60
                return a / 60.0

            if "open_time" in df.columns:
                ot = pd.to_numeric(df["open_time"], errors="coerce").astype(float)
                df["open_time"] = _to_minutes(ot - float(ot.iloc[0]))

            if "close_time" in df.columns:
                ct = pd.to_numeric(df["close_time"], errors="coerce").astype(float)
                df["close_time"] = _to_minutes(ct - float(ct.iloc[0]))

            df = df.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        except Exception:
            pass


    # --- optional log1p scaling for very large non-negative features (SGD saturation fix) ---
    # BT_LOG1P_FEATURES=1: apply log1p to selected large-scale columns
    try:
        _log1p = str(os.getenv('BT_LOG1P_FEATURES', '1')).lower() in ('1','true','yes','on')
    except Exception:
        _log1p = True
    
    if _log1p:
        try:
            cols = [
                'volume','quote_asset_volume','number_of_trades',
                'taker_buy_base_volume','taker_buy_quote_volume'
            ]
            for c in cols:
                if c in df.columns:
                    v = pd.to_numeric(df[c], errors='coerce').astype(float)
                    # yalnızca >=0 için log1p
                    if (v >= 0).all():
                        df[c] = np.log1p(v)
            df = df.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
        except Exception:
            pass

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
