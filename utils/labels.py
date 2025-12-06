"""
utils/labels.py

Label üretim fonksiyonları.

Temel mantık:
    future_close = df["close"].shift(-horizon)
    ret = future_close / df["close"] - 1.0
    y = (ret > 0.0).astype(int)
"""

from typing import Union

import numpy as np
import pandas as pd


def build_labels(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    Gelecekteki 'horizon' bar sonrası fiyata göre 0/1 label üretir.

    Label tanımı:
        future_close = close.shift(-horizon)
        ret = future_close / close - 1.0
        y = 1  if ret > 0
            0  otherwise

    Son horizon bar için future_close NaN olacağı için otomatik olarak atılır.
    Dönen seri index'i resetlenmemiş olabilir; kullanan taraf genelde
    feat_df ile hizalama yaparken zaten yeniden index ayarlayacak.
    """
    if "close" not in df.columns:
        raise ValueError("build_labels: 'close' kolonu df içinde bulunamadı.")

    close = df["close"].astype(float)

    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0

    y = (ret > 0.0).astype(int)

    # Gelecekteki fiyatı olmayan (NaN) barları düş
    mask = future_close.notna() & close.notna()
    y = y[mask]

    return y


def build_signed_returns(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """
    İsteğe bağlı: label yanında gerçek getiriyi de görmek istersen.

    ret = future_close / close - 1.0
    """
    if "close" not in df.columns:
        raise ValueError("build_signed_returns: 'close' kolonu df içinde yok.")

    close = df["close"].astype(float)
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    return ret

