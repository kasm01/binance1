import os
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from models.hybrid_inference import HybridModel

# ------------------------------------------------------------
# Ayarlar
# ------------------------------------------------------------
CSV_PATH = "data/offline_cache/BTCUSDT_5m_6m.csv"
INTERVAL = "5m"
HORIZON = 1  # 1 bar sonra fiyat yönü

LONG_THR_GRID = np.linspace(0.50, 0.70, 9)   # [0.50, 0.525, ..., 0.70]
SHORT_THR_GRID = np.linspace(0.30, 0.50, 9)  # [0.30, 0.325, ..., 0.50]


# ------------------------------------------------------------
# Feature builder (main.py ile uyumlu)
# ------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Temel fiyat/volume feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    # Eğitime uyum için ekstra dummy kolon
    df["dummy_extra"] = 0.0

    # Zaman kolonlarını saniyeye çevir (HybridModel schema ile uyumlu)
    for col in ["open_time", "close_time"]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt = df[col]
        else:
            # Binance tarzı ms ise:
            # burada raw_df zaten ms ise, to_datetime kullan
            dt = pd.to_datetime(df[col], unit="ms", errors="coerce")
        df[col] = dt.view("int64") / 1e9

    # NA'leri temizle
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df


# ------------------------------------------------------------
# Label generator
# ------------------------------------------------------------
def make_labels(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    y = (ret > 0).astype(int)
    """
    close = df["close"]
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    y = (ret > 0.0).astype(int)

    return y, ret


# ------------------------------------------------------------
# Basit PnL backtest
# ------------------------------------------------------------
def run_pnl_backtest(
    p: np.ndarray,
    ret: np.ndarray,
    long_thr: float,
    short_thr: float,
) -> dict:
    """
    p: model skoru (0-1)
    ret: future return (ör. 1 bar sonrası close/close - 1)
    """
    # Pozisyon kuralları
    # p > long_thr  -> +1 (long)
    # p < short_thr -> -1 (short)
    # arası         -> 0  (flat)
    pos = np.where(p > long_thr, 1, np.where(p < short_thr, -1, 0))

    trades_mask = pos != 0
    if trades_mask.sum() == 0:
        return {
            "total_ret": 0.0,
            "n_trades": 0,
            "win_rate": 0.0,
            "max_dd": 0.0,
        }

    pnl = pos * ret
    equity = pnl.cumsum()

    # NumPy ile max drawdown
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd = float(dd.max())

    total_ret = float(equity[-1])
    win_rate = float((pnl[trades_mask] > 0).mean())

    return {
        "total_ret": total_ret,
        "n_trades": int(trades_mask.sum()),
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


# ------------------------------------------------------------
# Ana akış
# ------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print(f"5m Hybrid Backtest + Threshold Sweep | CSV: {CSV_PATH} | horizon={HORIZON}")
    print("=" * 80)

    # 1) Veri yükle
    raw_df = pd.read_csv(CSV_PATH)
    print(f"raw_df shape: {raw_df.shape}")
    print(f"raw_df columns: {raw_df.columns.tolist()}")

    # 2) Feature ve label üret
    feat_df = build_features(raw_df)
    print(f"feat_df shape: {feat_df.shape}")
    print(f"feat_df columns: {feat_df.columns.tolist()}")

    y, ret = make_labels(feat_df, HORIZON)

    # Son HORIZON barın future_close'u yok, bunu at
    mask_valid = ret.notna()
    feat_df = feat_df[mask_valid].reset_index(drop=True)
    y = y[mask_valid].astype(int).reset_index(drop=True)
    ret = ret[mask_valid].reset_index(drop=True)

    n = len(feat_df)
    print(f"Aligned length n: {n}")
    print(f"Label long rate (mean(y)): {y.mean():.4f}")

    # 3) Modeli yükle (LSTM ağırlıklı: alpha = 1.0)
    logger = logging.getLogger("system")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    hm = HybridModel(model_dir="models", interval=INTERVAL, logger=logger)
    hm.alpha = 1.0  # LSTM-only hibrit (SGD katkısı yok)

    # 4) Skor üret
    p_arr, dbg = hm.predict_proba(feat_df)
    p_arr = np.asarray(p_arr, dtype=float)

    # Emin ol uzunluklar eşit
    min_len = min(len(p_arr), len(y), len(ret))
    p_arr = p_arr[-min_len:]
    y = y.iloc[-min_len:].to_numpy()
    ret = ret.iloc[-min_len:].to_numpy()

    # 5) AUC & genel skor istatistikleri
    try:
        auc = roc_auc_score(y, p_arr)
    except Exception:
        auc = float("nan")

    print("\n--- Genel Skor İstatistikleri ---")
    print(f"AUC (p vs label)      : {auc:.4f}")
    print(f"p_hybrid mean         : {p_arr.mean():.4f}")
    print(f"p_hybrid std          : {p_arr.std():.4f}")
    print(f"Hybrid debug mode     : {dbg.get('mode')}")
    print(f"best_auc/meta         : {dbg.get('best_auc')} | best_side={dbg.get('best_side')}")
    print(f"use_lstm_hybrid       : {dbg.get('use_lstm_hybrid')}")

    # 6) Threshold grid search
    print("\n--- Threshold Grid Search (5m) ---")
    best_result = None

    for long_thr in LONG_THR_GRID:
        for short_thr in SHORT_THR_GRID:
            if short_thr >= long_thr:
                continue

            res = run_pnl_backtest(p_arr, ret, long_thr, short_thr)
            if res["n_trades"] == 0:
                continue

            key = (long_thr, short_thr)
            if best_result is None or res["total_ret"] > best_result["total_ret"]:
                best_result = {
                    "long_thr": long_thr,
                    "short_thr": short_thr,
                    **res,
                }

    if best_result is None:
        print("Hiç trade çıkmadı, threshold grid çok agresif olabilir.")
        return

    print("\nEn iyi bulunan eşik kombinasyonu:")
    print(f"  long_thr  : {best_result['long_thr']:.3f}")
    print(f"  short_thr : {best_result['short_thr']:.3f}")
    print(f"  total_ret : {best_result['total_ret']:.4f}")
    print(f"  n_trades  : {best_result['n_trades']}")
    print(f"  win_rate  : {best_result['win_rate']:.4f}")
    print(f"  max_dd    : {best_result['max_dd']:.4f}")


if __name__ == "__main__":
    main()

