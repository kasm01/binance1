#!/usr/bin/env python
# offline_eval_hybrid.py
#
# Farklı zaman dilimleri (1m, 5m, 15m, 1h) için
# HybridModel (SGD + LSTM) offline performans değerlendirmesi.

import os
import sys
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Proje içi importlar
from models.hybrid_inference import HybridModel
from utils.labels import build_labels


# ----------------------------------------------------------------------
# Basit feature builder (main.py ile uyumlu)
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim ve inference pipeline'ında kullandığımız minimal feature set:

    Kolonlar:
    ----------
    ['open_time', 'open', 'high', 'low', 'close', 'volume',
     'close_time', 'quote_asset_volume', 'number_of_trades',
     'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore',
     'hl_range', 'oc_change', 'return_1', 'return_3', 'return_5',
     'ma_5', 'ma_10', 'ma_20', 'vol_10', 'dummy_extra']
    """
    df = raw_df.copy()

    # Price / range feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    # Basit getiriler
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # Hareketli ortalamalar
    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    # Volatilite benzeri feature
    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    # Eğitime uyum için ekstra dummy kolon
    df["dummy_extra"] = 0.0

    # pct_change kaynaklı NaN'leri temizle
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Tek interval için değerlendirme
# ----------------------------------------------------------------------
def evaluate_interval(
    interval: str,
    csv_path: str,
    model_dir: str = "models",
    horizon: int = 1,
    n_last_bars: int = 2000,
) -> Dict[str, Any]:
    """
    Tek bir zaman dilimi (ör. 5m) için:
      - CSV'den veri alır
      - Feature üretir
      - Label üretir
      - HybridModel ile skor hesaplayıp AUC vb istatistikleri döndürür.
    """

    print("=" * 80)
    print(f"Interval: {interval} | CSV: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV bulunamadı: {csv_path}")
        return {"interval": interval, "error": "csv_not_found"}

    # -----------------------------
    # 1) Veriyi oku ve son N barı al
    # -----------------------------
    raw_df = pd.read_csv(csv_path)
    print(f"raw_df shape: {raw_df.shape}")
    print(f"raw_df columns: {raw_df.columns.tolist()}")

    if len(raw_df) > n_last_bars:
        raw_df = raw_df.tail(n_last_bars).reset_index(drop=True)

    # -----------------------------
    # 2) Feature'ları üret
    # -----------------------------
    feat_df = build_features(raw_df)
    print(f"feat_df shape: {feat_df.shape}")
    print(f"feat_df columns: {feat_df.columns.tolist()}")

    # -----------------------------
    # 3) Label'ları üret (future close > 0 mu?)
    # -----------------------------
    labels = build_labels(raw_df, horizon=horizon)

    # Feature'lar dropna ile kısaldığı için, label'ları da en sondan
    # feat_df uzunluğu kadar alıp index'i resetleyelim.
    if len(labels) < len(feat_df):
        # Teorik olarak olmaması gerekir ama güvenlik için
        min_len = len(labels)
        feat_df = feat_df.tail(min_len).reset_index(drop=True)
    else:
        labels = labels.iloc[-len(feat_df):].reset_index(drop=True)

    y = labels.astype(int).to_numpy()
    n_samples = len(y)

    # -----------------------------
    # 4) Modeli yükle ve tahmin al
    # -----------------------------
    hm = HybridModel(model_dir=model_dir, interval=interval)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_hybrid, dbg = hm.predict_proba(feat_df)

    # Boyut eşitleme (güvenlik)
    if len(p_hybrid) > n_samples:
        p_hybrid = p_hybrid[-n_samples:]
    elif len(p_hybrid) < n_samples:
        y = y[-len(p_hybrid):]
        n_samples = len(y)

    # -----------------------------
    # 5) AUC ve basic stats
    # -----------------------------
    try:
        auc = roc_auc_score(y, p_hybrid)
    except ValueError:
        auc = float("nan")

    long_rate = float(y.mean())  # 1'lerin oranı
    p_mean = float(p_hybrid.mean()) if n_samples > 0 else float("nan")
    p_std = float(p_hybrid.std()) if n_samples > 0 else float("nan")

    print(f"[{interval}] n_samples         : {n_samples}")
    print(f"[{interval}] AUC               : {auc:.4f}" if not np.isnan(auc) else f"[{interval}] AUC               : nan")
    print(f"[{interval}] Label long rate   : {long_rate:.4f}")
    print(f"[{interval}] p_hybrid mean     : {p_mean:.4f}")
    print(f"[{interval}] p_hybrid std      : {p_std:.4f}")
    print(f"[{interval}] Hybrid debug mode : {dbg.get('mode')}")
    print(
        f"[{interval}] best_auc/meta     : {dbg.get('best_auc')} | best_side={dbg.get('best_side')}"
    )
    print(f"[{interval}] use_lstm_hybrid   : {dbg.get('use_lstm_hybrid')}")
    print("=" * 80)
    print()

    return {
        "interval": interval,
        "n_samples": n_samples,
        "auc": auc,
        "long_rate": long_rate,
        "p_mean": p_mean,
        "p_std": p_std,
        "debug": dbg,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    # Değerlendirilecek interval ve CSV path’leri
    intervals = {
        "1m": "data/offline_cache/BTCUSDT_1m_6m.csv",
        "5m": "data/offline_cache/BTCUSDT_5m_6m.csv",
        "15m": "data/offline_cache/BTCUSDT_15m_6m.csv",
        "1h": "data/offline_cache/BTCUSDT_1h_6m.csv",
    }

    results: Dict[str, Any] = {}

    for interval, csv_path in intervals.items():
        res = evaluate_interval(
            interval=interval,
            csv_path=csv_path,
            model_dir="models",
            horizon=1,
            n_last_bars=2000,
        )
        results[interval] = res

    # İstersen burada results sözlüğünü JSON olarak kaydedebilirsin:
    # import json
    # with open("offline_hybrid_eval_results.json", "w") as f:
    #     json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    # Projeyi kökten çalıştırmadıysan path sorunlarını çözmek için:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()

