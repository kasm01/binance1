import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.hybrid_inference import HybridModel

# build_features'i main.py'den alıyoruz ki şema %100 uyumlu olsun
try:
    from main import build_features, SYMBOL
except ImportError as e:
    print(f"[ERROR] main.py içinden build_features import edilemedi: {e}")
    raise


def load_offline_klines(symbol: str, interval: str, limit: int = 2000) -> pd.DataFrame:
    path = Path("data/offline_cache") / f"{symbol}_{interval}_6m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Offline CSV yok: {path}")
    df = pd.read_csv(path)
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    print(f"[DATA] Loaded {path} shape={df.shape}")
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("hybrid_test")

    symbol = SYMBOL  # main.py içindeki sembol (muhtemelen BTCUSDT)
    interval = "5m"  # test için 5m

    print(f"=== HybridModel LSTM test | symbol={symbol} | interval={interval} ===")

    # 1) Veri yükle
    raw_df = load_offline_klines(symbol, interval, limit=2000)

    # 2) Feature'ları üret
    feat_df = build_features(raw_df)
    print(f"[FE] feat_df shape={feat_df.shape}")

    # Biraz tail alalım (LSTM seq_len için yeterli olsun)
    tail_n = 500
    if len(feat_df) > tail_n:
        feat_df = feat_df.tail(tail_n).reset_index(drop=True)
        print(f"[FE] tail feat_df shape={feat_df.shape}")

    # 3) HybridModel yükle
    hm = HybridModel(model_dir="models", interval=interval, logger=logger)

    print(f"[META] use_lstm_hybrid={hm.use_lstm_hybrid} | meta.best_auc={hm.meta.get('best_auc')}")

    # 4) Tahmin al
    p_arr, dbg = hm.predict_proba(feat_df)

    print("\n--- Hybrid Debug ---")
    print(f"mode               : {dbg.get('mode')}")
    print(f"use_lstm_hybrid    : {dbg.get('use_lstm_hybrid')}")
    print(f"lstm_used flag     : {dbg.get('lstm_used')}")
    print(f"p_sgd_mean         : {dbg.get('p_sgd_mean'):.4f}")
    print(f"p_lstm_mean        : {dbg.get('p_lstm_mean'):.4f}")
    print(f"p_hybrid_mean      : {dbg.get('p_hybrid_mean'):.4f}")
    print(f"best_auc/meta      : {dbg.get('best_auc')} | best_side={dbg.get('best_side')}")
    print(f"n_samples          : {len(p_arr)}")

    # Son bar için skor
    if len(p_arr) > 0:
        print(f"last p_hybrid      : {float(p_arr[-1]):.4f}")


if __name__ == "__main__":
    main()
