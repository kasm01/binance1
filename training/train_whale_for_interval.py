#!/usr/bin/env python
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# main.py içindeki config ile uyumlu olsun
from config import config

logger = logging.getLogger("whale_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def load_offline_klines(symbol: str, interval: str, limit: int = 20000) -> pd.DataFrame:
    """
    Offline cache'ten klines yükler:
      data/offline_cache/{symbol}_{interval}_6m.csv
    """
    path = Path("data/offline_cache") / f"{symbol}_{interval}_6m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Offline cache yok: {path}")

    df = pd.read_csv(path)
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info(
        "[DATA] Loaded offline klines: symbol=%s interval=%s shape=%s path=%s",
        symbol,
        interval,
        df.shape,
        str(path),
    )
    return df


def compute_future_returns(df: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """
    horizon bar sonrası getiriyi hesaplar:
      r_t = (close_{t+h} / close_t) - 1
    """
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    ret = (future / close) - 1.0
    return ret


def generate_whale_signals(
    df: pd.DataFrame,
    window: int,
    volume_zscore_thr: float,
) -> pd.Series:
    """
    Basit whale sinyali:
      - hacim (volume) için rolling mean/std
      - zscore = (volume - mean) / std
      - zscore >= volume_zscore_thr ise 1, değilse 0
    """
    vol = df["volume"].astype(float)
    ma = vol.rolling(window).mean()
    std = vol.rolling(window).std()

    zscore = (vol - ma) / (std.replace(0.0, np.nan))
    zscore = zscore.fillna(0.0)

    sig = (zscore >= volume_zscore_thr).astype(int)
    return sig


def backtest_whale_params(
    df: pd.DataFrame,
    horizon: int,
    window: int,
    volume_zscore_thr: float,
) -> Dict[str, Any]:
    """
    Verilen param seti için:
      - whale sinyalleri (0/1)
      - sinyal anındaki horizon getirileri
      - n_signals, win_rate, avg_return hesaplar
    """
    signals = generate_whale_signals(df, window=window, volume_zscore_thr=volume_zscore_thr)
    fut_ret = compute_future_returns(df, horizon=horizon)

    # Sinyal olan barları seç
    mask = signals == 1
    sig_rets = fut_ret[mask].dropna()

    n_signals = int(mask.sum())
    if n_signals == 0 or len(sig_rets) == 0:
        return {
            "window": window,
            "volume_zscore_thr": volume_zscore_thr,
            "n_signals": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
        }

    wins = (sig_rets > 0).sum()
    win_rate = float(wins) / float(len(sig_rets))
    avg_return = float(sig_rets.mean())

    return {
        "window": window,
        "volume_zscore_thr": volume_zscore_thr,
        "n_signals": int(len(sig_rets)),
        "win_rate": float(win_rate),
        "avg_return": float(avg_return),
    }


def grid_search_whale(
    df: pd.DataFrame,
    horizon: int = 3,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Basit grid search:
      window ∈ {20, 50, 100}
      volume_zscore_thr ∈ {1.5, 2.0, 2.5, 3.0}

    Skor fonksiyonu:
      - min_sinyal sayısı altındaki konfigler elenir
      - kalanlar win_rate'e göre sıralanır
      - eşitlikte avg_return yüksek olan kazanır
    """
    windows = [20, 50, 100]
    thrs = [1.5, 2.0, 2.5, 3.0]
    min_signals = 30

    results: List[Dict[str, Any]] = []

    for w in windows:
        for thr in thrs:
            stats = backtest_whale_params(df, horizon=horizon, window=w, volume_zscore_thr=thr)
            logger.info(
                "[GRID] window=%d thr=%.2f -> n_signals=%d win_rate=%.3f avg_ret=%.5f",
                w,
                thr,
                stats["n_signals"],
                stats["win_rate"],
                stats["avg_return"],
            )
            results.append(stats)

    # Filtre: min sinyal sayısı
    candidates = [r for r in results if r["n_signals"] >= min_signals]

    if not candidates:
        logger.warning(
            "[GRID] min_signals=%d şartını sağlayan konfig yok. En çok sinyal üreten konfig seçilecek.",
            min_signals,
        )
        # En çok sinyal üreteni seç
        best = max(results, key=lambda r: r["n_signals"])
    else:
        # Önce win_rate, sonra avg_return
        best = max(
            candidates,
            key=lambda r: (r["win_rate"], r["avg_return"]),
        )

    logger.info(
        "[BEST] window=%d thr=%.2f | n_signals=%d win_rate=%.3f avg_ret=%.5f",
        best["window"],
        best["volume_zscore_thr"],
        best["n_signals"],
        best["win_rate"],
        best["avg_return"],
    )

    return best, results


def save_meta_and_backtest(
    symbol: str,
    interval: str,
    horizon: int,
    best: Dict[str, Any],
    all_results: List[Dict[str, Any]],
) -> None:
    """
    Sonuçları:
      - models/whale_detector_meta_{symbol}_{interval}.json
      - models/whale_detector_backtest_{symbol}_{interval}.json
    olarak kaydeder.
    """
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()

    meta = {
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "best_params": {
            "window": best["window"],
            "volume_zscore_thr": best["volume_zscore_thr"],
        },
        "n_signals": best["n_signals"],
        "win_rate": best["win_rate"],
        "avg_return": best["avg_return"],
        "created_at": timestamp,
    }

    backtest = {
        "symbol": symbol,
        "interval": interval,
        "horizon": horizon,
        "grid_results": all_results,
        "created_at": timestamp,
    }

    meta_path = models_dir / f"whale_detector_meta_{symbol}_{interval}.json"
    backtest_path = models_dir / f"whale_detector_backtest_{symbol}_{interval}.json"

    meta_path.write_text(json.dumps(meta, indent=2))
    backtest_path.write_text(json.dumps(backtest, indent=2))

    logger.info("[SAVE] Meta kaydedildi: %s", meta_path)
    logger.info("[SAVE] Backtest grid kaydedildi: %s", backtest_path)


def main() -> None:
    # Sembol ve interval
    symbol = os.getenv("SYMBOL", getattr(config, "SYMBOL", "BTCUSDT"))
    interval = os.getenv("INTERVAL", "5m")
    horizon = int(os.getenv("WHALE_HORIZON", "3"))

    logger.info(
        "[START] Whale training | symbol=%s interval=%s horizon=%d",
        symbol,
        interval,
        horizon,
    )

    df = load_offline_klines(symbol, interval, limit=20000)

    # Güvenlik için temel kolonlar kontrolü
    required_cols = ["close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Offline CSV'de eksik kolonlar var: {missing}")

    best, all_results = grid_search_whale(df, horizon=horizon)
    save_meta_and_backtest(symbol, interval, horizon, best, all_results)

    logger.info("[DONE] Whale training tamamlandı.")


if __name__ == "__main__":
    main()
