import os
import argparse
import json
import logging

import pandas as pd

from core.whale_detector import (
    WhaleDetector,
    WhaleDetectorBacktester,
    OptimizedWhaleDetector,
)


def load_raw_df(symbol: str, interval: str) -> pd.DataFrame:
    """
    Whale detector için ham kline verisini yükler.
    Şimdilik data/offline_cache içinden okuyoruz.
    """
    csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Whale eğitimi için CSV bulunamadı: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Temel kolonları kontrol et
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Whale eğitimi için eksik kolonlar: {missing}")

    # Tip düzeltmeleri
    for c in required:
        df[c] = df[c].astype(float)

    # Eğer timestamp varsa log için güzel olur
    if "open_time" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        except Exception:
            pass

    return df


def train_whale_for_interval(symbol: str, interval: str) -> None:
    logger = logging.getLogger("whale_train")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("[WHALE-TRAIN] Başlıyor -> %s %s", symbol, interval)

    df = load_raw_df(symbol, interval)
    logger.info("[WHALE-TRAIN] Ham veri shape=%s", df.shape)

    # ---------------------------------------------------------------------
    # 1) Optuna ile parametre optimizasyonu (OptimizedWhaleDetector)
    #    NOT: Bu kısım biraz zaman alabilir, ileride n_trials ayarlanabilir.
    # ---------------------------------------------------------------------
    try:
        logger.info("[WHALE-TRAIN] Optuna ile parametre optimizasyonu başlıyor...")
        opt = OptimizedWhaleDetector()
        best_params = opt.optimize_parameters(df)
        logger.info("[WHALE-TRAIN] En iyi parametreler: %s", best_params)

        # Meta'yı kaydet
        out_meta = {
            "symbol": symbol,
            "interval": interval,
            "best_params": best_params,
        }
        os.makedirs("models", exist_ok=True)
        meta_path = os.path.join(
            "models", f"whale_detector_meta_{symbol}_{interval}.json"
        )
        with open(meta_path, "w") as f:
            json.dump(out_meta, f, indent=2)
        logger.info("[WHALE-TRAIN] Meta kaydedildi -> %s", meta_path)

    except Exception as e:
        logger.warning(
            "[WHALE-TRAIN] Optuna optimizasyonu yapılamadı: %s", e
        )
        best_params = None

    # ---------------------------------------------------------------------
    # 2) Backtest (WhaleDetectorBacktester) – default veya best_params ile
    # ---------------------------------------------------------------------
    try:
        if best_params is not None:
            detector = WhaleDetector(**best_params)
            logger.info(
                "[WHALE-TRAIN] Backtest WhaleDetector(best_params) ile yapılıyor..."
            )
        else:
            detector = WhaleDetector()
            logger.info(
                "[WHALE-TRAIN] Backtest WhaleDetector(default) ile yapılıyor..."
            )

        backtester = WhaleDetectorBacktester(detector)
        results = backtester.backtest(df, lookahead_bars=10)

        logger.info("[WHALE-TRAIN] Backtest sonuçları: %s", results)

        # İsteğe bağlı: sonuçları da JSON olarak kaydedebiliriz
        out_res = {
            "symbol": symbol,
            "interval": interval,
            "results": results,
        }
        res_path = os.path.join(
            "models", f"whale_detector_backtest_{symbol}_{interval}.json"
        )
        with open(res_path, "w") as f:
            json.dump(out_res, f, indent=2)
        logger.info("[WHALE-TRAIN] Backtest sonuçları kaydedildi -> %s", res_path)

    except Exception as e:
        logger.warning("[WHALE-TRAIN] Backtest sırasında hata: %s", e)

    logger.info("[WHALE-TRAIN] Tamamlandı -> %s %s", symbol, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval",
        type=str,
        required=True,
        help="Zaman aralığı (ör. 1m, 5m, 15m, 1h)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Sembol (varsayılan=BTCUSDT)",
    )

    args = parser.parse_args()
    train_whale_for_interval(symbol=args.symbol, interval=args.interval)
