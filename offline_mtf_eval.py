import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Mevcut pipeline ile aynı feature/label mantığını kullanmak için main'den import
from main import (
    build_features,
    build_labels,
    load_offline_klines_from_cache,
    SYMBOL,
    MTF_INTERVALS,
)

from models.hybrid_inference import HybridModel, HybridMultiTFModel


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------
logger = logging.getLogger("mtf_eval")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ----------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------
def prepare_interval_data(
    symbol: str,
    interval: str,
    horizon: int = 1,
    limit: int = 20000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Tek bir timeframe için:
      - offline cache'den klines yükler
      - build_features ile feature DF üretir
      - build_labels ile horizon=1 label üretir
      - feature ve label uzunluklarını hizalar

    Dönüş:
      raw_df, feat_df, y
    """
    raw_df = load_offline_klines_from_cache(symbol, interval, limit=limit)
    logger.info(
        "[%s] raw_df shape: %s | columns=%s",
        interval,
        raw_df.shape,
        list(raw_df.columns),
    )

    feat_df = build_features(raw_df)
    logger.info(
        "[%s] feat_df shape: %s | columns=%s",
        interval,
        feat_df.shape,
        list(feat_df.columns),
    )

    # Label (eğitimle aynı mantık)
    y = build_labels(raw_df["close"], horizon=horizon)

    # Uzunluk hizalama:
    # - feat_df dropna nedeniyle kısalmış olabilir
    # - y son horizon bar için NaN / gelecek bilinemez
    n = min(len(feat_df), len(y) - horizon)
    if n <= 0:
        raise ValueError(f"[{interval}] Çok kısa seri, n={n}")

    feat_df = feat_df.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].astype(int).reset_index(drop=True)

    logger.info(
        "[%s] Aligned length n=%d | long_rate=%.4f",
        interval,
        n,
        float(y.mean()),
    )

    return raw_df, feat_df, y


def eval_single_interval(
    symbol: str,
    interval: str,
    horizon: int = 1,
    limit: int = 20000,
) -> Dict[str, float]:
    """
    Bir interval için HybridModel + offline AUC ölçümü.
    """
    logger.info("========== Single-TF Eval | interval=%s ==========", interval)
    raw_df, feat_df, y = prepare_interval_data(symbol, interval, horizon=horizon, limit=limit)

    model = HybridModel(
        model_dir="models",
        interval=interval,
        logger=logger,
    )

    p_arr, debug = model.predict_proba(feat_df)
    p_arr = np.asarray(p_arr, dtype=float)

    if len(p_arr) != len(y):
        n = min(len(p_arr), len(y))
        p_arr = p_arr[:n]
        y = y.iloc[:n].reset_index(drop=True)

    try:
        auc = roc_auc_score(y, p_arr)
    except Exception as e:
        logger.warning("[%s] AUC hesaplanırken hata: %s", interval, e)
        auc = float("nan")

    logger.info(
        "[%s] Offline AUC=%.4f | p_mean=%.4f | p_std=%.4f | mode=%s | meta_best_auc=%.4f",
        interval,
        float(auc),
        float(p_arr.mean()),
        float(p_arr.std()),
        debug.get("mode", "unknown"),
        float(model.meta.get("best_auc", 0.0)),
    )

    return {
        "interval": interval,
        "auc": float(auc),
        "p_mean": float(p_arr.mean()),
        "p_std": float(p_arr.std()),
        "meta_best_auc": float(model.meta.get("best_auc", 0.0)),
    }


def eval_mtf_ensemble_snapshot(
    symbol: str,
    intervals: List[str],
    horizon: int = 1,
    limit: int = 20000,
    tail_n: int = 500,
) -> None:
    """
    HybridMultiTFModel'i 1m/5m/15m/1h için offline snapshot olarak test eder.

    Adımlar:
      - Her interval için raw_df, feat_df, y hesapla
      - Her interval için son tail_n bar'ı al
      - X_dict = {interval: feat_df_tail} oluştur
      - HybridMultiTFModel.predict_proba_multi(X_dict) çağır
      - Per-interval ve ensemble debug logla
    """
    logger.info("========== Multi-Timeframe Ensemble Snapshot ==========")

    # Per-interval feature'ları hazırla
    X_dict: Dict[str, pd.DataFrame] = {}
    per_interval_info: Dict[str, Dict[str, float]] = {}

    for itv in intervals:
        try:
            _, feat_df, y = prepare_interval_data(
                symbol=symbol,
                interval=itv,
                horizon=horizon,
                limit=limit,
            )

            tail_df = feat_df.tail(tail_n).reset_index(drop=True)
            X_dict[itv] = tail_df

            logger.info(
                "[%s] tail_df shape: %s (tail_n=%d)",
                itv,
                tail_df.shape,
                tail_n,
            )

            per_interval_info[itv] = {
                "n_samples": float(len(tail_df)),
            }
        except Exception as e:
            logger.warning(
                "[%s] Interval hazırlanırken hata, ensemble'a dahil edilmeyecek: %s",
                itv,
                e,
            )

    if not X_dict:
        logger.error("Ensemble için kullanılabilir hiçbir interval yok.")
        return

    # HybridMultiTFModel'i kur (sadece hazır olan interval'lar)
    mtf_model = HybridMultiTFModel(
        model_dir="models",
        intervals=list(X_dict.keys()),
        logger=logger,
    )

    ensemble_p, debug_mtf = mtf_model.predict_proba_multi(
        X_dict=X_dict,
        standardize_auc_key="auc_used",
        standardize_overwrite=False,
    )

    logger.info("---------- Per-interval contribs ----------")
    per_int = debug_mtf.get("per_interval", {}) or {}
    for itv, info in per_int.items():
        logger.info(
            "[%s] p_last=%.4f | weight=%.4f | meta_best_auc=%.4f | best_side_meta=%s",
            itv,
            float(info.get("p_last", 0.5)),
            float(info.get("weight", 0.0)),
            float(info.get("best_auc_meta", 0.0)),
            info.get("best_side_meta", "best"),
        )

    logger.info("---------- Ensemble summary ----------")
    ens = debug_mtf.get("ensemble", {}) or {}
    logger.info(
        "ensemble_p=%.4f | n_used=%d",
        float(ens.get("p", ensemble_p)),
        int(ens.get("n_used", 0)),
    )

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    symbol = SYMBOL  # main.py içindeki default sembol (BTCUSDT)
    intervals = MTF_INTERVALS  # ["1m", "5m", "15m", "1h"]

    logger.info("Sembol: %s | Intervals: %s", symbol, intervals)

    # 1) Her interval için single-TF offline AUC (kontrol amaçlı)
    results = []
    for itv in intervals:
        try:
            res = eval_single_interval(symbol, itv, horizon=1, limit=20000)
            results.append(res)
        except Exception as e:
            logger.warning("[MAIN] %s eval_single_interval hata: %s", itv, e)

    logger.info("========== Single-TF Summary ==========")
    for r in results:
        logger.info(
            "[%s] AUC=%.4f | p_mean=%.4f | p_std=%.4f | meta_best_auc=%.4f",
            r["interval"],
            r["auc"],
            r["p_mean"],
            r["p_std"],
            r["meta_best_auc"],
        )

    # 2) Multi-timeframe ensemble snapshot (son 500 bar)
    eval_mtf_ensemble_snapshot(
        symbol=symbol,
        intervals=intervals,
        horizon=1,
        limit=20000,
        tail_n=500,
    )


if __name__ == "__main__":
    main()
