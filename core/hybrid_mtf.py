# core/hybrid_mtf.py (örnek)

from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from core.logger import system_logger


class MultiTimeframeHybridEnsemble:
    """
    Birden fazla interval (1m,5m,15m,1h) için HybridModel çalıştırıp
    AUC'ye göre ağırlıklandırılmış bir ensemble olasılığı üretir.

    Not:
      - Burada her interval için LSTM+SGD (mümkünse) kullanıyoruz.
      - best_auc düşükse interval'e düşük weight veriyoruz (ör: 0.30).
    """

    def __init__(
        self,
        models_by_interval: Dict[str, "HybridModel"],
    ) -> None:
        """
        models_by_interval:
            {
              "1m": hybrid_model_1m,
              "5m": hybrid_model_5m,
              "15m": hybrid_model_15m,
              "1h": hybrid_model_1h,
            }
        """
        self.models_by_interval = models_by_interval

    def predict_mtf(
        self,
        X_by_interval: Dict[str, pd.DataFrame],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Her interval için:
          - HybridModel.predict_proba(X, use_lstm=True) çağrılır.
          - best_auc'e göre ağırlık w hesaplanır.
          - ensemble_p = (Σ w_i * p_i) / (Σ w_i)

        Dönen:
          ensemble_p: float
          mtf_debug: {
             "per_interval": {
                 "1m": {
                     "p_last": ...,
                     "weight": ...,
                     "best_auc": ...,
                     "best_side": ...,
                     "mode": "lstm+sgd" veya "sgd_only"
                 },
                 ...
             },
             "ensemble_p": ...,
             "n_used": ...
          }
        """
        per_interval: Dict[str, Dict[str, Any]] = {}
        sum_w = 0.0
        sum_score = 0.0
        n_used = 0

        for itv, model in self.models_by_interval.items():
            X = X_by_interval.get(itv)
            if X is None or len(X) == 0:
                continue

            try:
                # 🔥 CRITICAL: LSTM'i de devreye sokuyoruz
                p_used, meta = model.predict_proba(X, use_lstm=True)
            except TypeError:
                # Eski imzada use_lstm yoksa fallback
                p_used, meta = model.predict_proba(X)

            # meta içinden best_auc / best_side / mode çek
            best_auc = float(meta.get("best_auc", 0.5))
            best_side = meta.get("best_side", "long")
            mode = meta.get("mode", "unknown")  # HybridModel içinde set ediyorsan

            # AUC'ye göre weight hesapla
            if best_auc <= 0.5:
                w = 0.30
                if system_logger:
                    system_logger.info(
                        "[HYBRID-MTF] Interval=%s düşük AUC ile düşük weight=%.2f "
                        "kullanılıyor (skip edilmedi).",
                        itv,
                        w,
                    )
            else:
                # basit örnek: AUC arttıkça 1.0–1.5 arası
                w = 1.0 + (best_auc - 0.5) * 1.0
                if system_logger:
                    system_logger.info(
                        "[HYBRID-MTF] Interval=%s AUC=%.4f -> weight=%.2f",
                        itv,
                        best_auc,
                        w,
                    )

            # Negatif veya sıfırsa dahil etme
            if w <= 0.0:
                if system_logger:
                    system_logger.info(
                        "[HYBRID-MTF] Interval=%s weight<=0, ensemble'dan çıkarıldı.",
                        itv,
                    )
                continue

            sum_w += w
            sum_score += w * float(p_used)
            n_used += 1

            per_interval[itv] = {
                "p_last": float(p_used),
                "weight": float(w),
                "best_auc": best_auc,
                "best_side": best_side,
                "mode": mode,
            }

        if sum_w > 0.0:
            ensemble_p = float(sum_score / sum_w)
        else:
            ensemble_p = 0.5  # nötr

        if system_logger:
            system_logger.info(
                "[HYBRID-MTF] ensemble_p=%.4f, n_used=%d",
                ensemble_p,
                n_used,
            )

        mtf_debug = {
            "per_interval": per_interval,
            "ensemble_p": ensemble_p,
            "n_used": n_used,
        }
        return ensemble_p, mtf_debug
