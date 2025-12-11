import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("system")


class MultiTimeframeHybridEnsemble:
    """
    Çoklu timeframe (1m, 5m, 15m, 1h) HybridModel ensemble sınıfı.

    - Her interval için ayrı bir HybridModel bekler.
    - predict_mtf:
        * Her intervalde model.predict_proba(X_itv) çağırır
        * Son barın ([-1]) olasılığını alır
        * Model meta içindeki AUC'a göre weight hesaplar
        * Ağırlıklı ortalama ile ensemble p üretir
        * Detayları mtf_debug içinde döner
    """

    def __init__(
        self,
        models_by_interval: Dict[str, Any],
        low_auc_threshold: float = 0.52,
        low_auc_weight: float = 0.30,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.models_by_interval = models_by_interval or {}
        self.low_auc_threshold = float(low_auc_threshold)
        self.low_auc_weight = float(low_auc_weight)
        self.logger = logger_ or logger

    # --------------------------------------------------------------
    # Yardımcı: son bar olasılığını düzgün scalar'a çevir
    # --------------------------------------------------------------
    @staticmethod
    def _extract_last_prob(proba_arr: Any) -> float:
        """
        HybridModel.predict_proba'nın döndürdüğü proba array'inden
        son barın olasılığını (float) çıkarır.

        - Giriş: np.ndarray, list vb. olabilir
        - Çıkış: scalar float
        """
        arr = np.asarray(proba_arr)

        # tek boyuta indir (n_samples, ) hale getir
        if arr.ndim > 1:
            # Örn: (n_samples, 1) ise
            arr = arr.reshape(arr.shape[0], -1)
            # Son kolon (genelde tek kolon) üzerinden ilerleyelim
            arr = arr[:, -1]

        if arr.size == 0:
            return 0.5  # tamamen fallback

        return float(arr[-1])

    # --------------------------------------------------------------
    # Ana fonksiyon: çoklu TF ensemble
    # --------------------------------------------------------------
    def predict_mtf(
        self,
        X_by_interval: Dict[str, pd.DataFrame],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        X_by_interval:
            {
              "1m": df_1m,
              "5m": df_5m,
              "15m": df_15m,
              "1h": df_1h,
            }

        Dönüş:
            (ensemble_p_used, mtf_debug)
        """

        per_interval: Dict[str, Any] = {}
        probs: list[float] = []
        weights: list[float] = []
        intervals_used: list[str] = []

        for itv, model in self.models_by_interval.items():
            X_itv = X_by_interval.get(itv)
            if X_itv is None or len(X_itv) == 0:
                continue

            try:
                proba_arr, dbg = model.predict_proba(X_itv)

                # Son barın olasılığını scalar'a çek
                p_last = self._extract_last_prob(proba_arr)

                # Meta'dan AUC / best_side çek
                meta = getattr(model, "meta", {}) or {}
                best_auc = float(meta.get("best_auc", 0.5) or 0.5)
                best_side = meta.get("best_side", "long")

                # Weight hesaplama
                # - AUC çok düşükse (ör: 0.52 altı) low weight ver
                if best_auc < self.low_auc_threshold:
                    weight = self.low_auc_weight
                    self.logger.info(
                        "[HYBRID-MTF] Interval=%s düşük AUC ile düşük weight=%.2f kullanılıyor (skip edilmedi).",
                        itv,
                        weight,
                    )
                else:
                    weight = best_auc

                per_interval[itv] = {
                    "p_last": float(p_last),
                    "weight": float(weight),
                    "auc": float(best_auc),
                    "best_side": best_side,
                    "debug": dbg,
                }

                probs.append(float(p_last))
                weights.append(float(weight))
                intervals_used.append(itv)

            except Exception as e:
                self.logger.warning(
                    "[HYBRID-MTF] Interval=%s için predict_proba hata: %s",
                    itv,
                    e,
                )
                continue

        # Hiç interval kullanılamadıysa fallback
        if not probs:
            self.logger.warning(
                "[HYBRID-MTF] Hiç interval için geçerli prob üretilemedi, ensemble p_single'a fallback yapılmalı."
            )
            return 0.5, {
                "per_interval": per_interval,
                "intervals_used": [],
                "weights_norm": [],
                "n_used": 0,
            }

        probs_arr = np.asarray(probs, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)

        # Ağırlıkları normalize et
        if weights_arr.sum() <= 0:
            weights_norm = np.ones_like(weights_arr) / len(weights_arr)
        else:
            weights_norm = weights_arr / weights_arr.sum()

        # Ağırlıklı ortalama ensemble probability
        ensemble_p = float((probs_arr * weights_norm).sum())

        mtf_debug = {
            "per_interval": per_interval,
            "intervals_used": intervals_used,
            "weights_raw": weights,
            "weights_norm": weights_norm.tolist(),
            "n_used": len(intervals_used),
            "ensemble_p": ensemble_p,
        }

        return ensemble_p, mtf_debug
