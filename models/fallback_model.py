# models/fallback_model.py
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FallbackModel:
    """
    ML modelleri (ensemble, online learner vs.) hata verirse
    veya henüz eğitilmemişse devreye giren basit yedek model.

    Amaç:
      - trade karar pipeline'ını bozmadan "makul" bir default üretmek
      - her yerde aynı interface'i sağlamak: predict_proba / predict
    """

    def __init__(self, default_proba: float = 0.5):
        """
        :param default_proba: Alım yönü için varsayılan olasılık (0 ile 1 arası).
                              0.5 = tarafsız (ne al, ne sat).
        """
        self.default_proba = float(default_proba)

    def predict_proba(self, X: Optional[np.ndarray]) -> np.ndarray:
        """
        X boyutuna göre sabit bir olasılık matrisi döner.
        İki sınıflı problem için [p0, p1] şeklinde.
        """
        try:
            if X is None:
                n = 1
            else:
                # X bir array ise satır sayısını al
                n = X.shape[0] if hasattr(X, "shape") else len(X)

            p_buy = np.clip(self.default_proba, 0.0, 1.0)
            p_sell = 1.0 - p_buy

            probs = np.tile(np.array([[p_sell, p_buy]]), (n, 1))
            return probs
        except Exception as e:
            logger.exception(f"[FallbackModel] predict_proba hatası: {e}")
            # Tamamen çakılırsa bile en azından 0.5/0.5 dön
            return np.tile(np.array([[0.5, 0.5]]), (1, 1))

    def predict(self, X: Optional[np.ndarray]) -> np.ndarray:
        """
        Sınıf tahmini döner (0 veya 1).
        Basit kural:
          - p_buy > 0.5 ise 1 (AL)
          - p_buy <= 0.5 ise 0 (SAT/ALMA)
        """
        probs = self.predict_proba(X)
        # class 1 (buy) için olasılık
        p_buy = probs[:, 1]
        return (p_buy > 0.5).astype(int)

