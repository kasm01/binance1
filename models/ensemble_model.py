# models/ensemble_model.py

import logging
from typing import Optional

import numpy as np
from joblib import load

# FallbackModel dosyanı bazen Fallback_model.py olarak tuttuğun için
# her iki ismi de deniyoruz.
try:
    from models.fallback_model import FallbackModel
except ImportError:  # dosya adı büyük harfle ise
    from models.Fallback_model import FallbackModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble / online model + fallback sarmalayıcı.

    Şu anda basit davranıyoruz:
      - Varsa joblib ile kaydedilmiş bir model (online_model/joblib),
        onun predict_proba'sını kullan.
      - Model yoksa veya hata verirse FallbackModel'e düş.
    """

    def __init__(
        self,
        model_path: str = "models/online_model.joblib",
        fallback: Optional[FallbackModel] = None,
    ) -> None:
        self.model_path = model_path
        self.fallback = fallback or FallbackModel()
        self.model = None

        try:
            self.model = load(self.model_path)
            logger.info(
                "[EnsembleModel] Loaded base model from %s",
                self.model_path,
            )
        except Exception as e:
            logger.warning(
                "[EnsembleModel] Could not load model from %s, using fallback only: %s",
                self.model_path,
                e,
            )
            self.model = None

    # -----------------------------------------------------
    # PREDICT_PROBA
    # -----------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """
        İki sınıflı çıktı: [p_class0, p_class1] (class1 = BUY gibi düşünülüyor)
        """
        # Önce gerçek modeli dene
        try:
            if self.model is not None:
                proba = self.model.predict_proba(X)

                # Eğer zaten (n_samples, 2) ise direkt dön
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba

                # Eğer tek kolonlu probability (p_buy) geldiyse, [p_sell, p_buy] yap
                if proba.ndim == 1 or proba.shape[1] == 1:
                    p_buy = proba.ravel()
                    p_sell = 1.0 - p_buy
                    return np.vstack([p_sell, p_buy]).T

                # Beklenmedik bir shape ise logla ve fallback'e düş
                logger.warning(
                    "[EnsembleModel] Unexpected proba shape from base model: %s",
                    proba.shape,
                )
        except Exception as e:
            logger.error(
                "[EnsembleModel] predict_proba failed, falling back: %s",
                e,
                exc_info=True,
            )

        # Buraya geldiysek fallback devrede
        return self.fallback.predict_proba(X)

    # -----------------------------------------------------
    # PREDICT (sınıf)
    # -----------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """
        0 / 1 sınıf tahmini döner.
        Basit kural:
          - p_buy > 0.5 ise 1
          - p_buy <= 0.5 ise 0
        """
        proba = self.predict_proba(X)
        p_buy = proba[:, 1]
        return (p_buy > 0.5).astype(int)

