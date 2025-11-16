import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Batch olarak eğitilmiş bir modelin üzerine,
    yeni gelen verilerle (stream / online) kademeli güncelleme yapan katman.

    main.py'de kullanım:
        batch_learner = BatchLearner(clean_features)
        batch_model = batch_learner.train()
        online_learner = OnlineLearner(batch_model)
    """

    def __init__(self, base_model: Optional[object] = None, classes=(0, 1)):
        """
        :param base_model: BatchLearner tarafından dönen model (isteğe bağlı)
        :param classes: Sınıf etiketleri (binary için varsayılan (0,1))
        """
        self.base_model = base_model
        self.classes = np.array(classes)

        # Online learning için ayrı bir SGDClassifier kullanıyoruz
        self.online_model = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            max_iter=5,
            tol=1e-3,
            random_state=42,
        )
        self._is_fitted = False

    def initialize_with_batch(self, X: np.ndarray, y: np.ndarray):
        """
        İlk kez partial_fit yapmak için batch veriyi kullan.
        """
        try:
            if X is None or len(X) == 0:
                logger.warning("[OnlineLearner] initialize_with_batch: boş X, atlanıyor.")
                return

            self.online_model.partial_fit(X, y, classes=self.classes)
            self._is_fitted = True
            logger.info("[OnlineLearner] İlk online model fit işlemi tamamlandı.")
        except Exception as e:
            logger.exception(f"[OnlineLearner] initialize_with_batch hatası: {e}")
            raise DataProcessingException(f"OnlineLearner initialize_with_batch failed: {e}") from e

    def partial_update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        Yeni gelen küçük bir batch ile modeli günceller.
        """
        if X_new is None or len(X_new) == 0:
            logger.warning("[OnlineLearner] partial_update: boş X_new, atlanıyor.")
            return

        try:
            if not self._is_fitted:
                # İlk kez eğitilecekse sınıfları vererek başlat
                self.online_model.partial_fit(X_new, y_new, classes=self.classes)
                self._is_fitted = True
                logger.info("[OnlineLearner] partial_update ile ilk fit tamamlandı.")
            else:
                self.online_model.partial_fit(X_new, y_new)
                logger.info("[OnlineLearner] partial_update ile model güncellendi.")
        except Exception as e:
            logger.exception(f"[OnlineLearner] partial_update hatası: {e}")
            raise DataProcessingException(f"OnlineLearner partial_update failed: {e}") from e

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Olasılık tahmini döner. Eğer online_model henüz eğitilmediyse,
        base_model'den tahmin almayı dener, o da yoksa 0.5 döner.
        """
        try:
            if self._is_fitted:
                return self.online_model.predict_proba(X)
            elif self.base_model is not None and hasattr(self.base_model, "predict_proba"):
                logger.info("[OnlineLearner] Online model eğitilmemiş, base_model kullanılıyor.")
                return self.base_model.predict_proba(X)
            else:
                logger.warning(
                    "[OnlineLearner] Ne online model ne de base_model hazır, 0.5 olasılık dönüyor."
                )
                # shape: (n_samples, 2) -> [p0, p1]
                n = X.shape[0]
                return np.tile(np.array([[0.5, 0.5]]), (n, 1))
        except Exception as e:
            logger.exception(f"[OnlineLearner] predict_proba hatası: {e}")
            raise DataProcessingException(f"OnlineLearner predict_proba failed: {e}") from e

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Sınıf tahmini döner (0/1). Eğer model hazır değilse,
        predict_proba üzerinden 0.5 eşik ile karar verir.
        """
        try:
            if self._is_fitted:
                return self.online_model.predict(X)
            else:
                proba = self.predict_proba(X)
                # p(class=1) > 0.5 ise 1, değilse 0
                return (proba[:, 1] > 0.5).astype(int)
        except Exception as e:
            logger.exception(f"[OnlineLearner] predict hatası: {e}")
            raise DataProcessingException(f"OnlineLearner predict failed: {e}") from e

