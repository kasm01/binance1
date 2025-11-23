# models/ensemble_model.py
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator

from .fallback_model import FallbackModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Birden fazla modeli soft voting ile birleştirir.
    Estimator verilmezse FallbackModel devreye girer.
    """

    def __init__(self, estimators: Optional[List[Tuple[str, BaseEstimator]]] = None):
        """
        :param estimators: [('lgbm', lgbm_model), ('cat', cat_model), ...]
        """
        self.estimators = estimators or []
        self.voting_clf: Optional[VotingClassifier] = None
        self.fallback = FallbackModel(default_proba=0.5)

        if self.estimators:
            self.voting_clf = VotingClassifier(
                estimators=self.estimators,
                voting="soft"
            )

    def fit(self, X, y):
        """
        Gerçek estimator varsa VotingClassifier'ı eğitir.
        Yoksa sadece fallback kullanılır (fit gerekmez).
        """
        if self.voting_clf is None:
            logger.warning(
                "[EnsembleModel] No estimators provided, using FallbackModel only."
            )
            return self

        self.voting_clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        2 sınıflı proba matrisi döner: [p_sell, p_buy]
        """
        if self.voting_clf is None:
            # Sadece fallback
            return self.fallback.predict_proba(X)

        try:
            return self.voting_clf.predict_proba(X)
        except Exception as e:
            logger.error(
                f"[EnsembleModel] Error in voting_clf.predict_proba: {e}, "
                "falling back to FallbackModel.",
                exc_info=True,
            )
            return self.fallback.predict_proba(X)

    def predict(self, X) -> str:
        """
        Trade sinyali döndürür: 'BUY' veya 'SELL'
        (İleride 'HOLD' eklemek istersen probaya göre ek kural yazabiliriz.)
        """
        probs = self.predict_proba(X)
        # Tek satır özel durumu: X tek örnek ise
        if probs.ndim == 1:
            p_buy = probs[1]
        else:
            p_buy = probs[0, 1]

        signal = "BUY" if p_buy >= 0.5 else "SELL"
        logger.info(f"[EnsembleModel] p_buy={p_buy:.4f} -> signal={signal}")
        return signal

