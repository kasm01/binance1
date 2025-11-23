# models/ensemble_model.py

import logging
from typing import List, Tuple, Optional

from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model orchestrator:
      - Accepts a list of estimators: [(name, model), ...]
      - Uses VotingClassifier (soft voting)
      - If no models provided, starts empty and waits for training
    """

    def __init__(
        self,
        estimators: Optional[List[Tuple[str, ClassifierMixin]]] = None,
        voting: str = "soft",
    ) -> None:
        self.estimators: List[Tuple[str, ClassifierMixin]] = estimators or []
        self.voting = voting
        self.ensemble: Optional[VotingClassifier] = None

        if self.estimators:
            self._build_ensemble()

    def _build_ensemble(self) -> None:
        try:
            self.ensemble = VotingClassifier(
                estimators=self.estimators,
                voting=self.voting,
            )
            logger.info("[EnsembleModel] Ensemble built successfully.")
        except Exception as e:
            logger.error(f"[EnsembleModel] Build failed: {e}", exc_info=True)
            self.ensemble = None

    def fit(self, X, y):
        if not self.ensemble:
            logger.warning("[EnsembleModel] No ensemble exists. Cannot fit.")
            return
        self.ensemble.fit(X, y)

    def predict(self, X):
        if not self.ensemble:
            logger.warning("[EnsembleModel] No ensemble exists. predict() fallback triggered.")
            return 0

        return self.ensemble.predict(X)

    def predict_proba(self, X):
        if not self.ensemble:
            logger.warning("[EnsembleModel] No ensemble exists. predict_proba() fallback.")
            return [[0.5, 0.5]]

        return self.ensemble.predict_proba(X)

