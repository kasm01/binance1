import logging
from typing import List, Tuple, Optional

from sklearn.ensemble import VotingClassifier
from sklearn.base import ClassifierMixin

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Esnek ensemble sarmalayıcı.

    - estimators: [('rf', rf_model), ('lgbm', lgbm_model), ...] listesi
    - voting='soft' varsayılan

    Parametresiz çağrılabilir:
        EnsembleModel()  -> estimators = []
    Bu durumda fit/predict/predict_proba çağırırsan ValueError fırlatır (model yok).
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
        if not self.estimators:
            raise ValueError("No base estimators provided to EnsembleModel.")
        self.ensemble = VotingClassifier(estimators=self.estimators, voting=self.voting)
        logger.info(
            f"[EnsembleModel] Built VotingClassifier with {len(self.estimators)} estimators."
        )

    def fit(self, X, y) -> None:
        if not self.estimators:
            raise ValueError(
                "[EnsembleModel] Cannot fit: no base estimators provided."
            )
        if self.ensemble is None:
            self._build_ensemble()
        self.ensemble.fit(X, y)

    def predict(self, X):
        if self.ensemble is None:
            raise ValueError("[EnsembleModel] Cannot predict: model not initialized.")
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        if self.ensemble is None:
            raise ValueError(
                "[EnsembleModel] Cannot predict_proba: model not initialized."
            )
        return self.ensemble.predict_proba(X)

