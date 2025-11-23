# models/catboost_model.py
from typing import Optional, Dict, Any

from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": False,
        }
        self.params = params or default_params
        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # 2D proba matrisi döner: [p_class0, p_class1]
        return self.model.predict_proba(X)

