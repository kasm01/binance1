# models/lightgbm_model.py
from typing import Optional, Dict, Any

import lightgbm as lgb


class LightGBMModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_estimators": 500,
            "verbose": -1,
        }
        self.params = params or default_params
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # 2D proba matrisi döner: [p_class0, p_class1]
        return self.model.predict_proba(X)

