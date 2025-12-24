from __future__ import annotations
from typing import Optional, Dict, Any, Union
from pathlib import Path
import joblib
import lightgbm as lgb

class LightGBMModel:
    """
    - Eğer `model_path` verilirse: joblib ile yükler (sklearn API: predict_proba).
    - Verilmezse: params ile yeni LGBMClassifier yaratır (train için).
    """
    def __init__(self, model_path: Optional[Union[str, Path]] = None, params: Optional[Dict[str, Any]] = None):
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
        self.model = None

        if model_path is not None:
            mp = Path(model_path)
            self.model = joblib.load(mp)
        else:
            self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)
        return proba

    @classmethod
    def load(cls, path: str):
        """Load a trained LightGBM model from disk (joblib/pickle)."""
        import joblib
        obj = cls()
        obj.model = joblib.load(path)
        return obj
