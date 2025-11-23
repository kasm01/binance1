# models/hyperparameter_tuner.py
from typing import Dict, Any, Tuple

from sklearn.model_selection import GridSearchCV


class HyperparameterTuner:
    def __init__(
        self,
        model,
        param_grid: Dict[str, Any],
        cv: int = 3,
        scoring: str = "accuracy",
    ):
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.best_params_: Dict[str, Any] | None = None

    def tune(self, X, y) -> Tuple[Any, Dict[str, Any]]:
        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        grid.fit(X, y)
        self.best_model = grid.best_estimator_
        self.best_params_ = grid.best_params_
        return self.best_model, grid.best_params_

