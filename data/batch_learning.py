import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

from core.logger import system_logger

logger = logging.getLogger(__name__)


class BatchLearner:
    """
    Batch (toplu) öğrenme yapan basit bir LightGBM eğiticisi.

    Beklenen giriş:
      - features_df: Feature kolonları + target kolonu içeren DataFrame
      - target_column: Label kolonu ismi (örn: "target")

    Çıkış:
      - self.model: sklearn API’si olan (fit/predict/predict_proba) bir model
    """

    def __init__(self, features_df: pd.DataFrame, target_column: str = "target"):
        if features_df is None or len(features_df) == 0:
            raise ValueError("BatchLearner: features_df is empty or None")

        if target_column not in features_df.columns:
            raise ValueError(
                f"BatchLearner: target_column='{target_column}' not in DataFrame columns"
            )

        self.features_df = features_df.copy()
        self.target_column = target_column
        self.model: Optional[LGBMClassifier] = None

    def _prepare_data(self):
        """
        X, y ve class_weight hesaplama.
        """
        df = self.features_df

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column].astype(int)

        n_samples = len(y)
        n_features = X.shape[1]

        # Sınıf dağılımı
        classes, counts = np.unique(y, return_counts=True)
        class_dist = {int(c): int(cnt) for c, cnt in zip(classes, counts)}

        system_logger.info(
            "[BATCH] Preparing data for training: n_samples=%d, n_features=%d, class_dist=%s",
            n_samples,
            n_features,
            class_dist,
        )

        # Class weight (dengesiz veri için)
        if len(classes) > 1:
            cw = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y,
            )
            class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        else:
            class_weight = None

        if class_weight is not None:
            system_logger.info("[BATCH] Computed class_weight=%s", class_weight)
        else:
            system_logger.info("[BATCH] Single class detected, no class_weight used.")

        return X, y, class_weight

    def train(self) -> Optional[LGBMClassifier]:
        """
        Basit bir LightGBM classifier eğitir.
        """
        try:
            X, y, class_weight = self._prepare_data()

            params = {
                "objective": "binary",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "n_estimators": 300,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42,
                "n_jobs": -1,
            }

            if class_weight is not None:
                params["class_weight"] = class_weight

            self.model = LGBMClassifier(**params)

            system_logger.info(
                "[BATCH] Starting LightGBM training with params=%s", params
            )
            self.model.fit(X, y)
            system_logger.info("[BATCH] LightGBM training completed successfully.")

            return self.model

        except Exception as e:
            logger.exception(f"[BATCH] Error while training batch model: {e}")
            system_logger.error(f"[BATCH] Error while training batch model: {e}")
            self.model = None
            return None
