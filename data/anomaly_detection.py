# data/anomaly_detection.py

from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.logger import system_logger


class AnomalyDetector:
    """
    Stateless anomaly detection wrapper.

    Kullanım:
        detector = AnomalyDetector()
        clean_df = detector.filter_anomalies(features_df)

    IsolationForest ile numeric feature'lar üzerinden aykırı gözlemleri filtreler.
    """

    def __init__(
        self,
        logger=None,
        contamination: float = 0.02,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.logger = logger or system_logger
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

    def _select_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if self.logger:
            self.logger.info(
                "[ANOM] Numeric columns for anomaly detection: %s",
                numeric_cols,
            )
        return numeric_cols

    def filter_anomalies(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        IsolationForest ile aykırı değerleri çıkarır.

        Girdi:
            features_df: Feature engineered dataframe

        Çıktı:
            clean_df: Aykırı gözlemler çıkarılmış dataframe (index preserve edilmez, resetlenir)
        """
        if features_df is None or len(features_df) == 0:
            if self.logger:
                self.logger.warning("[ANOM] Empty features_df, skipping anomaly filter.")
            return features_df

        numeric_cols = self._select_numeric_columns(features_df)
        if not numeric_cols:
            if self.logger:
                self.logger.warning(
                    "[ANOM] No numeric columns found in features_df. "
                    "Skipping anomaly filter."
                )
            return features_df

        X = features_df[numeric_cols].values
        n_samples, n_features = X.shape

        if self.logger:
            self.logger.info(
                "[ANOM] Running IsolationForest on %d samples, %d numeric features.",
                n_samples,
                n_features,
            )

        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        preds = iso.fit_predict(X)  # 1 = normal, -1 = anomaly
        mask = preds == 1

        removed_count = int((~mask).sum())
        clean_df = features_df.loc[mask].copy()
        clean_df.reset_index(drop=True, inplace=True)

        if self.logger:
            self.logger.info(
                "[ANOM] Removed %d anomalous rows. Remaining: %d",
                removed_count,
                clean_df.shape[0],
            )
            self.logger.info(
                "[ANOM] After anomaly filter: %d rows remain.",
                clean_df.shape[0],
            )

        return clean_df

    # Geriye dönük uyumluluk için alias
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Eski API ile uyum için transform() -> filter_anomalies() alias'ı.
        """
        return self.filter_anomalies(features_df)

