# data/anomaly_detection.py

from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.logger import system_logger
from features.schema import normalize_to_schema


class AnomalyDetector:
    """
    Stateless anomaly detection wrapper.

    IsolationForest ile numeric feature'lar üzerinden aykırı gözlemleri filtreler.

    TEK SÖZLEŞME:
      - schema verilirse anomaly X'i normalize_to_schema(df, schema) ile alınır
      - böylece 20/22/25 drift'anması engellenir
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
        return df.select_dtypes(include=["number"]).columns.tolist()

    def filter_anomalies(self, features_df: pd.DataFrame, schema: Optional[List[str]] = None) -> pd.DataFrame:
        """
        IsolationForest ile aykırı değerleri çıkarır.

        Args:
            features_df: Feature engineered dataframe
            schema: (opsiyonel) model input schema; verilirse X bu şemaya normalize edilir

        Returns:
            clean_df: Aykırı gözlemler çıkarılmış dataframe (index reset)
        """
        if features_df is None or len(features_df) == 0:
            if self.logger:
                self.logger.warning("[ANOM] Empty features_df, skipping anomaly filter.")
            return features_df

        # --- TEK SÖZLEŞME: schema varsa X'i normalize şemadan al ---
        if schema:
            try:
                df_x = normalize_to_schema(features_df, schema)
            except Exception as e:
                if self.logger:
                    self.logger.warning("[ANOM] normalize_to_schema failed (%r). Fallback to numeric cols.", e)
                df_x = features_df
        else:
            df_x = features_df

        numeric_cols = self._select_numeric_columns(df_x)
        if not numeric_cols:
            if self.logger:
                self.logger.warning("[ANOM] No numeric columns found. Skipping anomaly filter.")
            return features_df

        X = df_x[numeric_cols].to_numpy(dtype=float, copy=False)
        n_samples, n_features = X.shape

        if self.logger:
            self.logger.info("[ANOM] Running IsolationForest on %d samples, %d numeric features.", n_samples, n_features)

        try:
            iso = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            preds = iso.fit_predict(X)  # 1=normal, -1=anomaly
            mask = preds == 1
        except Exception as e:
            if self.logger:
                self.logger.warning("[ANOM] IsolationForest failed (%r). Skipping anomaly filter.", e)
            return features_df

        removed_count = int((~mask).sum())
        clean_df = features_df.loc[mask].copy()
        clean_df.reset_index(drop=True, inplace=True)

        if self.logger:
            self.logger.info("[ANOM] Removed %d anomalous rows. Remaining: %d", removed_count, clean_df.shape[0])

        return clean_df

    # Geriye dönük uyumluluk için alias
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return self.filter_anomalies(features_df)
