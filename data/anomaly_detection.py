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

    - schema verilirse: önce normalize_to_schema(schema) uygulanır (tek sözleşme)
    - çıktı index resetlenir
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
            self.logger.debug("[ANOM] Numeric columns for anomaly detection: %s", numeric_cols)
        return numeric_cols

    def filter_anomalies(self, features_df: pd.DataFrame, schema: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Girdi:
          - features_df: feature engineered DF
          - schema (ops): tek sözleşme kolon sırası. Verilirse normalize_to_schema uygulanır.

        Çıktı:
          - clean_df: aykırılar çıkarılmış DF (reset_index)
        """
        if features_df is None or len(features_df) == 0:
            if self.logger:
                self.logger.warning("[ANOM] Empty features_df, skipping anomaly filter.")
            return features_df

        df = features_df

        # TEK SÖZLEŞME: schema verilirse önce normalize et
        if isinstance(schema, list) and schema:
            try:
                df = normalize_to_schema(df, schema)
            except Exception as e:
                if self.logger:
                    self.logger.warning("[ANOM] normalize_to_schema failed, continuing raw. err=%s", e)
                df = features_df

        numeric_cols = self._select_numeric_columns(df)
        if not numeric_cols:
            if self.logger:
                self.logger.warning("[ANOM] No numeric columns found. Skipping anomaly filter.")
            return df.reset_index(drop=True)

        X = df[numeric_cols].to_numpy(dtype=float, copy=False)

        if self.logger:
            self.logger.info(
                "[ANOM] IsolationForest: n=%d, d=%d, contamination=%.4f",
                X.shape[0],
                X.shape[1],
                self.contamination,
            )

        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        preds = iso.fit_predict(X)  # 1=normal, -1=anomaly
        mask = preds == 1

        removed = int((~mask).sum())
        clean_df = df.loc[mask].copy().reset_index(drop=True)

        if self.logger:
            self.logger.info("[ANOM] Removed %d anomalous rows. Remaining=%d", removed, clean_df.shape[0])

        return clean_df

    # Backwards compat
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return self.filter_anomalies(features_df)
