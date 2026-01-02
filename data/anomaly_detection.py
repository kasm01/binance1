# data/anomaly_detection.py
from __future__ import annotations

from typing import Optional, List

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.logger import system_logger
from features.schema import normalize_to_schema


class AnomalyDetector:
    """
    Stateless anomaly detection wrapper.

    IsolationForest ile numeric feature'lar üzerinden aykırı gözlemleri filtreler.

    TEK SÖZLEŞME (opsiyonel):
      - schema verilirse, hem fit/predict hem de output normalize edilir:
        missing=0, extra=ignore, order=schema, numeric cleanup
    """

    def __init__(
        self,
        logger=None,
        contamination: float = 0.02,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.logger = logger or system_logger
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

        if n_jobs is None:
            # env override (ANOM_N_JOBS) yoksa -1
            try:
                self.n_jobs = int(os.getenv("ANOM_N_JOBS", "-1"))
            except Exception:
                self.n_jobs = -1
        else:
            self.n_jobs = int(n_jobs)

    def _select_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if self.logger:
            # INFO yerine DEBUG (spam azalt)
            self.logger.debug("[ANOM] Numeric columns for anomaly detection: %s", numeric_cols)
        return numeric_cols

    def filter_anomalies(self, features_df: pd.DataFrame, schema: Optional[List[str]] = None) -> pd.DataFrame:
        """
        IsolationForest ile aykırı değerleri çıkarır.

        Args:
            features_df: Feature engineered dataframe
            schema: (opsiyonel) model sözleşmesi. Verilirse:
                    - anomali X bu schema ile normalize edilir
                    - sonuç df tekrar schema'ya normalize edilir

        Returns:
            clean_df: Aykırı gözlemler çıkarılmış dataframe (index resetlenir)
        """
        if features_df is None or len(features_df) == 0:
            if self.logger:
                self.logger.warning("[ANOM] Empty features_df, skipping anomaly filter.")
            return features_df

        # --- schema verilmişse: anomaliyi de tek sözleşmeden koştur ---
        df_for_anom = features_df
        if schema:
            try:
                df_for_anom = normalize_to_schema(features_df, schema)
                if self.logger:
                    self.logger.info("[ANOM] Using schema-normalized matrix for anomaly detection (n_features=%d).", len(schema))
            except Exception as e:
                if self.logger:
                    self.logger.warning("[ANOM] normalize_to_schema failed; fallback to numeric-only df. err=%r", e)
                df_for_anom = features_df

        numeric_cols = self._select_numeric_columns(df_for_anom)
        if not numeric_cols:
            if self.logger:
                self.logger.warning("[ANOM] No numeric columns found. Skipping anomaly filter.")
            # çıktı schema'ya kilitlensin isteniyorsa burada da normalize edelim
            if schema:
                try:
                    return normalize_to_schema(features_df, schema)
                except Exception:
                    return features_df
            return features_df

        X = df_for_anom[numeric_cols].to_numpy(dtype=float, copy=False)
        n_samples, n_features = X.shape

        # çok az satır varsa isolation forest anlamsız
        if n_samples < 10:
            if self.logger:
                self.logger.info("[ANOM] Too few samples (%d). Skipping anomaly filter.", n_samples)
            if schema:
                try:
                    return normalize_to_schema(features_df, schema)
                except Exception:
                    return features_df
            return features_df

        if self.logger:
            self.logger.info("[ANOM] Running IsolationForest on %d samples, %d numeric features.", n_samples, n_features)

        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        preds = iso.fit_predict(X)  # 1=normal, -1=anomaly
        mask = preds == 1

        removed_count = int((~mask).sum())
        clean_df = features_df.loc[mask].copy()
        clean_df.reset_index(drop=True, inplace=True)

        if self.logger:
            self.logger.info("[ANOM] Removed %d anomalous rows. Remaining: %d", removed_count, clean_df.shape[0])

        # --- ÇIKIŞI da schema'ya kilitle (tek sözleşme) ---
        if schema:
            try:
                clean_df = normalize_to_schema(clean_df, schema)
                if self.logger:
                    self.logger.info("[ANOM] Output normalized to schema (n_features=%d).", len(schema))
            except Exception as e:
                if self.logger:
                    self.logger.warning("[ANOM] Output normalize_to_schema failed; returning raw clean_df. err=%r", e)

        return clean_df

    # Geriye dönük uyumluluk için alias
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Eski API ile uyum için transform() -> filter_anomalies() alias'ı.
        """
        return self.filter_anomalies(features_df)
