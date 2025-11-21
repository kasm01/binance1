import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


system_logger = logging.getLogger("system")


class AnomalyDetector:
    """
    Feature seviyesinde basit anomali tespitçisi.

    main.py içinde şöyle kullanıyoruz:
        anomaly_detector = AnomalyDetector(features_df=features_df)
        features_df = anomaly_detector.detect_and_handle_anomalies()

    Burada IsolationForest ile numerik kolonlara bakıp
    anomali olan satırları filtreliyoruz.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        contamination: float = 0.02,
        logger: Optional[logging.Logger] = None,
    ):
        self.features_df = features_df
        self.contamination = contamination
        self.logger = logger or system_logger

    def detect_and_handle_anomalies(self) -> pd.DataFrame:
        """
        - Sadece numerik kolonlara bakar
        - IsolationForest ile anomali skoru çıkarır
        - Anomali olanları (label = -1) atar, geri kalanı döner
        """
        df = self.features_df

        if df is None or df.empty:
            self.logger.warning("[ANOM] Received empty features_df, skipping anomaly detection.")
            return df

        # Numerik kolonları seç
        num_df = df.select_dtypes(include=[np.number])

        if num_df.empty:
            self.logger.warning("[ANOM] No numeric columns in features_df, skipping anomaly detection.")
            return df

        # Çok az satır varsa model anlamlı olmaz, direkt geç
        if len(num_df) < 50:
            self.logger.info(
                "[ANOM] Only %d rows available (<50), skipping anomaly detection.",
                len(num_df),
            )
            return df

        try:
            self.logger.info(
                "[ANOM] Running IsolationForest on %d samples, %d numeric features.",
                num_df.shape[0],
                num_df.shape[1],
            )

            iso = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,
                n_jobs=-1,
            )
            preds = iso.fit_predict(num_df)

            # IsolationForest: 1 = normal, -1 = anomali
            mask = preds == 1
            kept = df.loc[mask].copy()
            removed_count = int((~mask).sum())

            self.logger.info(
                "[ANOM] Removed %d anomalous rows. Remaining: %d",
                removed_count,
                kept.shape[0],
            )

            return kept

        except Exception as e:
            self.logger.error(
                "[ANOM] Error during anomaly detection, returning original features_df: %s",
                e,
                exc_info=True,
            )
            return df

