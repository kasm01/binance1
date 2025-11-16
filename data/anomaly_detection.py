import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Feature engineer edilmiş veride (features DataFrame) anomali tespiti yapar.

    Kullanım:
        detector = AnomalyDetector(features_df)
        clean_df = detector.remove_anomalies()
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        """
        :param features_df: FeatureEngineer çıkışı olan DataFrame
        :param contamination: Beklenen anomali oranı (0.01 = %1)
        :param random_state: Reprodüksiyon için sabit seed
        """
        self.features_df = features_df.copy() if features_df is not None else pd.DataFrame()
        self.contamination = contamination
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None

    def _select_feature_columns(self) -> pd.DataFrame:
        """
        Model için sayısal feature kolonlarını seçer.
        OHLC gibi kolonlar da dahil olabilir, ama temel amaç:
        - float / int tipindeki kolonlar
        """
        if self.features_df.empty:
            return self.features_df

        numeric_df = self.features_df.select_dtypes(include=["float64", "float32", "int64", "int32"])
        if numeric_df.empty:
            raise DataProcessingException("AnomalyDetector: sayısal feature kolonları bulunamadı.")

        return numeric_df

    def fit(self):
        """
        IsolationForest modelini fit eder.
        """
        if self.features_df.empty:
            logger.warning("[AnomalyDetector] Boş DataFrame, fit atlanıyor.")
            return

        try:
            X = self._select_feature_columns().values

            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
                n_jobs=-1,
            )
            self._model.fit(X)

            logger.info("[AnomalyDetector] IsolationForest modeli başarıyla eğitildi.")
        except Exception as e:
            logger.exception(f"[AnomalyDetector] Model fit hatası: {e}")
            raise DataProcessingException(f"AnomalyDetector fit failed: {e}") from e

    def mark_anomalies(self) -> pd.DataFrame:
        """
        DataFrame'e 'is_anomaly' kolonu ekler.

        Dönüş:
            features_df + 'is_anomaly' (0 = normal, 1 = anomali)
        """
        if self.features_df.empty:
            logger.warning("[AnomalyDetector] Boş DataFrame, anomaly flag eklenemedi.")
            return self.features_df

        if self._model is None:
            self.fit()

        try:
            X = self._select_feature_columns().values

            # IsolationForest: -1 = anomali, 1 = normal
            preds = self._model.predict(X)
            is_anomaly = (preds == -1).astype(int)

            df_flagged = self.features_df.copy()
            df_flagged["is_anomaly"] = is_anomaly

            anomaly_rate = df_flagged["is_anomaly"].mean()
            logger.info(
                "[AnomalyDetector] Anomali oranı: %.4f (%.2f%%)",
                anomaly_rate,
                anomaly_rate * 100,
            )

            return df_flagged
        except Exception as e:
            logger.exception(f"[AnomalyDetector] Anomali işaretleme hatası: {e}")
            raise DataProcessingException(f"AnomalyDetector mark_anomalies failed: {e}") from e

    def remove_anomalies(self) -> pd.DataFrame:
        """
        Anomalileri filtreleyip sadece normal kayıtları döner.

        Dönüş:
            Sadece is_anomaly == 0 olan satırları içeren DataFrame
        """
        try:
            df_flagged = self.mark_anomalies()
            if "is_anomaly" not in df_flagged.columns:
                logger.warning("[AnomalyDetector] is_anomaly kolonu yok, veri aynen döndürülüyor.")
                return df_flagged

            clean_df = df_flagged[df_flagged["is_anomaly"] == 0].drop(columns=["is_anomaly"])

            logger.info(
                "[AnomalyDetector] Temizlenen satır sayısı: %d / %d",
                len(df_flagged) - len(clean_df),
                len(df_flagged),
            )
            return clean_df
        except Exception as e:
            logger.exception(f"[AnomalyDetector] remove_anomalies hatası: {e}")
            raise DataProcessingException(f"AnomalyDetector remove_anomalies failed: {e}") from e

