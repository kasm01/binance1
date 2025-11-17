import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)


class BatchLearner:
    """
    Temizlenmiş feature DataFrame'i ile batch (offline) model eğitimi yapar.

    Beklenti:
      - DataFrame içinde 'target' isimli label kolonu bulunur.
      - Geriye, predict/predict_proba çağrılabilen bir model döner.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        target_column: str = "target",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.features_df = (
            features_df.copy() if features_df is not None else pd.DataFrame()
        )
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None

    def _split_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        X ve y arraylerini üretir.
        """
        if self.features_df.empty:
            raise DataProcessingException("BatchLearner: boş DataFrame ile eğitim yapılamaz.")

        if self.target_column not in self.features_df.columns:
            raise DataProcessingException(
                f"BatchLearner: '{self.target_column}' hedef kolonu DataFrame içinde bulunamadı."
            )

        df = self.features_df.dropna(subset=[self.target_column]).copy()

        y = df[self.target_column].astype(int).values

        # Sadece sayısal kolonlar (target hariç)
        X_df = df.drop(columns=[self.target_column]).select_dtypes(
            include=["float32", "float64", "int32", "int64"]
        )

        if X_df.empty:
            raise DataProcessingException(
                "BatchLearner: Sayısal feature kolonları bulunamadı (X boş)."
            )

        X = X_df.values

        # 🔎 Label dağılımını logla
        pos_ratio = float((y == 1).mean())
        num_pos = int((y == 1).sum())
        num_neg = int(len(y) - num_pos)
        logger.info(
            "[BatchLearner] Label stats -> pos=%d, neg=%d, n=%d, pos_ratio=%.3f (%.1f%%)",
            num_pos,
            num_neg,
            len(y),
            pos_ratio,
            pos_ratio * 100,
        )

        return X, y


    def train(self) -> Optional[RandomForestClassifier]:
        """
        Modeli eğitir ve döner. Hata durumunda None dönebilir.
        """
        try:
            X, y = self._split_xy()

            # Çok az positive varsa yine de loglayıp eğitelim ama uyarı verelim
            num_pos = int((y == 1).sum())
            if num_pos < 20:
                logger.warning(
                    "[BatchLearner] Positive sample sayısı çok düşük: %d (n=%d). "
                    "Model dengesiz olabilir.",
                    num_pos,
                    len(y),
                )

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=True,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            # Daha kontrollü, sınıf dengesini gözeten RF
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            )

            self.model.fit(X_train, y_train)

            # Basit validasyon log'u
            y_pred = self.model.predict(X_val)
            report = classification_report(
                y_val, y_pred, output_dict=False, zero_division=0
            )
            logger.info("[BatchLearner] RandomForest eğitim tamamlandı.\n%s", report)

            return self.model

        except DataProcessingException:
            # Yukarıda zaten loglanmış olabilir
            raise
        except Exception as e:
            logger.exception(f"[BatchLearner] train hatası: {e}")
            raise DataProcessingException(f"BatchLearner train failed: {e}") from e
