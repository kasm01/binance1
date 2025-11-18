import os
import logging
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from core.exceptions import ModelTrainingException

logger = logging.getLogger("system")


class BatchLearner:
    """
    Batch modda LightGBM modeli eğiten sınıf.

    main.py içinde aşağıya benzer şekilde çağrılabilir:

        batch_learner = BatchLearner(
            X=batch_input,                 # pandas DataFrame veya numpy array
            y=target,                      # pandas Series veya numpy array
            feature_cols=feature_cols,     # liste (kolon isimleri)
            target_column="target",        # hedef kolon adı (log için)
            model_dir=env_vars.get("MODEL_DIR", "/app/models"),
            model_name="lgbm_batch"        # opsiyonel
        )
        batch_learner.train()

    Ek argümanlar gelirse **kwargs ile sessizce kabul edilir (hata vermez).
    """

    def __init__(
        self,
        X: Any,
        y: Any,
        feature_cols: Optional[List[str]] = None,
        target_column: str = "target",
        model_dir: str = "/app/models",
        model_name: str = "lgbm_batch",
        **kwargs,
    ):
        # X ve y'yi numpy array'e çevir
        if isinstance(X, pd.DataFrame):
            self.feature_cols = feature_cols or list(X.columns)
            self.X = X[self.feature_cols].values
        else:
            self.X = np.asarray(X)
            self.feature_cols = feature_cols

        if isinstance(y, (pd.Series, pd.DataFrame)):
            # DataFrame gelirse ilk kolonu alalım
            if isinstance(y, pd.DataFrame):
                self.y = y.iloc[:, 0].values
            else:
                self.y = y.values
        else:
            self.y = np.asarray(y)

        self.target_column = target_column

        # Model kayıt yeri
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, f"{model_name}.pkl")

        # LightGBM batch modeli (makul default hiperparametreler)
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )

    # -------------------------------------------------
    # Eğitim
    # -------------------------------------------------
    def train(self) -> None:
        """
        Batch modeli eğitir, basit bir ROC-AUC hesaplar ve modeli diske kaydeder.
        Hata durumunda ModelTrainingException fırlatır.
        """
        try:
            logger.info(
                "[BATCH] Starting LightGBM training on %d samples, %d features.",
                self.X.shape[0],
                self.X.shape[1],
            )

            self.model.fit(self.X, self.y)

            logger.info("[BATCH] LightGBM training completed successfully.")

            # Basit in-sample ROC-AUC metriği (binary classification varsayımı)
            try:
                proba = self.model.predict_proba(self.X)[:, 1]
                auc = roc_auc_score(self.y, proba)
                logger.info("[BATCH] In-sample ROC-AUC: %.4f", auc)
            except Exception as e:
                logger.warning("[BATCH] Could not compute ROC-AUC: %s", e)

            # Modeli kaydet
            payload: Dict[str, Any] = {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "target_column": self.target_column,
            }
            joblib.dump(payload, self.model_path)
            logger.info("[BATCH] Model saved to %s", self.model_path)

        except Exception as e:
            logger.error("[BATCH] Error during LightGBM training: %s", e)
            raise ModelTrainingException(f"Batch training failed: {e}") from e

    # -------------------------------------------------
    # Tahmin
    # -------------------------------------------------
    def predict_proba(self, X_new: Any) -> np.ndarray:
        """
        Yeni veri için pozitif sınıf olasılıklarını döner.
        X_new: DataFrame veya numpy array.
        """
        if isinstance(X_new, pd.DataFrame):
            if self.feature_cols is not None:
                X_arr = X_new[self.feature_cols].values
            else:
                X_arr = X_new.values
        else:
            X_arr = np.asarray(X_new)

        return self.model.predict_proba(X_arr)[:, 1]
