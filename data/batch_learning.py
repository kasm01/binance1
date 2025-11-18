# data/batch_learning.py

from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from core.exceptions import ModelTrainingException

logger = logging.getLogger("system")


class BatchLearner:
    """
    Basit ama sağlam bir batch öğrenme sınıfı.
    - Girdi: X (features), y (target)
    - Çıktı: Eğitilmiş bir sklearn modeli ve diske kaydedilmiş dosya.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_dir: str,
        base_model_name: str = "batch_model",
        model_type: str = "rf",
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Özellik matrisi (n_samples, n_features)
        y : pd.Series / array-like
            Hedef (0/1)
        model_dir : str
            Modelin kaydedileceği klasör yolu.
        base_model_name : str
            Dosya ismi prefix'i (ör: 'batch_model').
        model_type : str
            Model tipi: 'rf', 'xgboost', 'lightgbm', 'catboost' (şimdilik rf odaklı).
        random_state : int
            Rastgelelik kontrolü.
        model_kwargs : dict
            Ek model parametreleri.
        """
        # Temel kontroller
        if X is None or y is None:
            raise ValueError("BatchLearner: X ve y boş olamaz.")

        if len(X) != len(y):
            raise ValueError(
                f"BatchLearner: X ve y uzunlukları uyuşmuyor: {len(X)} vs {len(y)}"
            )

        self.X = X
        self.y = y
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs: Dict[str, Any] = model_kwargs

        logger.info(
            "[BATCH] BatchLearner initialized with %d samples, %d features.",
            X.shape[0],
            X.shape[1],
        )
        logger.info("[BATCH] model_type=%s, model_dir=%s", model_type, model_dir)

    # ------------------------------------------------------------------
    # Model oluşturma
    # ------------------------------------------------------------------
    def _build_model(self):
        """
        model_type'a göre uygun modeli oluşturur.
        Şimdilik en stabil olan RandomForest'ı default yapıyoruz.
        """
        # Default: RandomForestClassifier
        if self.model_type.lower() in ["rf", "random_forest", "sklearn"]:
            n_estimators = int(self.model_kwargs.get("n_estimators", 300))
            max_depth = self.model_kwargs.get("max_depth", None)
            min_samples_leaf = int(self.model_kwargs.get("min_samples_leaf", 2))

            logger.info(
                "[BATCH] Using RandomForestClassifier (n_estimators=%d, max_depth=%s, min_samples_leaf=%d)",
                n_estimators,
                str(max_depth),
                min_samples_leaf,
            )

            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=self.random_state,
                class_weight="balanced",
            )

        # (İstersen ileride xgboost/lightgbm/catboost ekleriz; şimdilik rf yeterli)
        logger.warning(
            "[BATCH] Unknown model_type=%s, falling back to RandomForestClassifier.",
            self.model_type,
        )
        return RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    def fit(self):
        """
        Batch modeli eğitir ve modeli döner.
        main.py içinden:
            batch_learner = BatchLearner(...)
            model = batch_learner.fit()
            path = batch_learner.save_model(model)
        şeklinde kullanılmak için tasarlandı.
        """
        try:
            # X, y'yi numpy/Series olarak hazırlayalım
            if isinstance(self.X, pd.DataFrame):
                X = self.X.values
            else:
                X = np.asarray(self.X)

            if isinstance(self.y, (pd.Series, pd.DataFrame)):
                y = np.asarray(self.y).ravel()
            else:
                y = np.asarray(self.y).ravel()

            logger.info(
                "[BATCH] Starting training on %d samples, %d features.",
                X.shape[0],
                X.shape[1],
            )

            # Basit bir train/validation bölmesi (sadece log için)
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y if len(np.unique(y)) > 1 else None,
            )

            model = self._build_model()
            model.fit(X_train, y_train)

            # Küçük bir validasyon raporu
            try:
                y_pred = model.predict(X_val)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_proba = None

                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                logger.info(
                    "[BATCH] Validation accuracy=%.4f, f1=%.4f",
                    acc,
                    f1,
                )

                if y_proba is not None and len(np.unique(y_val)) > 1:
                    try:
                        auc = roc_auc_score(y_val, y_proba)
                        logger.info("[BATCH] Validation ROC-AUC=%.4f", auc)
                    except Exception:
                        pass

                logger.debug(
                    "[BATCH] Validation classification report:\n%s",
                    classification_report(
                        y_val, y_pred, zero_division=0, digits=4
                    ),
                )

            except Exception as eval_err:
                logger.warning(
                    "[BATCH] Validation metrics failed: %s", str(eval_err)
                )

            logger.info("[BATCH] Batch model training completed successfully.")
            return model

        except Exception as e:
            logger.exception("[BATCH] Batch training failed: %s", str(e))
            raise ModelTrainingException(f"Batch training failed: {e}") from e

    # ------------------------------------------------------------------
    # Model kaydet / yükle
    # ------------------------------------------------------------------
    def _get_model_path(self) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        filename = f"{self.base_model_name}.joblib"
        return os.path.join(self.model_dir, filename)

    def save_model(self, model) -> str:
        """
        Eğitilmiş modeli diske kaydeder ve dosya yolunu döner.
        """
        path = self._get_model_path()
        joblib.dump(model, path)
        logger.info("[BATCH] Model saved to %s", path)
        return path

    def load_model(self):
        """
        Kaydedilmiş modeli diskte bulup yükler.
        """
        path = self._get_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"[BATCH] Model file not found: {path}")
        model = joblib.load(path)
        logger.info("[BATCH] Model loaded from %s", path)
        return model
