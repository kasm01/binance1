# data/batch_learning.py

from __future__ import annotations

import os
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger("system")


class BatchLearner:
    """
    Basit batch öğrenme sınıfı.

    main.py içinde şöyle kullanılmak üzere tasarlandı:

        batch_learner = BatchLearner(
            X=features_df[feature_cols],
            y=features_df["target"],
            # model_dir verilmezse varsayılan "models" klasörünü kullanır
            base_model_name="batch_rf_btcusdt",
        )
        model = batch_learner.fit()
        path = batch_learner.save_model(model)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_dir: str | None = None,
        base_model_name: str = "batch_model",
        random_state: int = 42,
        **model_kwargs: Any,
    ) -> None:
        if X is None or y is None:
            raise ValueError("BatchLearner: X ve y boş olamaz.")

        if len(X) != len(y):
            raise ValueError(
                f"BatchLearner: X ve y uzunlukları uyuşmuyor: {len(X)} vs {len(y)}"
            )

        self.X = X
        self.y = y

        # model_dir verilmezse varsayılan klasör "models" olsun
        self.model_dir = model_dir or "models"
        self.base_model_name = base_model_name
        self.random_state = random_state
        self.model_kwargs: Dict[str, Any] = model_kwargs

        logger.info(
            "[BATCH] BatchLearner initialized with %d samples, %d features.",
            X.shape[0],
            X.shape[1],
        )
        logger.info(
            "[BATCH] model_dir=%s, base_model_name=%s",
            self.model_dir,
            self.base_model_name,
        )

    # ------------------------------------------------------------------
    # Model oluşturma
    # ------------------------------------------------------------------
    def _build_model(self) -> RandomForestClassifier:
        """
        Şimdilik sabit bir RandomForestClassifier kullanıyoruz.
        İstersen ileride XGBoost / LightGBM entegrasyonunu buraya ekleyebiliriz.
        """
        n_estimators = int(self.model_kwargs.get("n_estimators", 300))
        max_depth = self.model_kwargs.get("max_depth", None)
        min_samples_leaf = int(self.model_kwargs.get("min_samples_leaf", 2))

        logger.info(
            "[BATCH] Using RandomForestClassifier(n_estimators=%d, max_depth=%s, min_samples_leaf=%d)",
            n_estimators,
            str(max_depth),
            min_samples_leaf,
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=self.random_state,
            class_weight="balanced",
        )
        return model

    # ------------------------------------------------------------------
    # Eğitim
    # ------------------------------------------------------------------
    def fit(self):
        """
        Batch modeli eğitir ve modeli döner.
        Cloud Run log'unda çağırılan metot TAM OLARAK budur.
        """
        try:
            # X ve y'yi numpy array'e çevir
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

            # Basit train/validation split (sadece metrik loglamak için)
            if len(np.unique(y)) > 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=self.random_state,
                    stratify=y,
                )
            else:
                # Tüm target tek sınıf ise stratify yapmayalım
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=self.random_state,
                )

            model = self._build_model()
            model.fit(X_train, y_train)

            # Küçük validasyon raporu
            try:
                y_pred = model.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                logger.info(
                    "[BATCH] Validation accuracy=%.4f, f1=%.4f",
                    acc,
                    f1,
                )
            except Exception as eval_err:
                logger.warning(
                    "[BATCH] Validation metrics failed: %s", str(eval_err)
                )

            logger.info("[BATCH] Batch model training completed successfully.")
            return model

        except Exception as e:
            logger.exception("[BATCH] Batch training failed: %s", str(e))
            raise RuntimeError(f"Batch training failed: {e}") from e

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

