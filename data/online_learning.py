# data/online_learning.py

from __future__ import annotations

import os
import logging
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger("system")


class OnlineLearner:
    """
    Basit online (inkremental) öğrenme sınıfı.

    main.py içinde aşağıdaki gibi kullanılmaya uygundur:

        online_learner = OnlineLearner(
            model=batch_model,            # opsiyonel, yoksa kendi SGDClassifier'ını kurar
            feature_columns=feature_cols, # opsiyonel, sadece bilgi amaçlı tutulur
            model_dir="models",           # verilmezse "models" kullanılır
            base_model_name="online_btcusdt"
        )

        # Yeni gelen verilerle:
        online_learner.partial_fit(X_new, y_new)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_columns: Optional[Sequence[str]] = None,
        model_dir: Optional[str] = None,
        base_model_name: str = "online_model",
        classes: Iterable[int] = (0, 1),
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        model          : Eğer partial_fit destekleyen bir model verirsen onu kullanır.
                         Vermezsen kendi SGDClassifier'ını oluşturur.
        feature_columns: Sadece referans için tutulur (log, debug vs).
        model_dir      : Modellerin kaydedileceği klasör. None ise 'models'.
        base_model_name: Kaydedilen model dosyasının temel adı.
        classes        : partial_fit için sınıf listesi (binary: (0,1)).
        """

        self.feature_columns: Optional[List[str]] = (
            list(feature_columns) if feature_columns is not None else None
        )
        self.model_dir = model_dir or "models"
        self.base_model_name = base_model_name
        self.classes = np.array(list(classes))
        self.random_state = random_state

        # Dışarıdan model geldiyse ve partial_fit destekliyorsa onu kullan
        if model is not None and hasattr(model, "partial_fit"):
            self.model = model
            logger.info(
                "[ONLINE] Using provided model with partial_fit for online learning."
            )
        else:
            # Kendi SGDClassifier'ını kur (logistic regression benzeri)
            self.model = SGDClassifier(
                loss="log_loss",
                learning_rate="optimal",
                alpha=0.0001,
                penalty="l2",
                random_state=self.random_state,
            )
            logger.info(
                "[ONLINE] Initialized new SGDClassifier for online learning "
                "(loss=log_loss)."
            )

        logger.info(
            "[ONLINE] OnlineLearner initialized with model_dir=%s, base_model_name=%s, "
            "n_classes=%d",
            self.model_dir,
            self.base_model_name,
            len(self.classes),
        )
        if self.feature_columns is not None:
            logger.info(
                "[ONLINE] feature_columns set with %d columns.", len(self.feature_columns)
            )

        # kwargs içinden şimdilik hiçbir şeyi zorunlu kullanmıyoruz ama
        # gelebilecek extra parametreler hataya sebep olmasın diye yutuyoruz.
        self.extra_params: Dict[str, Any] = dict(kwargs)

    # ------------------------------------------------------------------
    # Yardımcı: numpy/pandas konversiyonları
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numpy_X(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    @staticmethod
    def _to_numpy_y(y: Any) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            return np.asarray(y).ravel()
        return np.asarray(y).ravel()

    # ------------------------------------------------------------------
    # Online eğitim
    # ------------------------------------------------------------------
    def partial_fit(self, X: Any, y: Any) -> None:
        """
        Yeni gelen batch ile online (inkremental) eğitim yapar.

        X : features (DataFrame veya numpy array)
        y : target (Series, array vs.)
        """
        try:
            X_arr = self._to_numpy_X(X)
            y_arr = self._to_numpy_y(y)

            logger.info(
                "[ONLINE] partial_fit called with %d samples, %d features.",
                X_arr.shape[0],
                X_arr.shape[1],
            )

            # İlk çağrıda classes parametresi gerekli, sonrakilerde opsiyonel ama
            # güvenli olsun diye her seferinde geçiyoruz.
            self.model.partial_fit(X_arr, y_arr, classes=self.classes)
            logger.info("[ONLINE] partial_fit completed successfully.")

        except Exception as e:
            logger.exception("[ONLINE] partial_fit failed: %s", str(e))
            raise RuntimeError(f"Online partial_fit failed: {e}") from e

    # ------------------------------------------------------------------
    # Tahmin fonksiyonları
    # ------------------------------------------------------------------
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Olasılık tahmini döner (model destekliyorsa).
        """
        X_arr = self._to_numpy_X(X)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                "[ONLINE] Current model does not support predict_proba."
            )
        return self.model.predict_proba(X_arr)

    def predict(self, X: Any) -> np.ndarray:
        """
        Sınıf etiketi tahmini döner.
        """
        X_arr = self._to_numpy_X(X)
        return self.model.predict(X_arr)

    # ------------------------------------------------------------------
    # Model kaydet / yükle
    # ------------------------------------------------------------------
    def _get_model_path(self) -> str:
        os.makedirs(self.model_dir, exist_ok=True)
        filename = f"{self.base_model_name}.joblib"
        return os.path.join(self.model_dir, filename)

    def save_model(self) -> str:
        """
        Online modelin mevcut halini kaydeder ve dosya yolunu döner.
        """
        path = self._get_model_path()
        joblib.dump(self.model, path)
        logger.info("[ONLINE] Online model saved to %s", path)
        return path

    def load_model(self) -> Any:
        """
        Daha önce kaydedilmiş online modeli yükler.
        """
        path = self._get_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ONLINE] Model file not found: {path}")
        self.model = joblib.load(path)
        logger.info("[ONLINE] Online model loaded from %s", path)
        return self.model
