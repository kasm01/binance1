import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


class OnlineLearner:
    """
    SGDClassifier ile online öğrenme yapan basit bir sınıf.

    Özellikler:
      - initial_fit(X, y): ilk eğitim (classes parametresiyle)
      - partial_update(X_new, y_new): yeni gelen verilerle incremental update
      - predict_proba(X_live): canlı veri için olasılık hesaplama
    """

    def __init__(
        self,
        model_dir: str = "models",
        base_model_name: str = "online_model",
        n_classes: int = 2,
        logger=None,
    ):
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.n_classes = n_classes
        self.logger = logger

        os.makedirs(self.model_dir, exist_ok=True)

        self.model: Optional[SGDClassifier] = None
        self.feature_columns: Optional[List[str]] = None

        # Var olan bir model varsa onu yüklemeye çalış
        model_path = self._model_path
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                if self.logger:
                    self.logger.info(
                        "[ONLINE] Loaded existing online model from %s", model_path
                    )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "[ONLINE] Failed to load existing online model (%s): %s",
                        model_path,
                        e,
                        exc_info=True,
                    )
                self.model = None

        # Eğer model yoksa yeni bir SGDClassifier oluştur
        if self.model is None:
            self.model = SGDClassifier(
                loss="log_loss",
                max_iter=10,
                learning_rate="optimal",
                tol=1e-3,
            )
            if self.logger:
                self.logger.info(
                    "[ONLINE] Initialized new SGDClassifier for online learning (loss=log_loss)."
                )

        if self.logger:
            self.logger.info(
                "[ONLINE] OnlineLearner initialized with model_dir=%s, base_model_name=%s, n_classes=%d",
                self.model_dir,
                self.base_model_name,
                self.n_classes,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _model_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.base_model_name}.joblib")

    def _save_model(self) -> None:
        try:
            joblib.dump(self.model, self._model_path)
            if self.logger:
                self.logger.info(
                    "[ONLINE] Online model saved to %s", self._model_path
                )
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "[ONLINE] Failed to save online model to %s: %s",
                    self._model_path,
                    e,
                    exc_info=True,
                )

    def _ensure_feature_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Eğitimde kullanılan feature_columns ile canlı gelen veri kolonlarını hizalar.
        Eksik kolonları 0 ile doldurur, fazla kolonları drop eder.
        """
        if self.feature_columns is None:
            # Eğer ilk sefer ise, mevcut kolonları kullan.
            self.feature_columns = list(X.columns)
            if self.logger:
                self.logger.info(
                    "[ONLINE] feature_columns set with %d columns.", len(self.feature_columns)
                )

        # Kolonları hizala
        X_aligned = X.reindex(columns=self.feature_columns, fill_value=0.0)
        return X_aligned

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        İlk batch verisiyle modeli eğitir.
        """
        X = self._ensure_feature_columns(X)
        y = y.astype(int)

        if self.logger:
            self.logger.info(
                "[ONLINE] initial_fit called with %d samples, %d features.",
                X.shape[0],
                X.shape[1],
            )

        classes = np.arange(self.n_classes)
        self.model.partial_fit(X.values, y.values, classes=classes)

        if self.logger:
            self.logger.info("[ONLINE] initial_fit completed successfully.")

        self._save_model()

    def partial_update(self, X_new: pd.DataFrame, y_new: pd.Series) -> None:
        """
        Yeni gelen verilerle incremental update yapar.
        """
        if X_new is None or len(X_new) == 0:
            if self.logger:
                self.logger.warning("[ONLINE] partial_update called with empty X_new.")
            return

        X_new = self._ensure_feature_columns(X_new)
        y_new = y_new.astype(int)

        if self.logger:
            self.logger.info(
                "[ONLINE] partial_update called with %d samples, %d features.",
                X_new.shape[0],
                X_new.shape[1],
            )

        self.model.partial_fit(X_new.values, y_new.values)

        if self.logger:
            self.logger.info("[ONLINE] partial_update completed successfully.")

        self._save_model()

    def predict_proba(self, X_live: pd.DataFrame) -> np.ndarray:
        """
        Canlı veri için olasılık tahmini.
        """
        X_live = self._ensure_feature_columns(X_live)

        if not hasattr(self.model, "predict_proba"):
            # SGDClassifier normalde predict_proba için loss='log_loss' ile gelir.
            # Yine de güvene almak için kontrol ediyoruz.
            if self.logger:
                self.logger.error(
                    "[ONLINE] Model does not support predict_proba. Check loss='log_loss'."
                )
            raise RuntimeError("Model does not support predict_proba")

        proba = self.model.predict_proba(X_live.values)
        return proba
    def predict_proba_live(self, X):
        """
        Tek bir son satır (veya birkaç satır) için pozitif sınıf (1) olasılığını döndürür.
        Dönen değer float olur.
        """
        if self.model is None:
            raise OnlineLearningException("Online model is not initialized.")

        import numpy as np
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        proba = self.model.predict_proba(X_arr)
        # proba shape: (n_samples, 2) -> pozitif sınıf proba'sını al
        return float(proba[-1, 1])

