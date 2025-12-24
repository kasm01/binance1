from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, List

import numpy as np

from models.fallback_model import FallbackModel


def _env_path(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def _to_p1(pred: Any) -> np.ndarray:
    """
    Normalize model outputs to p(class=1) array with shape (n,).
    Accepts:
      - float
      - list/np.ndarray shape (n,), (n,1), (n,2)
    """
    if pred is None:
        return np.array([], dtype=float)

    # float -> single
    if isinstance(pred, (float, int)):
        return np.array([float(pred)], dtype=float)

    arr = np.asarray(pred, dtype=float)

    # (n,2) -> class1
    if arr.ndim == 2 and arr.shape[1] >= 2:
        p1 = arr[:, 1]
    # (n,1) -> squeeze
    elif arr.ndim == 2 and arr.shape[1] == 1:
        p1 = arr[:, 0]
    # (n,) -> already p1
    elif arr.ndim == 1:
        p1 = arr
    else:
        # fallback: flatten first
        p1 = arr.reshape(-1)

    # clamp
    p1 = np.clip(p1, 0.0, 1.0)
    return p1.astype(float)


class EnsembleModel:
    """
    Robust ensemble loader:
      - LightGBM (joblib/pickle model file)
      - CatBoost (.cbm)
      - Optional LSTM (.keras) (if you later enable)
      - FallbackModel always available
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = str(model_dir)

        self.lightgbm_model: Optional[Any] = None
        self.catboost_model: Optional[Any] = None
        self.lstm_model: Optional[Any] = None

        self.fallback_model = FallbackModel()
        self.load_models()

    def load_models(self) -> None:
        mdir = Path(self.model_dir)

        # ---------- LightGBM ----------
        # default path: models/lightgbm_model.pkl (env override: LGB_MODEL_PATH)
        lgb_path = _env_path("LGB_MODEL_PATH", str(mdir / "lightgbm_model.pkl"))
        if _exists(lgb_path):
            try:
                import joblib
                self.lightgbm_model = joblib.load(lgb_path)
            except Exception:
                self.lightgbm_model = None

        # ---------- CatBoost ----------
        cb_path = _env_path("CB_MODEL_PATH", str(mdir / "catboost_model.cbm"))
        if _exists(cb_path):
            try:
                from models.catboost_model import CatBoostModel
                self.catboost_model = CatBoostModel(cb_path)
            except Exception:
                self.catboost_model = None

        # ---------- LSTM (optional) ----------
        # enable later if you have a loader class; keep safe for now
        lstm_path = _env_path("LSTM_MODEL_PATH", str(mdir / "lstm_model.keras"))
        if _exists(lstm_path):
            try:
                # If you later create models/lstm_model.py with load() or similar, plug it here.
                # For now: do nothing unless explicitly enabled.
                if os.getenv("ENABLE_LSTM_ENSEMBLE", "0") in ("1", "true", "True"):
                    from tensorflow.keras.models import load_model
                    self.lstm_model = load_model(lstm_path)
            except Exception:
                self.lstm_model = None

    def _predict_lgb(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.lightgbm_model is None:
            return None
        try:
            pred = self.lightgbm_model.predict_proba(X)
            return _to_p1(pred)
        except Exception:
            return None

    def _predict_cb(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.catboost_model is None:
            return None
        try:
            pred = self.catboost_model.predict_proba(X)
            return _to_p1(pred)
        except Exception:
            return None

    def _predict_lstm(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.lstm_model is None:
            return None
        try:
            # keras model output usually (n,1) sigmoid
            pred = self.lstm_model.predict(X, verbose=0)
            return _to_p1(pred)
        except Exception:
            return None

    def predict_proba(self, X: np.ndarray) -> float:
        """
        Returns mean p1 across available models for the LAST row of X (live usage).
        If X has one row -> that row.
        If no model works -> fallback.
        """
        Xn = np.asarray(X, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)

        preds: List[np.ndarray] = []

        for fn in (self._predict_lgb, self._predict_cb, self._predict_lstm):
            p1 = fn(Xn)
            if p1 is not None and p1.size > 0:
                preds.append(p1)

        if not preds:
            # fallback expected to return float or (n,) like; normalize
            fb = _to_p1(self.fallback_model.predict_proba(Xn))
            if fb.size == 0:
                return 0.5
            return float(fb[-1])

        # align lengths if different (take last element of each)
        last_vals = [float(np.clip(p[-1], 0.0, 1.0)) for p in preds]
        return float(np.mean(last_vals))

    def predict(self, X_row) -> float:
        return self.predict_proba(np.array([X_row], dtype=float))
