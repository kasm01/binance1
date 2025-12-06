import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model


class HybridModel:
    """
    Hybrid scorer: combines SGD (online model) and LSTM models.

    - SGD: online_model_<interval>_best.joblib
    - LSTM: lstm_long_<interval>.h5, lstm_short_<interval>.h5, lstm_scaler_<interval>.joblib
    - Meta: model_meta_<interval>.json
    """

    def __init__(self, model_dir: str, interval: str, logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir
        self.interval = interval
        self.logger = logger or logging.getLogger("system")

        # hybrid weight: p_hybrid = alpha * p_lstm + (1 - alpha) * p_sgd
        self.alpha: float = 0.6

        # models
        self.sgd_model = None
        self.lstm_long = None
        self.lstm_short = None
        self.lstm_scaler = None

        # meta info
        self.meta: Dict[str, Any] = {
            "best_auc": 0.0,
            "best_side": "best",
            "use_lstm_hybrid": False,
            "lstm_long_auc": 0.0,
            "lstm_short_auc": 0.0,
            "seq_len": 32,
        }
        self.use_lstm_hybrid: bool = False

        # load from disk
        self._load_sgd_model()
        self._load_meta()
        self._load_lstm_models()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, level: int, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.log(level, msg, *args)
        else:
            print(msg % args if args else msg)

    def _load_sgd_model(self) -> None:
        path = os.path.join(self.model_dir, f"online_model_{self.interval}_best.joblib")
        try:
            self.sgd_model = load(path)
            self._log(
                logging.INFO,
                "[HYBRID] SGD model loaded from %s (interval=%s)",
                path,
                self.interval,
            )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[HYBRID] Could not load SGD model from %s: %s",
                path,
                e,
            )
            self.sgd_model = None

    def _load_meta(self) -> None:
        path = os.path.join(self.model_dir, f"model_meta_{self.interval}.json")
        try:
            with open(path, "r") as f:
                meta_file = json.load(f) or {}
            self.meta.update(meta_file)
            self.use_lstm_hybrid = bool(self.meta.get("use_lstm_hybrid", False))
            self._log(logging.INFO, "[HYBRID] Meta loaded from %s", path)
        except FileNotFoundError:
            self._log(
                logging.WARNING,
                "[HYBRID] Meta file %s not found. Using defaults.",
                path,
            )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[HYBRID] Error while loading meta from %s: %s",
                path,
                e,
            )

    def _load_lstm_models(self) -> None:
        if not self.use_lstm_hybrid:
            return

        long_path = os.path.join(self.model_dir, f"lstm_long_{self.interval}.h5")
        short_path = os.path.join(self.model_dir, f"lstm_short_{self.interval}.h5")
        scaler_path = os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.joblib")

        try:
            self.lstm_long = load_model(long_path)
            self.lstm_short = load_model(short_path)
            self.lstm_scaler = load(scaler_path)

            self._log(
                logging.INFO,
                "[HYBRID] LSTM models and scaler loaded for %s (use_lstm_hybrid=True)",
                self.interval,
            )
        except Exception as e:
            self._log(
                logging.WARNING,
                "[HYBRID] Could not load LSTM models or scaler (%s): %s. Disabling LSTM.",
                self.interval,
                e,
            )
            self.lstm_long = None
            self.lstm_short = None
            self.lstm_scaler = None
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False

    # ------------------------------------------------------------------
    # Feature matrix helper
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numeric_matrix(
        X: Union[np.ndarray, pd.DataFrame, pd.Series, list]
    ) -> np.ndarray:
        """
        Convert X to numeric numpy array.

        - DataFrame: keep only numeric columns (drop timestamps, strings etc)
        - Series: reshape to (n, 1)
        - ndarray: if not numeric, convert via DataFrame and select numeric cols
        - list: convert to ndarray and recurse
        """
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
            if X_num.empty:
                raise ValueError("No numeric columns in DataFrame")
            return X_num.to_numpy(dtype=float)

        if isinstance(X, pd.Series):
            if pd.api.types.is_numeric_dtype(X.dtype):
                return X.to_numpy(dtype=float).reshape(-1, 1)
            raise ValueError("Series is not numeric")

        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            if np.issubdtype(X.dtype, np.number):
                return X.astype(float)
            # mixed dtype -> use DataFrame and select numeric columns
            df = pd.DataFrame(X)
            X_num = df.select_dtypes(include=[np.number])
            if X_num.empty:
                raise ValueError("No numeric columns in ndarray")
            return X_num.to_numpy(dtype=float)

        raise TypeError(f"Unsupported X type: {type(X)}")

    # ------------------------------------------------------------------
    # SGD part
    # ------------------------------------------------------------------
    def _predict_sgd_proba(self, X: np.ndarray) -> np.ndarray:
        if self.sgd_model is None:
            return np.full(X.shape[0], 0.5, dtype=float)

        try:
            proba = self.sgd_model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.reshape(-1)
        except Exception:
            return np.full(X.shape[0], 0.5, dtype=float)

    # ------------------------------------------------------------------
    # LSTM part
    # ------------------------------------------------------------------
    def _build_lstm_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = int(self.meta.get("seq_len", 32))
        if X.shape[0] < seq_len:
            raise ValueError("Not enough rows to build sequences")

        seqs = []
        for i in range(seq_len, X.shape[0] + 1):
            seqs.append(X[i - seq_len : i, :])
        return np.asarray(seqs)

    def _predict_lstm_proba(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        if not (
            self.use_lstm_hybrid
            and self.lstm_long is not None
            and self.lstm_short is not None
            and self.lstm_scaler is not None
        ):
            debug["lstm_used"] = False
            return np.full(X.shape[0], 0.5, dtype=float), debug

        try:
            X_scaled = self.lstm_scaler.transform(X)
            seqs = self._build_lstm_sequences(X_scaled)

            p_long = self.lstm_long.predict(seqs, verbose=0).reshape(-1)
            p_short = self.lstm_short.predict(seqs, verbose=0).reshape(-1)
            p_lstm = 0.5 * (p_long + p_short)

            # align length back to X rows
            if len(p_lstm) < X.shape[0]:
                pad_len = X.shape[0] - len(p_lstm)
                p_lstm = np.concatenate([np.full(pad_len, p_lstm[0]), p_lstm])

            debug.update(
                {
                    "lstm_used": True,
                    "p_long_mean": float(p_long.mean()),
                    "p_short_mean": float(p_short.mean()),
                }
            )
            return p_lstm, debug
        except Exception as e:
            debug["lstm_used"] = False
            debug["error"] = str(e)
            return np.full(X.shape[0], 0.5, dtype=float), debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_proba(
        self, X: Union[np.ndarray, pd.DataFrame, pd.Series, list]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        try:
            X_arr = self._to_numeric_matrix(X)
        except Exception as e:
            self._log(
                logging.WARNING,
                "[HYBRID] Failed to convert X to numeric matrix: %s. Using 0.5 uniform.",
                e,
            )
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            p_uniform = np.full(n, 0.5, dtype=float)
            debug.update(
                {
                    "mode": "uniform_fallback",
                    "error": str(e),
                    "p_sgd_mean": 0.0,
                    "p_lstm_mean": 0.0,
                    "p_hybrid_mean": 0.5,
                    "best_auc": float(self.meta.get("best_auc", 0.0)),
                    "best_side": self.meta.get("best_side", "best"),
                    "use_lstm_hybrid": bool(self.use_lstm_hybrid),
                }
            )
            return p_uniform, debug

        # SGD
        p_sgd = self._predict_sgd_proba(X_arr)

        # LSTM
        p_lstm, lstm_debug = self._predict_lstm_proba(X_arr)

        # Combine
        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            p_hybrid = self.alpha * p_lstm + (1.0 - self.alpha) * p_sgd
            mode = "lstm+sgd"
        else:
            p_hybrid = p_sgd
            mode = "sgd_only"

        debug.update(
            {
                "mode": mode,
                "p_sgd_mean": float(p_sgd.mean()),
                "p_lstm_mean": float(p_lstm.mean()) if lstm_debug else 0.0,
                "p_hybrid_mean": float(p_hybrid.mean()),
                "best_auc": float(self.meta.get("best_auc", 0.0)),
                "best_side": self.meta.get("best_side", "best"),
                "use_lstm_hybrid": bool(self.use_lstm_hybrid),
            }
        )
        debug.update(lstm_debug)

        self._log(
            logging.INFO,
            "[HYBRID] p_sgd_mean=%.4f, p_lstm_mean=%.4f, p_hybrid_mean=%.4f, best_auc=%.4f, best_side=%s",
            debug["p_sgd_mean"],
            debug["p_lstm_mean"],
            debug["p_hybrid_mean"],
            debug["best_auc"],
            debug["best_side"],
        )

        return p_hybrid, debug

