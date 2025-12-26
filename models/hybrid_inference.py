from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load

from features.pipeline import make_matrix
from models.sgd_helper_runtime import SGDHelperRuntime

# ----------------------------------------------------------------------
# TensorFlow / LSTM opsiyonel import
# ----------------------------------------------------------------------
try:
    from tensorflow.keras.models import load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

    def load_model(*_args, **_kwargs):
        raise RuntimeError("TensorFlow is not available; LSTM disabled.")


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

        # --- SGD helper (aux) ---
        self.sgd_helper = None
        self.enable_sgd_helper = str(os.getenv("ENABLE_SGD_HELPER", "1")).lower() in ("1", "true", "yes", "on")
        self.sgd_helper_sat_thr = float(os.getenv("SGD_HELPER_SAT_THR", "0.95"))

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

        # debug state
        self.last_debug: Dict[str, Any] = {}

        # load from disk
        self._load_sgd_model()
        self._load_meta()

        # ENV HYBRID_MODE ile meta'yı override et
        env_flag = os.getenv("HYBRID_MODE")
        if env_flag is not None:
            v = env_flag.strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                self.use_lstm_hybrid = True
            elif v in ("0", "false", "no", "n", "off"):
                self.use_lstm_hybrid = False

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
            self._log(logging.INFO, "[HYBRID] SGD model loaded from %s (interval=%s)", path, self.interval)
        except Exception as e:
            self._log(logging.WARNING, "[HYBRID] Could not load SGD model from %s: %s", path, e)
            self.sgd_model = None

    def _load_meta(self) -> None:
        path = os.path.join(self.model_dir, f"model_meta_{self.interval}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta_file = json.load(f) or {}
            self.meta.update(meta_file)
            self.use_lstm_hybrid = bool(self.meta.get("use_lstm_hybrid", False))
            self._log(logging.INFO, "[HYBRID] Meta loaded from %s", path)

            # --- load SGD helper bundle (optional) ---
            try:
                _disable_sgd = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")
                if self.enable_sgd_helper and (not _disable_sgd):
                    helper_path = os.getenv("SGD_HELPER_PATH", os.path.join(self.model_dir, "sgd_helper.joblib"))
                    if os.path.exists(helper_path):
                        self.sgd_helper = SGDHelperRuntime(helper_path)
                        self._log(logging.INFO, "[HYBRID] SGD helper loaded from %s", helper_path)
                    else:
                        self._log(logging.WARNING, "[HYBRID] SGD helper not found: %s", helper_path)
            except Exception as _e:
                self._log(logging.WARNING, "[HYBRID] SGD helper load failed: %s", _e)

        except FileNotFoundError:
            self._log(logging.WARNING, "[HYBRID] Meta file %s not found. Using defaults.", path)
        except Exception as e:
            self._log(logging.WARNING, "[HYBRID] Error while loading meta from %s: %s", path, e)

    def _load_lstm_models(self) -> None:
        if not self.use_lstm_hybrid:
            return
        if not TENSORFLOW_AVAILABLE:
            self._log(logging.WARNING, "[HYBRID] TensorFlow not available, LSTM disabled.")
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        long_path = os.path.join(self.model_dir, f"lstm_long_{self.interval}.h5")
        short_path = os.path.join(self.model_dir, f"lstm_short_{self.interval}.h5")
        scaler_path = os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.joblib")

        try:
            self.lstm_long = load_model(long_path)
            self.lstm_short = load_model(short_path)
            self.lstm_scaler = load(scaler_path)
            self._log(logging.INFO, "[HYBRID] LSTM models and scaler loaded for %s (use_lstm_hybrid=True)", self.interval)
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
    def _to_numeric_matrix(X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:
        """
        Convert X to numeric numpy array.

        - DataFrame: keep only numeric columns
        - Series: reshape to (n, 1)
        - ndarray: if mixed dtype, convert via DataFrame and select numeric cols
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
        # runtime switch: disable SGD
        _disable = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")
        if _disable:
            return np.full(X.shape[0], 0.5, dtype=float)

        if self.sgd_model is None:
            return np.full(X.shape[0], 0.5, dtype=float)

        try:
            proba = self.sgd_model.predict_proba(X)
        except Exception as e:
            # scaler/feature mismatch vb. durumda fallback
            self._log(logging.WARNING, "[HYBRID] SGD predict_proba failed: %r", e)
            return np.full(X.shape[0], 0.5, dtype=float)

        # proba -> p1
        try:
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = proba[:, 1]
            else:
                p1 = np.asarray(proba, dtype=float).reshape(-1)
        except Exception:
            return np.full(X.shape[0], 0.5, dtype=float)

        # === SGD_HELPER_RUNTIME (auto) ===
        # center->0.5, compress around 0.5, optional disable
        try:
            _center = str(os.getenv("SGD_CENTER", "1")).lower() in ("1", "true", "yes", "on")
            _band = float(os.getenv("SGD_BAND", "0.10"))
        except Exception:
            _center, _band = True, 0.10

        try:
            p1 = np.asarray(p1, dtype=float)
            p1 = np.clip(p1, 0.0, 1.0)
            if _center:
                p1 = p1 - (float(np.mean(p1)) - 0.5)
            p1 = 0.5 + np.clip(p1 - 0.5, -abs(_band), abs(_band))
            p1 = np.clip(p1, 0.0, 1.0)
        except Exception:
            return np.full(X.shape[0], 0.5, dtype=float)

        return p1.reshape(-1)

    # ------------------------------------------------------------------
    # LSTM part
    # ------------------------------------------------------------------
    def _build_lstm_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = int(self.meta.get("seq_len", 32))
        if X.shape[0] < seq_len:
            raise ValueError("Not enough rows to build sequences")

        seqs = []
        for i in range(seq_len, X.shape[0] + 1):
            seqs.append(X[i - seq_len: i, :])
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

            # align back to X rows
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
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        try:
            X_arr = self._to_numeric_matrix(X)
        except Exception as e:
            self._log(logging.WARNING, "[HYBRID] Failed to convert X to numeric matrix: %s. Using 0.5 uniform.", e)
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
        debug["sgd_disabled"] = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")

        # LSTM
        p_lstm, lstm_debug = self._predict_lstm_proba(X_arr)

        # Combine (şu an karar path'i LSTM odaklı; istersen p_hybrid=alpha*lstm+(1-a)*sgd yapılır)
        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            p_hybrid = p_lstm
            mode = "lstm_only"
        else:
            p_hybrid = np.full(X_arr.shape[0], 0.5, dtype=float)
            mode = "uniform_fallback"

        debug.update(
            {
                "mode": mode,
                "p_sgd_mean": float(np.mean(p_sgd)) if p_sgd is not None else 0.0,
                "p_lstm_mean": float(np.mean(p_lstm)) if p_lstm is not None else 0.0,
                "p_hybrid_mean": float(np.mean(p_hybrid)),
                "best_auc": float(self.meta.get("best_auc", 0.0)),
                "best_side": self.meta.get("best_side", "best"),
                "use_lstm_hybrid": bool(self.use_lstm_hybrid),
            }
        )
        debug.update(lstm_debug)

        self.last_debug = debug

        self._log(
            logging.INFO,
            "[HYBRID] mode=%s n_samples=%d n_features=%d p_sgd_mean=%.4f, p_lstm_mean=%.4f, p_hybrid_mean=%.4f, best_auc=%.4f, best_side=%s",
            mode,
            X_arr.shape[0],
            X_arr.shape[1],
            debug["p_sgd_mean"],
            debug["p_lstm_mean"],
            debug["p_hybrid_mean"],
            debug["best_auc"],
            debug["best_side"],
        )

        return p_hybrid, debug

    def predict_proba_single(self, X):
        """
        Online tarafta kullanılan convenience wrapper:
        sadece SON bar için class=1 olasılığını döner.
        """
        if hasattr(X, "tail") and hasattr(X, "values"):
            X_input = X
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_input = arr

        p_arr, dbg = self.predict_proba(X_input)
        self.last_debug = dbg

        if p_arr is None or len(p_arr) == 0:
            return np.array([0.5], dtype=float)

        last_p = float(p_arr[-1])
        return np.array([last_p], dtype=float)



class HybridMultiTFModel:
    """
    Multi-timeframe hibrit model sarmalayıcısı.
    """

    def __init__(self, model_dir: str, intervals: list[str], logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir
        self.intervals = intervals
        self.logger = logger or logging.getLogger("system")

        self.models: Dict[str, HybridModel] = {}
        self._init_models()

    def _log(self, level: int, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.log(level, msg, *args)
        else:
            print(msg % args if args else msg)

    def _init_models(self) -> None:
        for itv in self.intervals:
            try:
                self.models[itv] = HybridModel(model_dir=self.model_dir, interval=itv, logger=self.logger)
                self._log(logging.INFO, "[HYBRID-MTF] Loaded HybridModel for interval=%s", itv)
            except Exception as e:
                self._log(logging.WARNING, "[HYBRID-MTF] Failed to init HybridModel for %s: %s", itv, e)

    def _compute_weight_from_meta(self, meta: Dict[str, Any]) -> float:
        try:
            auc = float(meta.get("best_auc", 0.5))
        except Exception:
            auc = 0.5
        return max(auc - 0.5, 0.0)

    def predict_proba_multi(self, X_dict: Dict[str, Union[pd.DataFrame, np.ndarray, list]]) -> Tuple[float, Dict[str, Any]]:
        per_interval: Dict[str, Any] = {}
        probs_list: list[float] = []
        weights_list: list[float] = []

        for itv, model in self.models.items():
            X = X_dict.get(itv)
            if X is None:
                self._log(logging.INFO, "[HYBRID-MTF] No features provided for interval=%s, skipping.", itv)
                continue

            try:
                p_arr, dbg = model.predict_proba(X)
                if p_arr is None or len(p_arr) == 0:
                    self._log(logging.WARNING, "[HYBRID-MTF] Empty proba array for interval=%s, skipping.", itv)
                    continue

                p_last = float(p_arr[-1])
                w = self._compute_weight_from_meta(model.meta)

                if w <= 0.0:
                    if itv == "1m":
                        w = 0.30  # düşük güvenle de olsa katkı versin
                        self._log(logging.INFO, "[HYBRID-MTF] Interval=%s düşük AUC ile düşük weight=%.2f kullanılıyor.", itv, w)
                    else:
                        self._log(logging.INFO, "[HYBRID-MTF] Interval=%s weight=0 (auc<=0.5), skipping in ensemble.", itv)
                        continue

                probs_list.append(p_last)
                weights_list.append(w)

                per_interval[itv] = {
                    "p_last": p_last,
                    "weight": w,
                    "debug": dbg,
                    "best_auc_meta": float(model.meta.get("best_auc", 0.5)),
                    "best_side_meta": model.meta.get("best_side", "best"),
                }

            except Exception as e:
                self._log(logging.WARNING, "[HYBRID-MTF] predict_proba failed for interval=%s: %s", itv, e)

        if probs_list and weights_list and sum(weights_list) > 0:
            probs_arr = np.asarray(probs_list, dtype=float)
            weights_arr = np.asarray(weights_list, dtype=float)
            ensemble_p = float((probs_arr * weights_arr).sum() / weights_arr.sum())
        elif probs_list:
            ensemble_p = float(np.mean(probs_list))
        else:
            ensemble_p = 0.5

        debug_out = {
            "per_interval": per_interval,
            "ensemble": {
                "p": ensemble_p,
                "n_used": len(probs_list),
            },
        }

        self._log(logging.INFO, "[HYBRID-MTF] ensemble_p=%.4f, n_used=%d", ensemble_p, len(probs_list))
        return ensemble_p, debug_out
