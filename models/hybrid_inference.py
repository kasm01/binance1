from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
from joblib import load

from app_paths import MODELS_DIR
from models.sgd_helper_runtime import SGDHelperRuntime
from models.hybrid_mtf import HybridMTF
from features.schema import normalize_to_schema
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from tensorflow.keras.models import load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

    def load_model(*_args, **_kwargs):
        raise RuntimeError("TensorFlow is not available; LSTM disabled.")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


class HybridModel:
    """
    Hybrid scorer: SGD + LSTM

    SÖZLEŞME / DRIFT KİLİDİ:
      - SGD için: meta["feature_schema"] (order + size)
      - LSTM için: meta["lstm_feature_schema"] (yoksa feature_schema fallback)
    """

    def __init__(
        self,
        model_dir: Optional[str],
        interval: str,
        logger: Optional[logging.Logger] = None,
        startup_log_level: Optional[str] = None,
        disable_startup_log: Optional[bool] = None,
        inference_log_level: Optional[str] = None,
        disable_inference_log: Optional[bool] = None,
    ) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.interval = interval
        self.logger = logger or logging.getLogger("system")

        # --- SGD helper (aux) ---
        self.sgd_helper = None
        self.enable_sgd_helper = _env_bool("ENABLE_SGD_HELPER", True)
        self.sgd_helper_sat_thr = _env_float("SGD_HELPER_SAT_THR", 0.95)

        # --- logging control ---
        env_inf_level = str(os.getenv("HYBRID_MODEL_LOG_LEVEL", "DEBUG")).upper()
        env_disable_inf = _env_bool("DISABLE_HYBRID_MODEL_INFERENCE_LOG", False)
        self.hybrid_model_log_level = str(inference_log_level or env_inf_level).upper()
        self.disable_hybrid_model_inference_log = bool(env_disable_inf if disable_inference_log is None else disable_inference_log)

        env_start_level = str(os.getenv("HYBRID_MODEL_STARTUP_LOG_LEVEL", "INFO")).upper()
        env_disable_start = _env_bool("DISABLE_HYBRID_MODEL_STARTUP_LOG", False)
        self.hybrid_model_startup_log_level = str(startup_log_level or env_start_level).upper()
        self.disable_hybrid_model_startup_log = bool(env_disable_start if disable_startup_log is None else disable_startup_log)

        # --- alpha / auto-alpha ---
        # base formula: p_hybrid = alpha * p_lstm + (1-alpha) * p_sgd
        self.alpha: float = _env_float("HYBRID_ALPHA", 0.60)
        self.alpha_auto: bool = _env_bool("HYBRID_ALPHA_AUTO", False)
        self.alpha_min: float = _env_float("HYBRID_ALPHA_MIN", 0.05)
        self.alpha_max: float = _env_float("HYBRID_ALPHA_MAX", 0.95)
        self.alpha_floor_if_bad: float = _env_float("HYBRID_ALPHA_BAD_FLOOR", 0.15)  # LSTM çok kötüyse alpha cap

        # models
        self.sgd_model = None
        self.lstm_long = None
        self.lstm_short = None
        self.lstm_scaler = None

        # meta
        self.meta: Dict[str, Any] = {
            "best_auc": 0.0,
            "best_side": "best",
            "use_lstm_hybrid": False,
            "lstm_long_auc": 0.0,
            "lstm_short_auc": 0.0,
            "seq_len": _env_int("LSTM_SEQ_LEN_DEFAULT", 50),
        }
        self.use_lstm_hybrid: bool = False
        self.last_debug: Dict[str, Any] = {}

        # spam control
        self._schema_missing_warned: set[Tuple[str, ...]] = set()
        self._lstm_mismatch_warned: bool = False

        # load
        self._load_sgd_model()
        self._load_meta()

        # ENV HYBRID_MODE override
        env_flag = os.getenv("HYBRID_MODE")
        if env_flag is not None:
            v = env_flag.strip().lower()
            if v in ("1", "true", "yes", "y", "on"):
                self.use_lstm_hybrid = True
            elif v in ("0", "false", "no", "n", "off"):
                self.use_lstm_hybrid = False

        self._load_lstm_models()

    # --------------------------
    # Logging
    # --------------------------
    def _log(self, level: int, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.log(level, msg, *args)
        else:
            print(msg % args if args else msg)

    def _log_startup(self, level: int, msg: str, *args: Any) -> None:
        if self.disable_hybrid_model_startup_log:
            return
        self._log(level, msg, *args)

    def _log_inference(self, msg: str, *args: Any) -> None:
        if self.disable_hybrid_model_inference_log:
            return
        level = getattr(logging, self.hybrid_model_log_level, logging.DEBUG)
        self._log(level, msg, *args)

    # --------------------------
    # Loaders
    # --------------------------
    def _load_sgd_model(self) -> None:
        path = os.path.join(self.model_dir, f"online_model_{self.interval}_best.joblib")
        try:
            self.sgd_model = load(path)
            level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
            self._log_startup(level, "[HYBRID] SGD model loaded from %s (interval=%s)", path, self.interval)
        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] Could not load SGD model from %s: %s", path, e)
            self.sgd_model = None

    def _load_meta(self) -> None:
        path = os.path.join(self.model_dir, f"model_meta_{self.interval}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta_file = json.load(f) or {}
            if isinstance(meta_file, dict):
                self.meta.update(meta_file)

            # lstm_feature_schema yoksa feature_schema'dan türet
            try:
                if not self.meta.get("lstm_feature_schema"):
                    fs = self.meta.get("feature_schema")
                    if isinstance(fs, list) and fs and all(isinstance(x, str) for x in fs):
                        self.meta["lstm_feature_schema"] = list(fs)
            except Exception:
                pass

            self.use_lstm_hybrid = bool(self.meta.get("use_lstm_hybrid", False))

            level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
            self._log_startup(level, "[HYBRID] Meta loaded from %s", path)

            # SGD helper (optional)
            try:
                _disable_sgd = _env_bool("DISABLE_SGD", False)
                if self.enable_sgd_helper and (not _disable_sgd):
                    helper_path = os.getenv("SGD_HELPER_PATH", os.path.join(self.model_dir, "sgd_helper.joblib"))
                    if helper_path and os.path.exists(helper_path):
                        self.sgd_helper = SGDHelperRuntime(helper_path)
                        self._log_startup(level, "[HYBRID] SGD helper loaded from %s", helper_path)
                    else:
                        self._log_startup(logging.WARNING, "[HYBRID] SGD helper not found: %s", helper_path)
            except Exception as _e:
                self._log_startup(logging.WARNING, "[HYBRID] SGD helper load failed: %s", _e)

        except FileNotFoundError:
            self._log_startup(logging.WARNING, "[HYBRID] Meta file %s not found. Using defaults.", path)
        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] Error while loading meta from %s: %s", path, e)

    def _load_lstm_models(self) -> None:
        if not self.use_lstm_hybrid:
            return

        if not TENSORFLOW_AVAILABLE:
            self._log_startup(logging.WARNING, "[HYBRID] TensorFlow not available, LSTM disabled.")
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        long_path = os.path.join(self.model_dir, f"lstm_long_{self.interval}.h5")
        short_path = os.path.join(self.model_dir, f"lstm_short_{self.interval}.h5")

        # scaler search (joblib/pkl + alias)
        scaler_candidates = [
            os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.joblib"),
            os.path.join(self.model_dir, f"lstm_{self.interval}_scaler.joblib"),
            os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.pkl"),
        ]
        scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)

        if not scaler_path:
            self._log_startup(
                logging.WARNING,
                "[HYBRID] LSTM scaler not found for interval=%s. Checked: %s. Disabling LSTM.",
                self.interval,
                scaler_candidates,
            )
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        if not (os.path.exists(long_path) and os.path.exists(short_path)):
            self._log_startup(
                logging.WARNING,
                "[HYBRID] LSTM model file missing (interval=%s). long=%s exists=%s | short=%s exists=%s. Disabling LSTM.",
                self.interval,
                long_path,
                os.path.exists(long_path),
                short_path,
                os.path.exists(short_path),
            )
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        try:
            self.lstm_long = load_model(long_path, compile=False)
            self.lstm_short = load_model(short_path, compile=False)
            self.lstm_scaler = load(scaler_path)

            # input_shape mismatch -> log only
            try:
                li = tuple(getattr(self.lstm_long, "input_shape", ()) or ())
                si = tuple(getattr(self.lstm_short, "input_shape", ()) or ())
                if li and si and li != si:
                    self._log_startup(
                        logging.WARNING,
                        "[HYBRID] LSTM input_shape mismatch long=%s short=%s (interval=%s).",
                        li,
                        si,
                        self.interval,
                    )
            except Exception:
                pass

            level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
            self._log_startup(level, "[HYBRID] LSTM models and scaler loaded for %s (use_lstm_hybrid=True)", self.interval)

        except Exception as e:
            self._log_startup(
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

    # --------------------------
    # Schema helpers (DRIFT KILL)
    # --------------------------
    def _get_feature_schema(self) -> Optional[List[str]]:
        sch = self.meta.get("feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        return None

    def _get_lstm_feature_schema(self) -> Optional[List[str]]:
        sch = self.meta.get("lstm_feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        sch2 = self.meta.get("feature_schema")
        if isinstance(sch2, list) and sch2 and all(isinstance(x, str) for x in sch2):
            return sch2
        return None

    def _normalize_to_schema(self, df: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
        """
        Tek sözleşme (tek kaynak): features.schema.normalize_to_schema

        Burada sadece "warn-once" davranışını koruyoruz.
        """

        def _log_missing_once(missing: List[str]) -> None:
            # schema başına bir kez uyarı
            try:
                key = tuple(schema)
                if key in self._schema_missing_warned:
                    return
                self._schema_missing_warned.add(key)
            except Exception:
                # worst-case: log spam yerine loglamayı da geçebilir
                pass

            try:
                self._log_startup(logging.WARNING, "[HYBRID] schema missing cols (filled=0): %s", missing)
            except Exception:
                pass

        # normalize_to_schema: alias + missing=0 + order + numeric cleanup
        return normalize_to_schema(df, schema, log_missing=_log_missing_once)

    @staticmethod
    def _df_to_matrix_strict(df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame -> float matrix.
        (ÖNEMLİ) select_dtypes ile kolon düşürme YOK; schema zaten numeric olmalı.
        """
        arr = df.to_numpy(dtype=float, copy=False)
        if arr.ndim != 2:
            arr = np.asarray(arr, dtype=float).reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _to_numeric_matrix(X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return HybridModel._df_to_matrix_strict(X)

        if isinstance(X, pd.Series):
            return pd.to_numeric(X, errors="coerce").to_numpy(dtype=float).reshape(-1, 1)

        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.astype(float).reshape(-1, 1)
            return X.astype(float)

        raise TypeError(f"Unsupported X type: {type(X)}")

    def _align_df_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        sch = self._get_feature_schema()
        if not sch:
            return df
        return self._normalize_to_schema(df, sch)

    # --------------------------
    # SGD
    # --------------------------
    def _predict_sgd_proba(self, X: np.ndarray) -> np.ndarray:
        _disable = _env_bool("DISABLE_SGD", False)
        if _disable or self.sgd_model is None or X is None or X.size == 0:
            return np.full((0 if X is None else X.shape[0],), 0.5, dtype=float)

        # expected features: prefer schema length, then meta.n_features
        expected = None
        sch = self._get_feature_schema()
        if sch:
            expected = len(sch)
        else:
            try:
                expected = int(self.meta.get("n_features") or 0) or None
            except Exception:
                expected = None

        if expected is not None and X.shape[1] != expected:
            self._log_startup(
                logging.WARNING,
                "[HYBRID] SGD feature mismatch: X=%s expected=%s -> fallback 0.5",
                X.shape,
                expected,
            )
            return np.full(X.shape[0], 0.5, dtype=float)

        try:
            proba = np.asarray(self.sgd_model.predict_proba(X), dtype=float)

            if proba.ndim == 2 and proba.shape[1] >= 2:
                classes_ = getattr(self.sgd_model, "classes_", None)
                try:
                    if hasattr(self.sgd_model, "named_steps") and "sgd" in getattr(self.sgd_model, "named_steps", {}):
                        classes_ = getattr(self.sgd_model.named_steps["sgd"], "classes_", classes_)
                except Exception:
                    pass

                idx_pos = 1
                try:
                    if classes_ is not None:
                        cl = list(classes_)
                        idx_pos = cl.index(1) if 1 in cl else (len(cl) - 1)
                except Exception:
                    idx_pos = 1

                p1 = proba[:, idx_pos]
            else:
                p1 = np.asarray(proba, dtype=float).reshape(-1)

        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] SGD predict_proba failed: %r", e)
            return np.full(X.shape[0], 0.5, dtype=float)

        # optional proba transform
        _flip = _env_bool("SGD_FLIP_PROBA", False)
        _eps = _env_float("SGD_PROBA_EPS", 0.0)

        if _flip:
            p1 = 1.0 - p1
        if _eps > 0.0:
            _eps = max(0.0, min(0.49, float(_eps)))
            p1 = _eps + (1.0 - 2.0 * _eps) * p1

        return np.clip(p1, 0.0, 1.0).reshape(-1)

    # --------------------------
    # LSTM
    # --------------------------
    def _build_lstm_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = int(self.meta.get("seq_len", _env_int("LSTM_SEQ_LEN_DEFAULT", 50)))
        if seq_len <= 0:
            seq_len = _env_int("LSTM_SEQ_LEN_DEFAULT", 50)

        if X.shape[0] < seq_len:
            raise ValueError("Not enough rows to build sequences")

        seqs = [X[i - seq_len : i, :] for i in range(seq_len, X.shape[0] + 1)]
        return np.asarray(seqs)

    def _predict_lstm_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        n = int(X.shape[0]) if hasattr(X, "shape") else 0
        if n <= 0:
            return np.array([], dtype=float), {"lstm_used": False, "error": "empty_X"}

        if not (
            self.use_lstm_hybrid
            and self.lstm_long is not None
            and self.lstm_short is not None
            and self.lstm_scaler is not None
        ):
            debug["lstm_used"] = False
            return np.full(n, 0.5, dtype=float), debug

        try:
            # LSTM schema lock (order + size)
            if isinstance(X, pd.DataFrame):
                lstm_schema = self._get_lstm_feature_schema()
                if lstm_schema:
                    Xdf = self._normalize_to_schema(X, lstm_schema)
                else:
                    Xdf = X
                X_num = self._df_to_matrix_strict(Xdf)
            else:
                X_num = self._to_numeric_matrix(X)

            need = getattr(self.lstm_scaler, "n_features_in_", None)
            try:
                need = int(need) if need is not None else None
            except Exception:
                need = None

            if need is not None and need > 0 and X_num.shape[1] != need:
                # mismatch: fallback only (do NOT disable runtime permanently)
                if not self._lstm_mismatch_warned:
                    self._lstm_mismatch_warned = True
                    self._log_startup(
                        logging.WARNING,
                        "[HYBRID] LSTM scaler feature mismatch (interval=%s): X=%s need=%s -> fallback 0.5 (no disable)",
                        self.interval,
                        X_num.shape,
                        need,
                    )
                debug.update({
                    "lstm_used": False,
                    "error": "lstm_feature_mismatch",
                    "lstm_need_features": int(need),
                    "lstm_got_features": int(X_num.shape[1]),
                })
                return np.full(n, 0.5, dtype=float), debug

            X_scaled = self.lstm_scaler.transform(X_num)
            seqs = self._build_lstm_sequences(X_scaled)

            p_long = self.lstm_long.predict(seqs, verbose=0).reshape(-1)
            p_short = self.lstm_short.predict(seqs, verbose=0).reshape(-1)
            p_lstm = 0.5 * (p_long + p_short)

            # seq padding
            if len(p_lstm) < n and len(p_lstm) > 0:
                pad_len = n - len(p_lstm)
                p_lstm = np.concatenate([np.full(pad_len, float(p_lstm[0])), p_lstm])
            elif len(p_lstm) == 0:
                p_lstm = np.full(n, 0.5, dtype=float)

            debug.update(
                {
                    "lstm_used": True,
                    "p_long_mean": float(p_long.mean()) if len(p_long) else 0.0,
                    "p_short_mean": float(p_short.mean()) if len(p_short) else 0.0,
                    "lstm_need_features": int(need) if need is not None else None,
                    "lstm_got_features": int(X_num.shape[1]),
                }
            )
            return p_lstm.astype(float), debug

        except Exception as e:
            debug["lstm_used"] = False
            debug["error"] = str(e)
            self._log_startup(logging.WARNING, "[HYBRID] LSTM predict failed (%s): %s", self.interval, str(e))
            return np.full(n, 0.5, dtype=float), debug
 # --------------------------
    # Alpha auto
    # --------------------------
    def _get_sgd_auc(self) -> float:
        try:
            return float(self.meta.get("best_auc", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _get_lstm_auc(self) -> float:
        # prefer explicit lstm_auc, otherwise max(long/short)
        try:
            v = self.meta.get("lstm_auc", None)
            if v is not None:
                return float(v)
        except Exception:
            pass

        vals = []
        for k in ("lstm_long_auc", "lstm_short_auc"):
            try:
                vals.append(float(self.meta.get(k, 0.0) or 0.0))
            except Exception:
                continue
        return float(max(vals) if vals else 0.0)

    def _compute_alpha_used(self) -> Tuple[float, Dict[str, Any]]:
        dbg: Dict[str, Any] = {}
        a_base = float(self.alpha)
        a_base = max(0.0, min(1.0, a_base))

        if not self.alpha_auto:
            a = float(np.clip(a_base, self.alpha_min, self.alpha_max))
            dbg.update({"alpha_used": a, "alpha_mode": "fixed", "alpha_base": a_base})
            return a, dbg

        sgd_auc = self._get_sgd_auc()
        lstm_auc = self._get_lstm_auc()

        dbg.update({"alpha_base": a_base, "sgd_auc": sgd_auc, "lstm_auc": lstm_auc})

        # if no meaningful auc -> keep base
        if sgd_auc <= 0.0 or lstm_auc <= 0.0:
            a = float(np.clip(a_base, self.alpha_min, self.alpha_max))
            dbg.update({"alpha_used": a, "alpha_mode": "auto_no_auc"})
            return a, dbg

        # compare improvements above 0.5 (AUC baseline)
        s = max(0.0, sgd_auc - 0.5)
        l = max(0.0, lstm_auc - 0.5)

        # if SGD has no edge, keep base
        if s <= 1e-9:
            a = float(np.clip(a_base, self.alpha_min, self.alpha_max))
            dbg.update({"alpha_used": a, "alpha_mode": "auto_sgd_flat"})
            return a, dbg

        ratio = float(np.clip(l / s, 0.0, 1.0))  # LSTM relative quality
        a = a_base * ratio

        # extra cap: if lstm_auc < sgd_auc, keep it below a_floor_if_bad (optional)
        if lstm_auc < sgd_auc:
            a = min(a, self.alpha_floor_if_bad)

        a = float(np.clip(a, self.alpha_min, self.alpha_max))
        dbg.update({"alpha_used": a, "alpha_mode": "auto_ratio", "alpha_ratio": ratio})
        return a, dbg

    # --------------------------
    # Public API
    # --------------------------
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        if X is None:
            debug.update({"mode": "uniform_fallback", "reason": "X_is_None", "p_hybrid_mean": 0.5})
            return np.full(1, 0.5, dtype=float), debug

        if isinstance(X, pd.DataFrame) and X.shape[0] == 0:
            debug.update({"mode": "uniform_fallback", "reason": "empty_dataframe", "p_hybrid_mean": 0.5})
            return np.full(1, 0.5, dtype=float), debug

        # DataFrame ise schema lock (SGD tarafı)
        try:
            if isinstance(X, pd.DataFrame):
                X_norm = self._align_df_to_schema(X)
                X_arr = self._df_to_matrix_strict(X_norm)
            else:
                X_arr = self._to_numeric_matrix(X)
        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] Failed to convert X to numeric: %s -> uniform 0.5", e)
            n = 1
            try:
                n = int(getattr(X, "shape")[0]) if hasattr(X, "shape") else int(len(X))  # type: ignore[arg-type]
            except Exception:
                n = 1
            n = 1 if n <= 0 else n

            p_uniform = np.full(n, 0.5, dtype=float)
            debug.update(
                {
                    "mode": "uniform_fallback",
                    "error": str(e),
                    "p_sgd_mean": 0.5,
                    "p_lstm_mean": 0.5,
                    "p_hybrid_mean": 0.5,
                    "best_auc": float(self.meta.get("best_auc", 0.0)),
                    "best_side": self.meta.get("best_side", "best"),
                    "use_lstm_hybrid": bool(self.use_lstm_hybrid),
                }
            )
            return p_uniform, debug

        p_sgd = self._predict_sgd_proba(X_arr)
        p_lstm, lstm_debug = self._predict_lstm_proba(X if isinstance(X, pd.DataFrame) else X_arr)

        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            a_used, a_dbg = self._compute_alpha_used()
            p_hybrid = a_used * p_lstm + (1.0 - a_used) * p_sgd
            mode = "lstm+sgd"
            debug.update(a_dbg)
        else:
            p_hybrid = p_sgd
            mode = "sgd_only"
            debug.update({"alpha_used": 0.0, "alpha_mode": "sgd_only"})

        debug.update(
            {
                "mode": mode,
                "p_sgd_mean": float(np.mean(p_sgd)) if p_sgd is not None else 0.5,
                "p_lstm_mean": float(np.mean(p_lstm)) if p_lstm is not None else 0.5,
                "p_hybrid_mean": float(np.mean(p_hybrid)) if p_hybrid is not None else 0.5,
                "best_auc": float(self.meta.get("best_auc", 0.0)),
                "best_side": self.meta.get("best_side", "best"),
                "use_lstm_hybrid": bool(self.use_lstm_hybrid),
            }
        )
        debug.update(lstm_debug)

        self.last_debug = debug

        self._log_inference(
            "[HYBRID] mode=%s itv=%s n_samples=%d n_features=%d alpha=%.4f(%s) p_sgd_mean=%.4f p_lstm_mean=%.4f p_hybrid_mean=%.4f best_auc=%.4f",
            mode,
            self.interval,
            int(X_arr.shape[0]),
            int(X_arr.shape[1]),
            float(debug.get("alpha_used", 0.0)),
            str(debug.get("alpha_mode", "na")),
            float(debug["p_sgd_mean"]),
            float(debug["p_lstm_mean"]),
            float(debug["p_hybrid_mean"]),
            float(debug["best_auc"]),
        )

        return p_hybrid, debug


class HybridMultiTFModel:
    """
    Multi-timeframe hibrit model sarmalayıcısı.
    Ensemble ağırlıkları / AUC standardizasyonu: models.hybrid_mtf.HybridMTF
    """

    def __init__(
        self,
        model_dir: Optional[str],
        intervals: List[str],
        logger: Optional[logging.Logger] = None,
        models_by_interval: Optional[Dict[str, HybridModel]] = None,
    ) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.intervals = list(intervals)
        self.logger = logger or logging.getLogger("system")

        self.hybrid_model_startup_log_level = str(os.getenv("HYBRID_MODEL_STARTUP_LOG_LEVEL", "INFO")).upper()
        self.disable_hybrid_model_startup_log = _env_bool("DISABLE_HYBRID_MODEL_STARTUP_LOG", False)

        if models_by_interval:
            self.models: Dict[str, HybridModel] = {itv: models_by_interval[itv] for itv in self.intervals if itv in models_by_interval}
            missing = [itv for itv in self.intervals if itv not in self.models]
            if missing:
                level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
                self._log_startup(level, "[HYBRID-MTF] models_by_interval provided, missing=%s -> loading missing", missing)
                for itv in missing:
                    self.models[itv] = HybridModel(
                        interval=itv,
                        model_dir=self.model_dir,
                        logger=self.logger,
                        startup_log_level=self.hybrid_model_startup_log_level,
                        disable_startup_log=self.disable_hybrid_model_startup_log,
                    )
        else:
            self.models = {}
            for itv in self.intervals:
                self.models[itv] = HybridModel(
                    interval=itv,
                    model_dir=self.model_dir,
                    logger=self.logger,
                    startup_log_level=self.hybrid_model_startup_log_level,
                    disable_startup_log=self.disable_hybrid_model_startup_log,
                )

        # HybridMTF: auc_history disk persistence + calibration
        self.mtf = HybridMTF(
            models_by_interval=self.models,
            logger=self.logger,
            auc_key_priority=("auc_used", "wf_auc_mean", "val_auc", "best_auc", "auc", "oof_auc"),
        )

    def _log(self, level: int, msg: str, *args: Any) -> None:
        if self.logger:
            self.logger.log(level, msg, *args)
        else:
            print(msg % args if args else msg)

    def _log_startup(self, level: int, msg: str, *args: Any) -> None:
        if self.disable_hybrid_model_startup_log:
            return
        self._log(level, msg, *args)

    def predict_proba_multi(
        self,
        X_dict: Dict[str, Union[pd.DataFrame, np.ndarray, list]],
        standardize_auc_key: str = "auc_used",
        standardize_overwrite: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        ensemble_p, mtf_debug = self.mtf.predict_mtf(
            X_by_interval=dict(X_dict),
            standardize_auc_key=standardize_auc_key,
            standardize_overwrite=standardize_overwrite,
        )
        return float(ensemble_p), mtf_debug

