# models/hybrid_inference.py

from __future__ import annotations

import os
from app_paths import MODELS_DIR

import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load

from models.sgd_helper_runtime import SGDHelperRuntime
from models.hybrid_mtf import HybridMTF  # TEK KAYNAK: weight/log/ensemble/auc standardization


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


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
    - LSTM: lstm_long_<interval>.h5, lstm_short_<interval>.h5, lstm_scaler_<interval>.joblib/.pkl
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

        # --- runtime logging control (inference spam azaltma) ---
        self.hybrid_model_log_level = str(os.getenv("HYBRID_MODEL_LOG_LEVEL", "DEBUG")).upper()
        self.disable_hybrid_model_inference_log = str(os.getenv("DISABLE_HYBRID_MODEL_INFERENCE_LOG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # --- startup logging control (load/init spam azaltma) ---
        self.hybrid_model_startup_log_level = str(os.getenv("HYBRID_MODEL_STARTUP_LOG_LEVEL", "INFO")).upper()
        self.disable_hybrid_model_startup_log = str(os.getenv("DISABLE_HYBRID_MODEL_STARTUP_LOG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

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
            "seq_len": _env_int("LSTM_SEQ_LEN_DEFAULT", 50),
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

    def _log_startup(self, level: int, msg: str, *args: Any) -> None:
        if self.disable_hybrid_model_startup_log:
            return
        self._log(level, msg, *args)

    def _log_inference(self, msg: str, *args: Any) -> None:
        if self.disable_hybrid_model_inference_log:
            return
        level = getattr(logging, self.hybrid_model_log_level, logging.DEBUG)
        self._log(level, msg, *args)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
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
            self.meta.update(meta_file)
            self.use_lstm_hybrid = bool(self.meta.get("use_lstm_hybrid", False))

            level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
            self._log_startup(level, "[HYBRID] Meta loaded from %s", path)

            # --- load SGD helper bundle (optional) ---
            try:
                _disable_sgd = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")
                if self.enable_sgd_helper and (not _disable_sgd):
                    helper_path = os.getenv("SGD_HELPER_PATH", os.path.join(self.model_dir, "sgd_helper.joblib"))
                    if os.path.exists(helper_path):
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

        scaler_joblib = os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.joblib")
        scaler_pkl = os.path.join(self.model_dir, f"lstm_scaler_{self.interval}.pkl")
        scaler_path = scaler_joblib if os.path.exists(scaler_joblib) else scaler_pkl

        # Fail-fast: scaler yoksa LSTM'i tamamen kapat
        if not os.path.exists(scaler_path):
            self._log_startup(
                logging.WARNING,
                "[HYBRID] LSTM scaler not found for interval=%s. Checked: %s , %s. Disabling LSTM.",
                self.interval,
                scaler_joblib,
                scaler_pkl,
            )
            self.lstm_long = None
            self.lstm_short = None
            self.lstm_scaler = None
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        # Fail-fast: model dosyaları yoksa LSTM kapat
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
            self.lstm_long = None
            self.lstm_short = None
            self.lstm_scaler = None
            self.use_lstm_hybrid = False
            self.meta["use_lstm_hybrid"] = False
            return

        try:
            self.lstm_long = load_model(long_path, compile=False)
            self.lstm_short = load_model(short_path, compile=False)
            self.lstm_scaler = load(scaler_path)

            # KURAL: long/short input_shape aynı olmalı. Değilse hack yok -> LSTM disable.
            li = tuple(getattr(self.lstm_long, "input_shape", ()) or ())
            si = tuple(getattr(self.lstm_short, "input_shape", ()) or ())
            if li and si and li != si:
                self._log_startup(
                    logging.WARNING,
                    "[HYBRID] LSTM input_shape mismatch (interval=%s) long=%s short=%s -> disabling LSTM (no hacks).",
                    self.interval,
                    li,
                    si,
                )
                self.lstm_long = None
                self.lstm_short = None
                self.lstm_scaler = None
                self.use_lstm_hybrid = False
                self.meta["use_lstm_hybrid"] = False
                return

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

    # ------------------------------------------------------------------
    # Feature matrix helper
    # ------------------------------------------------------------------
    @staticmethod
    def _to_numeric_matrix(X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> np.ndarray:
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

    def _get_feature_schema(self) -> Optional[list[str]]:
        sch = self.meta.get("feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        return None

    def _normalize_to_schema(self, df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
        """
        Tek kaynak sözleşme:
          - alias fix
          - eksik kolonları 0 ile doldur
          - fazla kolonları ignore et
          - schema sırasına göre seç/sırala
          - numeric coerce + inf/nan temizle
        """
        out = df.copy()

        aliases = {
            "taker_buy_base_asset_volume": "taker_buy_base_volume",
            "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
            "taker_buy_base_volume": "taker_buy_base_asset_volume",
            "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
        }
        for src, dst in aliases.items():
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]

        missing = [c for c in schema if c not in out.columns]
        if missing:
            self._log_startup(logging.WARNING, "[HYBRID] feature_schema missing cols (filled with 0): %s", missing)
            for c in missing:
                out[c] = 0.0

        # select & order (fazla kolonları ignore)
        out = out[schema].copy()

        # numeric coerce
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

    def _align_df_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Geriye uyumlu isim; davranış 'drop' değil 'normalize'.
        """
        sch = self._get_feature_schema()
        if not sch:
            return df
        return self._normalize_to_schema(df, sch)

    # ------------------------------------------------------------------
    # LSTM part
    # ------------------------------------------------------------------
    def _build_lstm_sequences(self, X: np.ndarray) -> np.ndarray:
        seq_len = int(self.meta.get("seq_len", _env_int("LSTM_SEQ_LEN_DEFAULT", 50)))
        if seq_len <= 0:
            seq_len = _env_int("LSTM_SEQ_LEN_DEFAULT", 50)

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
            # KURAL: LSTM input feature sayısı scaler ile aynı olmalı.
            # Pad/trim YOK. Uyuşmazsa LSTM disable (sözleşme bozulmuş demektir).
            need = None
            if hasattr(self.lstm_scaler, "n_features_in_"):
                try:
                    need = int(getattr(self.lstm_scaler, "n_features_in_", 0) or 0) or None
                except Exception:
                    need = None

            if need is not None and X.shape[1] != need:
                self._log_startup(
                    logging.WARNING,
                    "[HYBRID] LSTM feature mismatch (interval=%s): X=%s scaler_need=%s -> disabling LSTM (no pad/trim).",
                    self.interval,
                    X.shape,
                    need,
                )
                # disable for runtime to stop spam
                self.lstm_long = None
                self.lstm_short = None
                self.lstm_scaler = None
                self.use_lstm_hybrid = False
                self.meta["use_lstm_hybrid"] = False
                debug["lstm_used"] = False
                debug["error"] = "feature_mismatch"
                return np.full(X.shape[0], 0.5, dtype=float), debug

            X_scaled = self.lstm_scaler.transform(X)
            seqs = self._build_lstm_sequences(X_scaled)

            p_long = self.lstm_long.predict(seqs, verbose=0).reshape(-1)
            p_short = self.lstm_short.predict(seqs, verbose=0).reshape(-1)
            p_lstm = 0.5 * (p_long + p_short)

            # seq padding: ilk seq_len-1 satırı doldur
            if len(p_lstm) < X.shape[0]:
                pad_len = X.shape[0] - len(p_lstm)
                p_lstm = np.concatenate([np.full(pad_len, p_lstm[0]), p_lstm])

            debug.update(
                {
                    "lstm_used": True,
                    "p_long_mean": float(p_long.mean()) if len(p_long) else 0.0,
                    "p_short_mean": float(p_short.mean()) if len(p_short) else 0.0,
                }
            )
            return p_lstm, debug

        except Exception as e:
            debug["lstm_used"] = False
            debug["error"] = str(e)
            self._log_startup(logging.WARNING, "[HYBRID] LSTM predict failed (%s): %s", self.interval, str(e))
            return np.full(X.shape[0], 0.5, dtype=float), debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        try:
            if isinstance(X, pd.DataFrame):
                X_norm = self._align_df_to_schema(X)  # normalize
                X_arr = self._to_numeric_matrix(X_norm)
            else:
                X_arr = self._to_numeric_matrix(X)
        except Exception as e:
            self._log_startup(
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

        p_sgd = self._predict_sgd_proba(X_arr)
        debug["sgd_disabled"] = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")

        p_lstm, lstm_debug = self._predict_lstm_proba(X_arr)

        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            try:
                a = float(getattr(self, "alpha", 0.6))
            except Exception:
                a = 0.6
            a = max(0.0, min(1.0, a))
            p_hybrid = a * p_lstm + (1.0 - a) * p_sgd
            mode = "lstm+sgd"
        else:
            p_hybrid = p_sgd
            mode = "sgd_only"

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

        self._log_inference(
            "[HYBRID] mode=%s n_samples=%d n_features=%d p_sgd_mean=%.4f p_lstm_mean=%.4f p_hybrid_mean=%.4f best_auc=%.4f best_side=%s",
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

    TEK KAYNAK:
    - Ensemble ağırlıkları / logları / AUC standardizasyonu: models.hybrid_mtf.HybridMTF
    """

    def __init__(self, model_dir: Optional[str], interval: str, logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.intervals = intervals
        self.logger = logger or logging.getLogger("system")

        self.hybrid_model_startup_log_level = str(os.getenv("HYBRID_MODEL_STARTUP_LOG_LEVEL", "INFO")).upper()
        self.disable_hybrid_model_startup_log = str(os.getenv("DISABLE_HYBRID_MODEL_STARTUP_LOG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        self.models: Dict[str, HybridModel] = {}
        self._init_models()

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

    def _init_models(self) -> None:
        level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)

        for itv in self.intervals:
            try:
                self.models[itv] = HybridModel(model_dir=self.model_dir, interval=itv, logger=self.logger)
                self._log_startup(level, "[HYBRID-MTF] Loaded HybridModel for interval=%s", itv)
            except Exception as e:
                self._log_startup(logging.WARNING, "[HYBRID-MTF] Failed to init HybridModel for %s: %s", itv, e)

    def predict_proba_multi(
        self,
        X_dict: Dict[str, Union[pd.DataFrame, np.ndarray, list]],
        standardize_auc_key: str = "auc_used",
        standardize_overwrite: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Hepsi HybridMTF içinde yapılır.
        """
        X_by_interval: Dict[str, Any] = dict(X_dict)

        ensemble_p, mtf_debug = self.mtf.predict_mtf(
            X_by_interval=X_by_interval,
            standardize_auc_key=standardize_auc_key,
            standardize_overwrite=standardize_overwrite,
        )

        return ensemble_p, mtf_debug
