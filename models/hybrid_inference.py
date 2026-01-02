# models/hybrid_inference.py
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load

from app_paths import MODELS_DIR
from models.sgd_helper_runtime import SGDHelperRuntime
from models.hybrid_mtf import HybridMTF

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


class HybridModel:
    """
    Hybrid scorer: combines SGD (online model) and LSTM models.

    - SGD: online_model_<interval>_best.joblib
    - LSTM: lstm_long_<interval>.h5, lstm_short_<interval>.h5, lstm_scaler_<interval>.joblib/.pkl
    - Meta: model_meta_<interval>.json

    Notlar:
    - feature_schema: SGD / genel DF normalize için
    - lstm_feature_schema: LSTM için özel schema (order+size garanti)
    """

    def __init__(self, model_dir: Optional[str], interval: str, logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.interval = interval
        self.logger = logger or logging.getLogger("system")

        # --- SGD helper (aux) ---
        self.sgd_helper = None
        self.enable_sgd_helper = str(os.getenv("ENABLE_SGD_HELPER", "1")).lower() in ("1", "true", "yes", "on")
        self.sgd_helper_sat_thr = float(os.getenv("SGD_HELPER_SAT_THR", "0.95"))

        # --- runtime logging control ---
        self.hybrid_model_log_level = str(os.getenv("HYBRID_MODEL_LOG_LEVEL", "DEBUG")).upper()
        self.disable_hybrid_model_inference_log = str(os.getenv("DISABLE_HYBRID_MODEL_INFERENCE_LOG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # --- startup logging control ---
        self.hybrid_model_startup_log_level = str(os.getenv("HYBRID_MODEL_STARTUP_LOG_LEVEL", "INFO")).upper()
        self.disable_hybrid_model_startup_log = str(os.getenv("DISABLE_HYBRID_MODEL_STARTUP_LOG", "0")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # hybrid weight: p_hybrid = alpha * p_lstm + (1 - alpha) * p_sgd
        self.alpha: float = float(os.getenv("HYBRID_ALPHA", "0.60"))

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
            self.meta.update(meta_file)
            self.use_lstm_hybrid = bool(self.meta.get("use_lstm_hybrid", False))

            level = getattr(logging, self.hybrid_model_startup_log_level, logging.INFO)
            self._log_startup(level, "[HYBRID] Meta loaded from %s", path)

            # --- load SGD helper (optional) ---
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

        # scaler yoksa LSTM disable
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

        # modeller yoksa LSTM disable
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

            # input_shape mismatch -> sadece logla (hack yok)
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
    # Feature helpers
    # --------------------------
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

    def _get_lstm_feature_schema(self) -> Optional[list[str]]:
        sch = self.meta.get("lstm_feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        return None

    def _normalize_to_schema(self, df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
        """
        Tek sözleşme:
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
            self._log_startup(logging.WARNING, "[HYBRID] schema missing cols (filled=0): %s", missing)
            for c in missing:
                out[c] = 0.0

        out = out[schema].copy()

        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

    def _align_df_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        sch = self._get_feature_schema()
        if not sch:
            return df
        return self._normalize_to_schema(df, sch)

    # --------------------------
    # SGD
    # --------------------------
    def _predict_sgd_proba(self, X: np.ndarray) -> np.ndarray:
        _disable = str(os.getenv("DISABLE_SGD", "0")).lower() in ("1", "true", "yes", "on")
        if _disable or self.sgd_model is None:
            return np.full(X.shape[0], 0.5, dtype=float)

        expected = None
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
            proba = self.sgd_model.predict_proba(X)
        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] SGD predict_proba failed: %r", e)
            return np.full(X.shape[0], 0.5, dtype=float)

        try:
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = proba[:, 1]
            else:
                p1 = np.asarray(proba, dtype=float).reshape(-1)
        except Exception:
            return np.full(X.shape[0], 0.5, dtype=float)

        # stabilizer (center/band)
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

        n = X.shape[0] if hasattr(X, "shape") else 0
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
            # --- LSTM FEATURE SCHEMA KİLİDİ (ORDER + SIZE GUARANTEED) ---
            lstm_schema = self._get_lstm_feature_schema()
            if isinstance(X, pd.DataFrame) and isinstance(lstm_schema, list) and lstm_schema:
                Xdf = self._normalize_to_schema(X, lstm_schema)
                X_num = self._to_numeric_matrix(Xdf)
            else:
                X_num = self._to_numeric_matrix(X)

            # scaler mismatch trim/pad (nadir ama kalsın)
            try:
                need = int(getattr(self.lstm_scaler, "n_features_in_", 0) or 0)
                if need > 0 and X_num.shape[1] != need:
                    self._log_startup(
                        logging.WARNING,
                        "[HYBRID] LSTM scaler feature mismatch (interval=%s): X=%s need=%s -> trim/pad fallback",
                        self.interval,
                        X_num.shape,
                        need,
                    )
                    if X_num.shape[1] > need:
                        X_num = X_num[:, :need]
                    else:
                        X_num = np.pad(X_num, ((0, 0), (0, need - X_num.shape[1])), constant_values=0.0)
            except Exception:
                pass

            X_scaled = self.lstm_scaler.transform(X_num)
            seqs = self._build_lstm_sequences(X_scaled)

            p_long = self.lstm_long.predict(seqs, verbose=0).reshape(-1)
            p_short = self.lstm_short.predict(seqs, verbose=0).reshape(-1)
            p_lstm = 0.5 * (p_long + p_short)

            # seq padding: ilk seq_len-1 satırı doldur
            if len(p_lstm) < n:
                pad_len = n - len(p_lstm)
                p_lstm = np.concatenate([np.full(pad_len, float(p_lstm[0])), p_lstm])

            debug.update(
                {
                    "lstm_used": True,
                    "p_long_mean": float(p_long.mean()) if len(p_long) else 0.0,
                    "p_short_mean": float(p_short.mean()) if len(p_short) else 0.0,
                }
            )
            return p_lstm.astype(float), debug

        except Exception as e:
            debug["lstm_used"] = False
            debug["error"] = str(e)
            self._log_startup(logging.WARNING, "[HYBRID] LSTM predict failed (%s): %s", self.interval, str(e))
            return np.full(n, 0.5, dtype=float), debug


    # --------------------------
    # Public API
    # --------------------------
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame, pd.Series, list]) -> Tuple[np.ndarray, Dict[str, Any]]:
        debug: Dict[str, Any] = {}

        # X -> numeric
        try:
            if isinstance(X, pd.DataFrame):
                X_norm = self._align_df_to_schema(X)
                X_arr = self._to_numeric_matrix(X_norm)
            else:
                X_arr = self._to_numeric_matrix(X)
        except Exception as e:
            self._log_startup(logging.WARNING, "[HYBRID] Failed to convert X to numeric: %s -> uniform 0.5", e)
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
        p_lstm, lstm_debug = self._predict_lstm_proba(X if isinstance(X, pd.DataFrame) else X_arr)

        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            a = float(getattr(self, "alpha", 0.6))
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
                "p_hybrid_mean": float(np.mean(p_hybrid)) if p_hybrid is not None else 0.5,
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


class HybridMultiTFModel:
    """
    Multi-timeframe hibrit model sarmalayıcısı.

    TEK KAYNAK:
    - Ensemble ağırlıkları / logları / AUC standardizasyonu: models.hybrid_mtf.HybridMTF
    """

    def __init__(self, model_dir: Optional[str], intervals: list[str], logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.intervals = list(intervals)
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
        ensemble_p, mtf_debug = self.mtf.predict_mtf(
            X_by_interval=dict(X_dict),
            standardize_auc_key=standardize_auc_key,
            standardize_overwrite=standardize_overwrite,
        )
        return float(ensemble_p), mtf_debug
