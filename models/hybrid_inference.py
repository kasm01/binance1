import os
from models.sgd_helper_runtime import SGDHelperRuntime
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import load
# ----------------------------------------------------------------------
# TensorFlow / LSTM opsiyonel import
# ----------------------------------------------------------------------
try:
    from tensorflow.keras.models import load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:
    # TensorFlow kütüphanesi veya .so kernel'leri yüklenemiyorsa
    # LSTM tarafını tamamen devre dışı bırakıyoruz.
    TENSORFLOW_AVAILABLE = False

    def load_model(*_args, **_kwargs):
        # HybridModel içinde çağrılırsa try/except ile yakalanacak
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
        self.enable_sgd_helper = str(os.getenv("ENABLE_SGD_HELPER","1")).lower() in ("1","true","yes","on")
        self.sgd_helper_sat_thr = float(os.getenv("SGD_HELPER_SAT_THR","0.95"))

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

        # main.py içinde kullandığımız debug için
        self.last_debug: Dict[str, Any] = {}

        # load from disk
        self._load_sgd_model()
        self._load_meta()

        # ⬇️ ENV HYBRID_MODE ile meta'yı override et
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
            # --- load SGD helper bundle (optional) ---
            try:
                _disable_sgd = str(os.getenv("DISABLE_SGD","0")).lower() in ("1","true","yes","on")
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
        # --- runtime switch: disable SGD (ENV: DISABLE_SGD=1) ---
        try:
            import os
            _disable = str(os.getenv('DISABLE_SGD','0')).lower() in ('1','true','yes','on')
        except Exception:
            _disable = False
        if _disable:
            import numpy as _np
            return _np.full(X.shape[0], 0.5, dtype=float)

        if self.sgd_model is None:
            # === SGD_SAT_PROOF_DBG (auto) ===
            try:
                import os, numpy as np, time
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
            except Exception:
                _dbg = False
            if _dbg:
                try:
                    _now = time.time()
                    _last = getattr(self, '_sgd_sat_proof_last_ts', 0) or 0
                    if (_now - float(_last)) > 60:
                        setattr(self, '_sgd_sat_proof_last_ts', _now)
                        _X = np.asarray(X, dtype=float)
                        _nan = int(np.isnan(_X).sum()) if _X.size else 0
                        _xmin = float(np.nanmin(_X)) if _X.size else 0.0
                        _xmax = float(np.nanmax(_X)) if _X.size else 0.0
                        _Xs = None
                        try:
                            sc = getattr(self, 'scaler', None)
                            _Xs = sc.transform(_X) if sc is not None else None
                        except Exception:
                            _Xs = None
                        if _Xs is not None:
                            _Xs = np.asarray(_Xs, dtype=float)
                            _snan = int(np.isnan(_Xs).sum()) if _Xs.size else 0
                            _smin = float(np.nanmin(_Xs)) if _Xs.size else 0.0
                            _smax = float(np.nanmax(_Xs)) if _Xs.size else 0.0
                        else:
                            _snan = None; _smin = None; _smax = None
            
                        mdl = getattr(self, 'model', None) or getattr(self, 'sgd_model', None)
                        dfmn = dfmx = None
                        try:
                            if mdl is not None and hasattr(mdl, 'decision_function'):
                                _df = mdl.decision_function(_Xs if _Xs is not None else _X)
                                _df = np.asarray(_df, dtype=float).reshape(-1)
                                dfmn = float(np.nanmin(_df)) if _df.size else None
                                dfmx = float(np.nanmax(_df)) if _df.size else None
                        except Exception:
                            pass
            
                        # proba değişken adını yakalamak için: local scope'da en son üretileni arar
                        _p = locals().get('proba', None)
                        pmin = pmax = None
                        try:
                            if _p is not None:
                                _p = np.asarray(_p, dtype=float)
                                pmin = float(np.nanmin(_p))
                                pmax = float(np.nanmax(_p))
                        except Exception:
                            pass
            
                        self._log(
                            20,
                            '[SGD_SAT_PROOF] X[min,max,nan]=%.6g %.6g %s | Xs[min,max,nan]=%s %s %s | df[min,max]=%s %s | proba[min,max]=%s %s',
                            _xmin, _xmax, _nan, _smin, _smax, _snan, dfmn, dfmx, pmin, pmax
                        )
                except Exception:
                    pass
            # === /SGD_SAT_PROOF_DBG ===
            return np.full(X.shape[0], 0.5, dtype=float)

        try:
            # === SGD_SAT_DBG (auto) ===
            try:
                import os
                import numpy as _np
                from datetime import datetime as _dt
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
                _now = _dt.utcnow().timestamp()
                _last = getattr(self, '_sgd_dbg_last_ts', 0) or 0
                if _dbg and (_now - float(_last)) > 60:
                    self._sgd_dbg_last_ts = _now
                    _X = _np.asarray(X, dtype=float)
                    _nan = float(_np.isnan(_X).mean()) if _X.size else 0.0
                    self._log(20, '[SGDDBG] X shape=%s nan=%.4f min=%.6g max=%.6g mean=%.6g',
                             tuple(_X.shape), _nan,
                             float(_np.nanmin(_X)) if _X.size else 0.0,
                             float(_np.nanmax(_X)) if _X.size else 0.0,
                             float(_np.nanmean(_X)) if _X.size else 0.0)
            except Exception:
                pass
            # === SGD_COLMAX_DBG (auto) ===
            try:
                import os, numpy as _np, logging
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
            except Exception:
                _dbg = False
            if _dbg:
                try:
                    _X = _np.asarray(X)
                    if _X.ndim == 2 and _X.shape[1] > 0:
                        _m = _np.nanmax(_np.abs(_X), axis=0)
                        _top = _np.argsort(_m)[-5:][::-1]
                        _pairs = []
                        for _i in _top:
                            _pairs.append('c%d=%.4g' % (int(_i), float(_m[_i])))
                        self._log(logging.INFO, '[SGDDBG] top_abs_cols=%s', ','.join(_pairs))
                except Exception:
                    pass

            # === SGD_PROOF_DBG (auto) ===
            # Proof: what SGD model returns (ENV: SGD_PROBA_DEBUG=1)
            try:
                import os, numpy as _np
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
            except Exception:
                _dbg = False
            if _dbg:
                try:
                    _t = type(getattr(self,'sgd_model',None))
                    _name = getattr(_t,'__name__', str(_t))
                    _mod  = getattr(_t,'__module__', '')
                    _df = None
                    try:
                        _df = self.sgd_model.decision_function(X)
                    except Exception:
                        _df = None
                    if _df is not None:
                        _df = _np.asarray(_df)
                        self._log(logging.INFO, '[SGDPROOF] model=%s.%s decision_function: min=%.6g max=%.6g mean=%.6g', _mod, _name, float(_df.min()), float(_df.max()), float(_df.mean()))
                    else:
                        self._log(logging.INFO, '[SGDPROOF] model=%s.%s decision_function: None', _mod, _name)
                except Exception as _e:
                    self._log(logging.INFO, '[SGDPROOF] meta_error=%r', _e)
            # === SGDPROOF_MIN (auto) ===
            try:
                import os, numpy as _np
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
            except Exception:
                _dbg = False
            if _dbg:
                try:
                    _t = type(getattr(self,'sgd_model',None))
                    self._log(logging.INFO, '[SGDPROOF] model=%s.%s Xshape=%s Xmin=%.6g Xmax=%.6g', getattr(_t,'__module__',''), getattr(_t,'__name__',''), getattr(X,'shape',None), float(_np.min(X)), float(_np.max(X)))
                except Exception as _e:
                    self._log(logging.INFO, '[SGDPROOF] meta_error=%r', _e)
            proba = self.sgd_model.predict_proba(X)
            # === SGD_PROBA_SAT_DBG (auto) ===
            try:
                import os, numpy as _np, logging
                _dbg = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
            except Exception:
                _dbg = False
            if _dbg:
                try:
                    from datetime import datetime as _dt
                    _now = _dt.utcnow().timestamp()
                    _last = getattr(self, '_sgd_dbg_last_ts', 0) or 0
                    if (_now - float(_last)) > 60:
                        setattr(self, '_sgd_dbg_last_ts', _now)
                        _X = _np.asarray(X)
                        _nan = int(_np.isnan(_X).sum()) if _X.size else 0
                        _inf = int(_np.isinf(_X).sum()) if _X.size else 0
                        _xmin = float(_np.nanmin(_X)) if _X.size else 0.0
                        _xmax = float(_np.nanmax(_X)) if _X.size else 0.0
                        _xmean = float(_np.nanmean(_X)) if _X.size else 0.0
                        # kolon std (tamamen sabit feature var mı?)
                        _std_min = None
                        _std_max = None
                        try:
                            if _X.ndim == 2 and _X.shape[0] > 1:
                                _std = _np.nanstd(_X, axis=0)
                                _std_min = float(_np.nanmin(_std))
                                _std_max = float(_np.nanmax(_std))
                        except Exception:
                            pass
                        _P = _np.asarray(proba)
                        # proba matrisi ise class1 sütununu al
                        if _P.ndim == 2 and _P.shape[1] >= 2:
                            _p1 = _P[:, 1]
                        else:
                            _p1 = _P.reshape(-1)
                        _p1 = _np.clip(_p1.astype(float), 0.0, 1.0)
                        _p_mean = float(_p1.mean()) if _p1.size else 0.0
                        _p_min = float(_p1.min()) if _p1.size else 0.0
                        _p_max = float(_p1.max()) if _p1.size else 0.0
                        _p_ones = float((_p1 >= 0.999999).mean()) if _p1.size else 0.0
                        _p_zeros = float((_p1 <= 0.000001).mean()) if _p1.size else 0.0
                        # decision_function varsa ek bilgi
                        _df_min = _df_max = None
                        try:
                            if hasattr(self.sgd_model, 'decision_function'):
                                _df = self.sgd_model.decision_function(_X)
                                _df = _np.asarray(_df).reshape(-1)
                                _df_min = float(_np.nanmin(_df))
                                _df_max = float(_np.nanmax(_df))
                        except Exception:
                            pass
                        # LOG: tek satırda kanıt
                        try:
                            self._log(
                                logging.INFO,
                                '[SGDDBG] Xshape=%s nan=%d inf=%d xmin=%.4g xmax=%.4g xmean=%.4g std_min=%s std_max=%s | p1_mean=%.6f p1_min=%.6f p1_max=%.6f ones=%.3f zeros=%.3f df_min=%s df_max=%s',
                                tuple(_X.shape), _nan, _inf, _xmin, _xmax, _xmean,
                                ('%.4g'%_std_min if _std_min is not None else 'None'),
                                ('%.4g'%_std_max if _std_max is not None else 'None'),
                                _p_mean, _p_min, _p_max, _p_ones, _p_zeros,
                                ('%.4g'%_df_min if _df_min is not None else 'None'),
                                ('%.4g'%_df_max if _df_max is not None else 'None'),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
            # === /SGD_PROBA_SAT_DBG ===
            # === SGD_PROBA_STATS_DBG (auto) ===
            try:
                import os
                import numpy as _np
                from datetime import datetime as _dt
                # After p1 extraction, log raw p1 stats (before/after helper if present)
                if _dbg:
                    try:
                        import numpy as _np
                        _p1 = _np.asarray(p1, dtype=float)
                        self._log(logging.INFO, '[SGDPROOF] p1_raw: min=%.6g max=%.6g mean=%.6g uniq_head=%s', float(_p1.min()), float(_p1.max()), float(_p1.mean()), _np.unique(_p1)[:5])
                    except Exception as _e:
                        self._log(logging.INFO, '[SGDPROOF] p1_stat_error=%r', _e)
                _dbg2 = str(os.getenv('SGD_PROBA_DEBUG','0')).lower() in ('1','true','yes','on')
                _now2 = _dt.utcnow().timestamp()
                _last2 = getattr(self, '_sgd_dbg_proba_ts', 0) or 0
                if _dbg2 and (_now2 - float(_last2)) > 60:
                    self._sgd_dbg_proba_ts = _now2
                    _P = _np.asarray(proba)
                    if _P.ndim == 2 and _P.shape[1] >= 2:
                        _p1 = _P[:, 1].astype(float)
                    else:
                        _p1 = _P.reshape(-1).astype(float)
                    _p1 = _np.nan_to_num(_p1, nan=0.5, posinf=1.0, neginf=0.0)
                    one = float((_p1 >= 0.999999).mean()) if _p1.size else 0.0
                    zero = float((_p1 <= 0.000001).mean()) if _p1.size else 0.0
                    uniq = _np.unique(_p1[:min(5000, _p1.size)])
                    self._log(20, '[SGDDBG] p1 stats: min=%.6g max=%.6g mean=%.6g one_ratio=%.3f zero_ratio=%.3f uniq_head=%s',
                             float(_np.min(_p1)) if _p1.size else 0.0,
                             float(_np.max(_p1)) if _p1.size else 0.0,
                             float(_np.mean(_p1)) if _p1.size else 0.0,
                             one, zero,
                             str(uniq[:10]))
            except Exception:
                pass
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
                # === SGD_HELPER_RUNTIME (auto) ===
                # SGD helper: center->0.5, compress around 0.5, optional disable
                try:
                    import os
                    _disable = str(os.getenv('DISABLE_SGD','0')).lower() in ('1','true','yes','on')
                    _center  = str(os.getenv('SGD_CENTER','1')).lower() in ('1','true','yes','on')
                    _band    = float(os.getenv('SGD_BAND','0.10'))  # +/- band around 0.5
                except Exception:
                    _disable, _center, _band = False, True, 0.10
                if _disable:
                    try:
                        import numpy as np
                        return np.full(X.shape[0], 0.5, dtype=float)
                    except Exception:
                        return np.array([0.5], dtype=float)
                try:
                    import numpy as np
                    p1 = np.asarray(p1, dtype=float)
                    p1 = np.clip(p1, 0.0, 1.0)
                    if _center:
                        # shift mean towards 0.5
                        p1 = p1 - (float(np.mean(p1)) - 0.5)
                    # compress around 0.5 so SGD can't dominate
                    p1 = 0.5 + np.clip(p1 - 0.5, -abs(_band), abs(_band))
                    p1 = np.clip(p1, 0.0, 1.0)
                except Exception:
                    pass
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
        # === DISABLE_SGD_GUARD (auto) ===
        try:
            import os
            _disable_sgd = str(os.getenv('DISABLE_SGD','0')).lower() in ('1','true','yes','on')
        except Exception:
            _disable_sgd = False
        if _disable_sgd:
            try:
                import numpy as np
                p_sgd = np.full(X_arr.shape[0], 0.5, dtype=float)
            except Exception:
                pass
            debug['sgd_disabled'] = True
        else:
            debug['sgd_disabled'] = False

        # LSTM
        p_lstm, lstm_debug = self._predict_lstm_proba(X_arr)

        # Combine
        # Combine (TEMP): disable SGD in decision path (keep computed for debug)
        # If LSTM available -> use LSTM; else fallback to 0.5 (not SGD)
        if self.use_lstm_hybrid and lstm_debug.get("lstm_used", False):
            p_hybrid = p_lstm
            mode = "lstm_only"
        else:
            p_hybrid = np.full(X_arr.shape[0], 0.5, dtype=float)
            mode = "uniform_fallback"

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

        # main.py logger'ı için son debug state'ini sakla
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

        return p_hybrid, debug

    def predict_proba_single(self, X):
        """
        Online tarafta kullanılan convenience wrapper.

        - X bir DataFrame, numpy array veya list olabilir.
        - Bütün seriyi hibrit pipeline'dan geçirir (SGD + LSTM),
          sadece SON bar için yukarı (class=1) olasılığını döner.
        - Çıkış: shape (1,) numpy array.
        - Ayrıca self.last_debug içine p_sgd_mean, p_lstm_mean, p_hybrid_mean vs. yazar.
        """
        import numpy as np  # lokal import

        # DataFrame ise olduğu gibi bırak (tüm history ile LSTM/SGD çalışsın)
        if hasattr(X, "tail") and hasattr(X, "values"):
            X_input = X
        else:
            # Diğer tipler için (list, ndarray vs.) güvenli 2D array'e çevir
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            X_input = arr

        # Hibrit tahmin (SGD + optional LSTM)
        p_arr, dbg = self.predict_proba(X_input)

        # Debug state'i sakla (main.py buradan okuyor)
        self.last_debug = dbg

        # Hiç çıktı yoksa güvenli fallback
        if p_arr is None or len(p_arr) == 0:
            return np.array([0.5], dtype=float)

        # Sadece SON bar için olasılık
        last_p = float(p_arr[-1])
        return np.array([last_p], dtype=float)

class HybridMultiTFModel:
    """
    Multi-timeframe hibrit model sarmalayıcısı.

    - Her interval için bir HybridModel yaratır (ör: ["1m", "5m", "15m", "1h"])
    - Her interval için son barın hibrit skorunu alır
    - Meta'daki best_auc'a göre ağırlıklandırarak tek bir ensemble skor üretir

    Kullanım:
        mtf = HybridMultiTFModel(
            model_dir="models",
            intervals=["1m", "5m", "15m", "1h"],
            logger=system_logger,
        )

        # X_dict: her interval için feature DF'leri
        ensemble_p, per_tf = mtf.predict_proba_multi({
            "1m": feat_df_1m,
            "5m": feat_df_5m,
            "15m": feat_df_15m,
            "1h": feat_df_1h,
        })
    """

    def __init__(
        self,
        model_dir: str,
        intervals: list[str],
        logger: Optional[logging.Logger] = None,
    ) -> None:
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
                self.models[itv] = HybridModel(
                    model_dir=self.model_dir,
                    interval=itv,
                    logger=self.logger,
                )
                self._log(
                    logging.INFO,
                    "[HYBRID-MTF] Loaded HybridModel for interval=%s",
                    itv,
                )
            except Exception as e:
                self._log(
                    logging.WARNING,
                    "[HYBRID-MTF] Failed to init HybridModel for %s: %s",
                    itv,
                    e,
                )

    def _compute_weight_from_meta(self, meta: Dict[str, Any]) -> float:
        """
        Meta'daki best_auc'a göre ağırlık hesaplar.
        - AUC <= 0.5 → 0 (bilgi yok sayılır)
        - AUC >  0.5 → auc - 0.5 (max 0.5 civarı olur)
        """
        try:
            auc = float(meta.get("best_auc", 0.5))
        except Exception:
            auc = 0.5
        w = max(auc - 0.5, 0.0)
        return w

    def predict_proba_multi(
        self,
        X_dict: Dict[str, Union[pd.DataFrame, np.ndarray, list]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        X_dict: {"1m": feat_df_1m, "5m": feat_df_5m, ...}

        Dönüş:
          - ensemble_p: tek bir float (0-1 arası hibrit skor)
          - debug: {
                "per_interval": {
                    "5m": {
                        "p_last": ...,
                        "weight": ...,
                        "dbg": {...},
                    },
                    ...
                },
                "ensemble": {
                    "p": ...,
                    "n_used": ...,
                }
            }
        """
        per_interval: Dict[str, Any] = {}
        probs_list = []
        weights_list = []

        for itv, model in self.models.items():
            X = X_dict.get(itv)
            if X is None:
                self._log(
                    logging.INFO,
                    "[HYBRID-MTF] No features provided for interval=%s, skipping.",
                    itv,
                )
                continue

            try:
                p_arr, dbg = model.predict_proba(X)
                if p_arr is None or len(p_arr) == 0:
                    self._log(
                        logging.WARNING,
                        "[HYBRID-MTF] Empty proba array for interval=%s, skipping.",
                        itv,
                    )
                    continue

                p_last = float(p_arr[-1])
                w = self._compute_weight_from_meta(model.meta)

                # Eğer weight 0 ise, normalde bu interval'i dışlıyoruz.
                # Ancak 1m için istisna: düşük bir weight ile yine de dahil ediyoruz.
                if w <= 0.0:
                    if itv == "1m":
                        w = 0.30  # düşük güvenle de olsa katkı versin
                        self._log(
                            logging.INFO,
                            "[HYBRID-MTF] Interval=%s düşük AUC ile düşük weight=%.2f kullanılıyor (skip edilmedi).",
                            itv,
                            w,
                        )
                    else:
                        self._log(
                            logging.INFO,
                            "[HYBRID-MTF] Interval=%s weight=0 (auc<=0.5), skipping in ensemble.",
                            itv,
                        )
                        continue
                else:
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
                self._log(
                    logging.WARNING,
                    "[HYBRID-MTF] predict_proba failed for interval=%s: %s",
                    itv,
                    e,
                )

        # Ensemble hesapla
        if probs_list and weights_list and sum(weights_list) > 0:
            probs_arr = np.asarray(probs_list, dtype=float)
            weights_arr = np.asarray(weights_list, dtype=float)
            ensemble_p = float((probs_arr * weights_arr).sum() / weights_arr.sum())
        elif probs_list:
            # ağırlıklar 0 ise basit ortalama
            ensemble_p = float(np.mean(probs_list))
        else:
            # Hiç kullanılabilir sinyal yoksa 0.5
            ensemble_p = 0.5

        debug_out = {
            "per_interval": per_interval,
            "ensemble": {
                "p": ensemble_p,
                "n_used": len(probs_list),
            },
        }

        self._log(
            logging.INFO,
            "[HYBRID-MTF] ensemble_p=%.4f, n_used=%d",
            ensemble_p,
            len(probs_list),
        )

        return ensemble_p, debug_out
