import os
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
