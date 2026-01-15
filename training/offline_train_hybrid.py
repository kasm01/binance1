#!/usr/bin/env python
"""
Offline SGD trainer (hybrid pipeline)
- Feature contract is enforced via features.pipeline.make_matrix + FEATURE_SCHEMA_22
- Time columns are excluded from SGD input to avoid saturation / scale issues
- Meta includes lstm_feature_schema so LSTM training/inference can stay aligned
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import joblib

# ------------------------------------------------------------
# MODELS_DIR import (robust)
# ------------------------------------------------------------
try:
    from app_paths import MODELS_DIR  # project constant
except Exception:
    MODELS_DIR = os.getenv("MODELS_DIR", "models")

from features.pipeline import make_matrix, FEATURE_SCHEMA_22

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger("offline_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------
# ENV helpers
# ------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v != "" else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _env_csv_list(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return list(default)
    parts = [p.strip() for p in str(v).split(",")]
    parts = [p for p in parts if p]
    return parts if parts else list(default)


# ------------------------------------------------------------
# Config (ENV-driven)
# ------------------------------------------------------------
SYMBOL = _env_str("SYMBOL", "BTCUSDT")
DATA_DIR = _env_str("DATA_DIR", "data/offline_cache")

# interval list (requested)
INTERVALS: List[str] = _env_csv_list("INTERVALS", ["1m", "3m", "5m", "15m", "30m", "1h"])

# cap training bars
MAX_BARS = _env_int("OFFLINE_MAX_BARS", 50000)

# Label config
HORIZON = _env_int("LABEL_HORIZON", 3)      # sen genelde 3 kullanıyorsun
LABEL_THR = _env_float("LABEL_THR", 0.0005) # sen genelde 0.0005 kullanıyorsun

# SGD hyperparams
SGD_ALPHA = _env_float("SGD_ALPHA", 1e-4)
SGD_MAX_ITER = _env_int("SGD_MAX_ITER", 50)
SGD_TOL = _env_float("SGD_TOL", 1e-3)
SGD_LOSS = _env_str("SGD_LOSS", "log_loss")
SGD_PENALTY = _env_str("SGD_PENALTY", "l2")
SGD_SHUFFLE = _env_bool("SGD_SHUFFLE", False)
SGD_NJOBS = _env_int("SGD_NJOBS", -1)
SGD_RANDOM_STATE = _env_int("SGD_RANDOM_STATE", 42)

# LSTM meta defaults
LSTM_SEQ_LEN_DEFAULT = _env_int("LSTM_SEQ_LEN_DEFAULT", 50)
LSTM_INTERVALS: List[str] = _env_csv_list("LSTM_INTERVALS", ["1m", "3m", "5m", "15m", "30m", "1h"])

# ------------------------------------------------------------
# Feature schema (contract)
# ------------------------------------------------------------
# IMPORTANT: Time columns excluded from SGD input (saturation fix)
SGD_SCHEMA_USED: List[str] = [c for c in FEATURE_SCHEMA_22 if c not in ("open_time", "close_time")]

# Optional: if you *really* want time features later, add env flag and implement proper normalization.
# For now we keep them excluded to match your current stable setup.
USE_TIME_FEATURES = _env_bool("SGD_USE_TIME_FEATURES", False)
if USE_TIME_FEATURES:
    SGD_SCHEMA_USED = list(FEATURE_SCHEMA_22)

# ------------------------------------------------------------
# Label helpers
# ------------------------------------------------------------
def make_labels_from_close(close: np.ndarray, horizon: int, thr: float) -> np.ndarray:
    """
    y[t] = 1 if (close[t+h] / close[t] - 1) > thr else 0
    last horizon rows are forced to 0 (no future)
    """
    close = np.asarray(close, dtype=float)
    close = np.where(np.isfinite(close), close, np.nan)

    fut = np.roll(close, -horizon)
    ret = (fut / np.maximum(close, 1e-12)) - 1.0
    y = (ret > float(thr)).astype(int)

    # last horizon have no future
    if horizon > 0:
        y[-horizon:] = 0

    # NaNs -> 0
    y = np.where(np.isfinite(ret), y, 0).astype(int)
    return y


def align_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean X (finite rows) and align y to same length.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    # ensure same length (horizon, make_matrix differences)
    n = min(len(y), X.shape[0])
    X = X[:n]
    y = y[:n]

    # remove any rows with NaN/Inf
    X = np.where(np.isfinite(X), X, np.nan)
    row_ok = ~np.isnan(X).any(axis=1)
    X = X[row_ok]
    y = y[row_ok]

    return X, y


# ------------------------------------------------------------
# Data loader
# ------------------------------------------------------------
def load_offline_klines(symbol: str, interval: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}_6m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Offline cache yok: {path}")

    df = pd.read_csv(path)
    if MAX_BARS > 0 and len(df) > MAX_BARS:
        df = df.tail(MAX_BARS).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info("[%s] Loaded offline klines: shape=%s | path=%s", interval, df.shape, path)
    return df


# ------------------------------------------------------------
# Training logic per interval
# ------------------------------------------------------------
def train_interval(interval: str) -> None:
    logger.info("========== TRAIN interval=%s ==========", interval)

    raw_df = load_offline_klines(SYMBOL, interval)

    # --- build X from the SINGLE SOURCE OF TRUTH ---
    schema_used = list(SGD_SCHEMA_USED)
    X = make_matrix(raw_df, schema=schema_used)
    X = np.asarray(X, dtype=float)

    # --- build y from close (raw_df close) ---
    if "close" not in raw_df.columns:
        raise RuntimeError(f"[{interval}] raw_df has no 'close' column")

    close = pd.to_numeric(raw_df["close"], errors="coerce").astype(float).values
    y = make_labels_from_close(close, horizon=HORIZON, thr=LABEL_THR)

    # --- align + clean ---
    X, y = align_xy(X, y)

    if X.shape[0] < 1000:
        logger.warning("[%s] Too few clean samples: n=%d, skip.", interval, int(X.shape[0]))
        return

    # label distribution sanity
    u, cnt = np.unique(y, return_counts=True)
    ydist = dict(zip(u.tolist(), cnt.tolist()))
    if len(ydist) < 2:
        logger.warning("[%s] Single-class labels: %s (thr/horizon?), skip.", interval, ydist)
        return

    logger.info(
        "[%s] X shape=%s | n_features=%d | ydist=%s",
        interval, X.shape, X.shape[1], ydist
    )

    # split
    n = X.shape[0]
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # optional class imbalance weighting (stable SGD)
    # gives roughly balanced contribution
    cnt0 = max(int(ydist.get(0, 1)), 1)
    cnt1 = max(int(ydist.get(1, 1)), 1)
    w0 = 0.5 / cnt0
    w1 = 0.5 / cnt1
    sample_weight = np.where(y_train == 1, w1, w0).astype(float)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("sgd", SGDClassifier(
                loss=SGD_LOSS,
                penalty=SGD_PENALTY,
                alpha=SGD_ALPHA,
                max_iter=SGD_MAX_ITER,
                tol=SGD_TOL,
                shuffle=SGD_SHUFFLE,
                random_state=SGD_RANDOM_STATE,
                n_jobs=SGD_NJOBS,
            )),
        ]
    )

    logger.info(
        "[%s] Training SGD... alpha=%g max_iter=%d tol=%g loss=%s penalty=%s",
        interval, SGD_ALPHA, SGD_MAX_ITER, SGD_TOL, SGD_LOSS, SGD_PENALTY
    )
    clf.fit(X_train, y_train, sgd__sample_weight=sample_weight)

    # metrics
    try:
        proba = clf.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc = float(roc_auc_score(y_val, proba))
        else:
            auc = 0.5
            logger.warning("[%s] y_val single-class, AUC=0.5", interval)
    except Exception as e:
        logger.warning("[%s] AUC compute error: %r", interval, e)
        auc = 0.5

    logger.info("[%s] Validation AUC=%.4f | y_mean=%.4f", interval, auc, float(np.mean(y)))

    # save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"online_model_{interval}_best.joblib")
    joblib.dump(clf, model_path)

    best_side = "long" if auc >= 0.5 else "short"

    meta_path = os.path.join(MODELS_DIR, f"model_meta_{interval}.json")

    meta: Dict[str, Any] = {
        "interval": interval,
        "symbol": SYMBOL,

        # IMPORTANT: schema contract used in training
        "feature_schema": list(schema_used),

        # IMPORTANT: keep LSTM aligned (LSTM train should use this)
        "lstm_feature_schema": list(schema_used),

        "feature_source": "offline_train_hybrid::features.pipeline.make_matrix",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "best_auc": float(auc),
        "best_side": best_side,

        # hybrid flags
        "use_lstm_hybrid": True if interval in set(LSTM_INTERVALS) else False,
        "seq_len": int(LSTM_SEQ_LEN_DEFAULT),

        # versioning
        "meta_version": 4,

        # label config
        "label_horizon": int(HORIZON),
        "label_thr": float(LABEL_THR),
        "label_dist": ydist,

        # sgd config
        "sgd_alpha": float(SGD_ALPHA),
        "sgd_max_iter": int(SGD_MAX_ITER),
        "sgd_tol": float(SGD_TOL),
        "sgd_loss": str(SGD_LOSS),
        "sgd_penalty": str(SGD_PENALTY),
        "sgd_shuffle": bool(SGD_SHUFFLE),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("[%s] Model & meta saved: %s | %s", interval, model_path, meta_path)


def main() -> None:
    logger.info("Offline SGD training started | symbol=%s | intervals=%s", SYMBOL, INTERVALS)
    logger.info(
        "LABEL_HORIZON=%d LABEL_THR=%g | SGD_ALPHA=%g MAX_ITER=%d TOL=%g | schema_len=%d (time=%s)",
        HORIZON, LABEL_THR, SGD_ALPHA, SGD_MAX_ITER, SGD_TOL, len(SGD_SCHEMA_USED), USE_TIME_FEATURES
    )
    logger.info("MODELS_DIR=%s | DATA_DIR=%s | MAX_BARS=%d", MODELS_DIR, DATA_DIR, MAX_BARS)

    for itv in INTERVALS:
        try:
            train_interval(itv)
        except FileNotFoundError as e:
            logger.warning("[%s] CSV yok, skip: %s", itv, e)
        except Exception as e:
            logger.exception("[%s] Train error: %s", itv, e)

    logger.info("Offline SGD training finished.")


if __name__ == "__main__":
    main()
