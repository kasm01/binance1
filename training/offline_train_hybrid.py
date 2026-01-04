#!/usr/bin/env python
import os

from app_paths import MODELS_DIR
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

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
SYMBOL    = _env_str("SYMBOL", "BTCUSDT")
DATA_DIR  = _env_str("DATA_DIR", "data/offline_cache")

# INTERVALS env: "1m,3m,5m,15m,30m,1h"
INTERVALS: List[str] = _env_csv_list("INTERVALS", ["1m","3m","5m","15m","30m","1h"])

MAX_BARS = _env_int("OFFLINE_MAX_BARS", 50000)

# Label config
HORIZON   = _env_int("LABEL_HORIZON", 1)
LABEL_THR = _env_float("LABEL_THR", 0.0)  # 0.0005 gibi threshold istersen

# SGD hyperparams (ENV)
SGD_ALPHA    = _env_float("SGD_ALPHA", 1e-4)
SGD_MAX_ITER = _env_int("SGD_MAX_ITER", 50)
SGD_TOL      = _env_float("SGD_TOL", 1e-3)
SGD_LOSS     = _env_str("SGD_LOSS", "log_loss")
SGD_PENALTY  = _env_str("SGD_PENALTY", "l2")
SGD_SHUFFLE  = _env_bool("SGD_SHUFFLE", False)
SGD_NJOBS    = _env_int("SGD_NJOBS", -1)
SGD_RANDOM_STATE = _env_int("SGD_RANDOM_STATE", 42)

# LSTM meta default (inference tarafı seq_len kullanıyor)
LSTM_SEQ_LEN_DEFAULT = _env_int("LSTM_SEQ_LEN_DEFAULT", 50)

# LSTM kullanım bayrağı: istersen sadece bazı interval’lerde aç
# ör: LSTM_INTERVALS="1m,5m,15m,3m,30m"
LSTM_INTERVALS: List[str] = _env_csv_list("LSTM_INTERVALS", ["1m","3m","5m","15m","30m","1h"])

# ------------------------------------------------------------
# Feature & Label helpers
# ------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # --- ROBUST NUMERIC COERCION (offline CSV -> safe) ---
    # Binance kline kolonları string gelebiliyor, arithmetic öncesi float'a çevir.
    num_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.ffill().bfill().fillna(0.0)

    # Column name normalization (LIVE schema compatibility)
    if "taker_buy_base_asset_volume" in df.columns and "taker_buy_base_volume" not in df.columns:
        df["taker_buy_base_volume"] = df["taker_buy_base_asset_volume"]
    if "taker_buy_quote_asset_volume" in df.columns and "taker_buy_quote_volume" not in df.columns:
        df["taker_buy_quote_volume"] = df["taker_buy_quote_asset_volume"]


    # Zaman kolonlarını saniye float yap (tutarlı)
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
            df[col] = dt.astype("int64") / 1e9

    # Base numeric
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    df["dummy_extra"] = 0.0

    # NA temizliği
    df = df.ffill().bfill().fillna(0.0)
    return df

def build_labels(close: pd.Series, horizon: int, thr: float) -> pd.Series:
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    y = (ret > float(thr)).astype(int)
    return y

def make_xy(df_raw: pd.DataFrame, horizon: int, thr: float) -> Tuple[pd.DataFrame, pd.Series]:
    feat_df = build_features(df_raw)
    y_all = build_labels(feat_df["close"], horizon=horizon, thr=thr)

    mask = y_all.notna()
    feat_df_aligned = feat_df[mask].copy()
    y_aligned = y_all[mask].astype(int)
    return feat_df_aligned, y_aligned

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
    X_df, y = make_xy(raw_df, horizon=HORIZON, thr=LABEL_THR)

    if len(X_df) < 1000:
        logger.warning("[%s] Çok az sample: n=%d, skip.", interval, len(X_df))
        return

    # Safety: prevent schema drift (asset_volume cols must never reach training matrix)
    for _c in ("taker_buy_base_asset_volume","taker_buy_quote_asset_volume"):
        if _c in X_df.columns:
            X_df = X_df.drop(columns=[_c])

    X_num = X_df.select_dtypes(include=[np.number])
    feature_cols = list(X_num.columns)
    X = X_num.to_numpy(dtype=float)

    logger.info("[%s] X shape=%s | y len=%d | n_features=%d", interval, X.shape, len(y), X.shape[1])

    n = X.shape[0]
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X[split_idx:], y.iloc[split_idx:]

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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

    logger.info("[%s] Training SGD... alpha=%g max_iter=%d tol=%g", interval, SGD_ALPHA, SGD_MAX_ITER, SGD_TOL)
    clf.fit(X_train, y_train)

    try:
        proba = clf.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc = float(roc_auc_score(y_val, proba))
        else:
            auc = 0.5
            logger.warning("[%s] y_val tek sınıf, AUC=0.5 set edildi.", interval)
    except Exception as e:
        logger.warning("[%s] AUC hesap hatası: %s", interval, e)
        auc = 0.5

    logger.info("[%s] Validation AUC=%.4f | label_mean=%.4f", interval, auc, float(y.mean()))

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"online_model_{interval}_best.joblib")
    joblib.dump(clf, model_path)

    best_side = "long" if auc >= 0.5 else "short"

    meta_path = os.path.join(MODELS_DIR, f"model_meta_{interval}.json")
    meta: Dict[str, Any] = {
        "interval": interval,
        "symbol": SYMBOL,
        "feature_schema": feature_cols,
        "feature_source": "offline_train_hybrid::build_features",
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "best_auc": float(auc),
        "best_side": best_side,
        "use_lstm_hybrid": True if interval in set(LSTM_INTERVALS) else False,
        "seq_len": int(LSTM_SEQ_LEN_DEFAULT),
        "meta_version": 2,
        "label_horizon": int(HORIZON),
        "label_thr": float(LABEL_THR),
        "sgd_alpha": float(SGD_ALPHA),
        "sgd_max_iter": int(SGD_MAX_ITER),
        "sgd_tol": float(SGD_TOL),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("[%s] Model & meta saved: %s | %s", interval, model_path, meta_path)

def main() -> None:
    logger.info("Offline SGD training started | symbol=%s | intervals=%s", SYMBOL, INTERVALS)
    logger.info("LABEL_HORIZON=%d LABEL_THR=%g | SGD_ALPHA=%g MAX_ITER=%d TOL=%g", HORIZON, LABEL_THR, SGD_ALPHA, SGD_MAX_ITER, SGD_TOL)

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
