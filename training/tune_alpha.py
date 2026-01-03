#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so `import app_paths` works when running from /training)
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import math
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from app_paths import MODELS_DIR
from features.schema import normalize_to_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("tune_alpha")


# -------------------------
# Env helpers
# -------------------------
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()

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


SYMBOL   = _env_str("SYMBOL", "BTCUSDT")
INTERVAL = _env_str("INTERVAL", "5m")
DATA_DIR = _env_str("DATA_DIR", "data/offline_cache")

OFFLINE_MAX_BARS = _env_int("OFFLINE_MAX_BARS", 50000)
LABEL_HORIZON    = _env_int("LABEL_HORIZON", 3)
LABEL_THR        = _env_float("LABEL_THR", 0.0005)

ALPHA_MIN  = _env_float("ALPHA_MIN", 0.0)
ALPHA_MAX  = _env_float("ALPHA_MAX", 1.0)
ALPHA_STEP = _env_float("ALPHA_STEP", 0.05)

SGD_MAX_ITER = _env_int("SGD_MAX_ITER", 20000)
SGD_TOL      = _env_float("SGD_TOL", 1e-5)
RANDOM_SEED  = _env_int("SEED", 42)

TRAIN_SPLIT = _env_float("TRAIN_SPLIT", 0.80)  # time split

SAVE_MODEL = _env_int("SAVE_MODEL", 1)  # 0 => sadece rapor, yazmaz
OVERWRITE  = _env_int("OVERWRITE", 1)


# -------------------------
# CSV loading
# -------------------------
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

def load_offline_csv(path: str, max_bars: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] >= len(COLUMNS):
        df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS[:df.shape[1]]

    # numeric cast
    float_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    if max_bars > 0 and len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)

    return df


# -------------------------
# Feature builder discovery
# -------------------------
def resolve_build_features():
    """
    Projede build_features nerede ise onu bulmaya çalışır.
    Öncelik: training/offline_train_hybrid.py -> main.py -> local fallback
    """
    # 1) training.offline_train_hybrid
    try:
        from training.offline_train_hybrid import build_features as bf  # type: ignore
        log.info("Using build_features from training.offline_train_hybrid")
        return bf
    except Exception:
        pass

    # 2) main.build_features
    try:
        from main import build_features as bf  # type: ignore
        log.info("Using build_features from main.py")
        return bf
    except Exception:
        pass

    # 3) fallback (minimum)
    def _fallback_build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()
        df["hl_range"] = df["high"] - df["low"]
        df["oc_change"] = df["close"] - df["open"]
        df["return_1"] = df["close"].pct_change(1)
        df["return_3"] = df["close"].pct_change(3)
        df["return_5"] = df["close"].pct_change(5)
        df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
        df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
        df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()
        df["vol_10"] = df["volume"].rolling(10, min_periods=1).std()
        df["dummy_extra"] = 0.0
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return df

    log.warning("build_features not found in project; using fallback builder (may reduce quality).")
    return _fallback_build_features


# -------------------------
# Labels + dataset
# -------------------------
def make_xy(raw_df: pd.DataFrame, feature_schema: List[str], build_features_fn) -> Tuple[np.ndarray, np.ndarray]:
    feat_df = build_features_fn(raw_df)

    # schema lock (order + missing fill)
    if feature_schema:
        feat_df = normalize_to_schema(feat_df, feature_schema)

    # label: future close return
    if "close" not in feat_df.columns:
        raise ValueError("Feature DF missing 'close' column; cannot build labels.")

    future_close = feat_df["close"].shift(-int(LABEL_HORIZON))
    ret = (future_close / feat_df["close"]) - 1.0
    y = (ret > float(LABEL_THR)).astype(int)

    m = y.notna()
    X_df = feat_df.loc[m]
    y_s = y.loc[m].astype(int)

    # ensure numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = X_df.to_numpy(dtype=float)
    yv = y_s.to_numpy(dtype=int)

    return X, yv


def time_split(X: np.ndarray, y: np.ndarray, train_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    if n < 2000:
        log.warning("Dataset small (n=%d). Alpha tuning still runs but AUC may be noisy.", n)

    split = int(max(1, min(n - 1, math.floor(n * train_ratio))))
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    return X_tr, y_tr, X_va, y_va


def auc_safe(y_true: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, p))
    except Exception:
        return 0.5


def alpha_grid(a_min: float, a_max: float, a_step: float) -> List[float]:
    a_min = float(a_min)
    a_max = float(a_max)
    a_step = float(a_step)
    if a_step <= 0:
        return [max(a_min, 1e-6)]
    vals = []
    x = a_min
    # include end
    while x <= a_max + 1e-12:
        vals.append(float(round(x, 12)))
        x += a_step
    # avoid all zeros -> include small epsilon as well
    if all(v == 0.0 for v in vals):
        vals.append(1e-6)
    return vals


def load_meta(interval: str) -> Tuple[Dict[str, Any], str]:
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{interval}.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}, meta_path
        except Exception:
            return {}, meta_path
    return {}, meta_path


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, f"{SYMBOL}_{INTERVAL}_6m.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV yok: {csv_path}")

    meta, meta_path = load_meta(INTERVAL)
    feature_schema = meta.get("feature_schema") or []
    if not isinstance(feature_schema, list):
        feature_schema = []

    build_features_fn = resolve_build_features()

    raw_df = load_offline_csv(csv_path, OFFLINE_MAX_BARS)

    X, y = make_xy(raw_df, feature_schema, build_features_fn)
    X_tr, y_tr, X_va, y_va = time_split(X, y, TRAIN_SPLIT)

    log.info("TUNE ALPHA %s | X_tr=%s X_va=%s | horizon=%d thr=%.6f", INTERVAL, X_tr.shape, X_va.shape, LABEL_HORIZON, LABEL_THR)

    grid = alpha_grid(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP)

    best = {
        "alpha": None,
        "auc": -1.0,
        "pos_rate_train": float(np.mean(y_tr)) if len(y_tr) else 0.0,
        "pos_rate_val": float(np.mean(y_va)) if len(y_va) else 0.0,
    }

    results: List[Dict[str, Any]] = []

    for a in grid:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("sgd", SGDClassifier(
                    loss="log_loss",
                    alpha=float(a),
                    max_iter=int(SGD_MAX_ITER),
                    tol=float(SGD_TOL),
                    random_state=int(RANDOM_SEED),
                    class_weight="balanced",
                )),
            ]
        )

        try:
            clf.fit(X_tr, y_tr)
            p = clf.predict_proba(X_va)[:, 1]
            auc = auc_safe(y_va, p)
        except Exception as e:
            log.warning("alpha=%.6f failed: %s", float(a), e)
            auc = 0.5

        results.append({"alpha": float(a), "val_auc": float(auc)})

        if auc > best["auc"]:
            best["auc"] = float(auc)
            best["alpha"] = float(a)

    # retrain best on full data (train+val) for final save
    best_alpha = float(best["alpha"]) if best["alpha"] is not None else 1e-4
    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("sgd", SGDClassifier(
                loss="log_loss",
                alpha=float(best_alpha),
                max_iter=int(SGD_MAX_ITER),
                tol=float(SGD_TOL),
                random_state=int(RANDOM_SEED),
                class_weight="balanced",
            )),
        ]
    )
    final_model.fit(X, y)

    out_path = os.path.join(MODELS_DIR, f"online_model_{INTERVAL}_best.joblib")

    if SAVE_MODEL:
        if (not os.path.exists(out_path)) or OVERWRITE:
            joblib.dump(final_model, out_path)
            log.info("Saved model: %s", out_path)
        else:
            log.info("Model exists and OVERWRITE=0, skipped save: %s", out_path)

    # meta update
    meta = meta if isinstance(meta, dict) else {}
    meta.update({
        "sgd_alpha": float(best_alpha),
        "sgd_val_auc": float(best["auc"]),
        "sgd_alpha_grid": {"min": float(ALPHA_MIN), "max": float(ALPHA_MAX), "step": float(ALPHA_STEP)},
        "sgd_max_iter": int(SGD_MAX_ITER),
        "sgd_tol": float(SGD_TOL),
        "label_horizon": int(LABEL_HORIZON),
        "label_thr": float(LABEL_THR),
        "alpha_tuned": True,
        "alpha_tuned_at": pd.Timestamp.utcnow().isoformat(),
        # MTF standardization için kalıcı alan
        "auc_used": float(meta.get("auc_used", meta.get("best_auc", 0.5)) or 0.5),
        "auc_used_source": str(meta.get("auc_used_source", "auto_fallback_from_existing_keys")),
    })

    # best_auc'yi SGD açısından güncellemek istersen: val_auc daha iyi ise
    try:
        prev_best = float(meta.get("best_auc", 0.5) or 0.5)
    except Exception:
        prev_best = 0.5

    if float(best["auc"]) > prev_best:
        meta["best_auc"] = float(best["auc"])
        meta["best_auc_source"] = "tune_alpha_val_auc"

    # AUC_USED: en azından best_auc ile sync kalsın
    try:
        meta["auc_used"] = float(meta.get("best_auc", meta["auc_used"]))
        meta["auc_used_source"] = "sync_from_best_auc_after_tune"
    except Exception:
        pass

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # print summary (stdout)
    results_sorted = sorted(results, key=lambda r: r["val_auc"], reverse=True)
    topk = results_sorted[:10]

    print("=" * 80)
    print("INTERVAL:", INTERVAL, "| best_alpha:", best_alpha, "| best_val_auc:", float(best["auc"]))
    print("pos_rate_train:", best["pos_rate_train"], "| pos_rate_val:", best["pos_rate_val"])
    print("TOP10:")
    for r in topk:
        print("  alpha=%-10s val_auc=%.6f" % (r["alpha"], r["val_auc"]))
    print("Saved:", out_path if SAVE_MODEL else "(SAVE_MODEL=0)")
    print("Meta :", meta_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
