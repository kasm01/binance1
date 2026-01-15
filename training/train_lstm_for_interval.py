#!/usr/bin/env python
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
import tempfile
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# OPTIONAL CONFIG (patched)
try:
    from config.config import Config as _Config  # type: ignore
    _CFG = _Config()
except Exception:
    _CFG = None

# SINGLE SOURCE OF TRUTH for features
from features.pipeline import make_matrix, FEATURE_SCHEMA_22  # type: ignore

logger = logging.getLogger("lstm_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------
# Helpers
# ---------------------------
def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


def _interval_seq_len(interval: str, default: int = 50) -> int:
    """
    .env:
      LSTM_SEQ_LEN_1m=50
      LSTM_SEQ_LEN_3m=50
      ...
    fallback:
      LSTM_WINDOW
      default
    """
    key = f"LSTM_SEQ_LEN_{interval}"
    return _env_int(key, _env_int("LSTM_WINDOW", default))


def _load_meta(interval: str) -> dict:
    meta_path = Path("models") / f"model_meta_{interval}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[META] meta dosyası yok: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_feature_cols(interval: str) -> List[str]:
    """
    Öncelik:
      1) lstm_feature_schema (yeni)
      2) feature_schema (sgd schema)
      3) feature_cols (legacy)
    """
    meta = _load_meta(interval)
    feats = meta.get("lstm_feature_schema") or meta.get("feature_schema") or meta.get("feature_cols")

    if not feats or not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
        raise ValueError("[META] lstm_feature_schema/feature_schema/feature_cols meta içinde yok veya geçersiz.")

    logger.info("[META] feature schema loaded | interval=%s | n=%d", interval, len(feats))
    return list(feats)


def _load_label_thr(interval: str) -> float:
    """
    Öncelik:
      1) LSTM_LABEL_THR
      2) LABEL_THR
      3) model_meta_<interval>.json içindeki label_thr
      4) 0.0
    """
    if os.getenv("LSTM_LABEL_THR"):
        return _env_float("LSTM_LABEL_THR", 0.0)
    if os.getenv("LABEL_THR"):
        return _env_float("LABEL_THR", 0.0)

    try:
        meta = _load_meta(interval)
        if meta.get("label_thr") is not None:
            return float(meta["label_thr"])
    except Exception:
        pass

    return 0.0


def _load_label_horizon(interval: str, default: int = 1) -> int:
    """
    Öncelik:
      1) LSTM_HORIZON env
      2) LABEL_HORIZON env
      3) model_meta_<interval>.json içindeki label_horizon
      4) default
    """
    if os.getenv("LSTM_HORIZON"):
        return _env_int("LSTM_HORIZON", default)
    if os.getenv("LABEL_HORIZON"):
        return _env_int("LABEL_HORIZON", default)

    try:
        meta = _load_meta(interval)
        if meta.get("label_horizon") is not None:
            return int(meta["label_horizon"])
    except Exception:
        pass

    return int(default)


def make_labels_from_close(close: np.ndarray, horizon: int, thr: float) -> np.ndarray:
    """
    y[t] = 1 if (close[t+h] / close[t] - 1) > thr else 0
    last horizon rows forced to 0 (no future)
    """
    close = np.asarray(close, dtype=float)
    close = np.where(np.isfinite(close), close, np.nan)

    fut = np.roll(close, -horizon)
    ret = (fut / np.maximum(close, 1e-12)) - 1.0
    y = (ret > float(thr)).astype(int)

    if horizon > 0:
        y[-horizon:] = 0

    y = np.where(np.isfinite(ret), y, 0).astype(int)
    return y


def align_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean X (finite rows) and align y to same length.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n = min(len(y), X.shape[0])
    X = X[:n]
    y = y[:n]

    X = np.where(np.isfinite(X), X, np.nan)
    row_ok = ~np.isnan(X).any(axis=1)
    X = X[row_ok]
    y = y[row_ok]

    return X, y


# ---------------------------
# Data pipeline
# ---------------------------
_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline cache dosyaları bazen header=None / bazen header var.
    Burada 12 kolon kline formatına normalize ediyoruz.
    """
    if df is None or df.empty:
        return df

    if df.shape[1] > 12:
        df = df.iloc[:, :12].copy()

    if list(df.columns) == list(range(len(df.columns))) and df.shape[1] == 12:
        df = df.copy()
        df.columns = _KLINE_COLS

    if df.shape[1] == 12 and not set(_KLINE_COLS).issubset(set(df.columns)):
        df = df.copy()
        df.columns = _KLINE_COLS

    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df = df.replace([float("inf"), float("-inf")], pd.NA).ffill().bfill().fillna(0)
    return df


def load_offline_klines(symbol: str, interval: str, limit: int = 20000) -> pd.DataFrame:
    """
    Offline cache'ten klines yükler:
      data/offline_cache/{symbol}_{interval}_6m.csv
    """
    path = Path("data/offline_cache") / f"{symbol}_{interval}_6m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Offline cache yok: {path}")

    df = pd.read_csv(path, header=None, low_memory=False)
    df = _normalize_kline_df(df)

    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info(
        "[DATA] Loaded offline klines: symbol=%s interval=%s shape=%s path=%s",
        symbol, interval, df.shape, str(path),
    )
    return df


def make_lstm_dataset(X: np.ndarray, y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    LSTM için (window, n_features) sekansları üretir.
    X: (N, n_features)
    y: (N,)
    """
    xs, ys = [], []
    for i in range(window, len(X)):
        xs.append(X[i - window: i])
        ys.append(y[i])
    X_seq = np.asarray(xs, dtype=np.float32)
    y_seq = np.asarray(ys, dtype=np.float32)
    return X_seq, y_seq


def prepare_data(
    symbol: str,
    interval: str,
    horizon: int,
    window: int,
    thr: float,
    limit_rows: int,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str], Dict[str, Any]]:
    """
    - Offline kline yükler
    - Meta'dan feature schema alır (lstm_feature_schema > feature_schema)
    - make_matrix ile X üretir (SGD ile aynı sözleşme)
    - close üzerinden label üretir (horizon/thr)
    - align + clean
    - StandardScaler ile scale
    - LSTM seq dataset üretir
    """
    if window < 2:
        raise ValueError(f"[DATA] window çok küçük: window={window}")

    df_raw = load_offline_klines(symbol, interval, limit=limit_rows)

    feature_cols = _load_feature_cols(interval)
    meta = _load_meta(interval)

    # ---- SINGLE SOURCE OF TRUTH ----
    X = make_matrix(df_raw, schema=feature_cols)
    X = np.asarray(X, dtype=float)

    # label: close üzerinden
    close = pd.to_numeric(df_raw["close"], errors="coerce").astype(float).values
    y = make_labels_from_close(close, horizon=horizon, thr=thr)

    # horizon > 0 ise son horizon satır “gelecek yok” -> X/y kırp
    if horizon > 0:
        X = X[:-horizon] if X.shape[0] > horizon else X[:0]
        y = y[:-horizon] if len(y) > horizon else y[:0]

    # align + clean
    X, y = align_xy(X, y)

    if X.shape[0] <= window + 5:
        raise ValueError(f"[DATA] Yetersiz örnek: N={X.shape[0]} window={window} (sequence üretilemez)")

    # hedef dağılımı (kalite kontrol)
    u, cnt = np.unique(y, return_counts=True)
    ydist = dict(zip(u.tolist(), cnt.tolist()))
    pos_ratio = float((y == 1).mean()) if len(y) else 0.0
    logger.info("[DATA] X=%s y=%s | thr=%.8f horizon=%d | ydist=%s pos_ratio=%.4f",
                X.shape, y.shape, thr, horizon, ydist, pos_ratio)

    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # sequences
    X_seq, y_seq = make_lstm_dataset(X_scaled, y.astype(np.float32), window)
    logger.info("[DATA] LSTM seq dataset: X_seq=%s y_seq=%s (window=%d n_features=%d)",
                X_seq.shape, y_seq.shape, window, X_seq.shape[2])

    return X_seq, y_seq, scaler, feature_cols, {"ydist": ydist, "pos_ratio": pos_ratio, "meta": meta}


# ---------------------------
# Model
# ---------------------------
def build_lstm_model(window: int, n_features: int) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(window, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=_env_float("LSTM_LR", 1e-3))
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ---------------------------
# Save (atomic)
# ---------------------------
def _atomic_joblib_dump(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, tmp)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_model_save(model: tf.keras.Model, path: Path) -> None:
    """
    Atomik HDF5 (.h5) save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp.h5", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        tf.keras.models.save_model(model, str(tmp_path), save_format="h5")
        os.replace(str(tmp_path), str(path))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _save_scalers(models_dir: Path, interval: str, scaler: StandardScaler) -> None:
    """
    Projede iki isim var:
      - lstm_scaler_{interval}.joblib
      - lstm_{interval}_scaler.joblib
    """
    p1 = models_dir / f"lstm_scaler_{interval}.joblib"
    p2 = models_dir / f"lstm_{interval}_scaler.joblib"

    _atomic_joblib_dump(scaler, p1)
    _atomic_joblib_dump(scaler, p2)

    ok1 = p1.exists() and p1.stat().st_size > 0
    ok2 = p2.exists() and p2.stat().st_size > 0
    if not (ok1 and ok2):
        raise RuntimeError(f"[SAVE] scaler doğrulama FAIL: {p1} ok={ok1}, {p2} ok={ok2}")

    logger.info("[SAVE] Scaler saved: %s (and alias %s)", p1, p2)


def _save_models(models_dir: Path, interval: str, model: tf.keras.Model) -> None:
    long_path = models_dir / f"lstm_long_{interval}.h5"
    short_path = models_dir / f"lstm_short_{interval}.h5"

    _atomic_model_save(model, long_path)
    _atomic_model_save(model, short_path)

    logger.info("[SAVE] Models saved: %s and %s", long_path, short_path)


def _save_lstm_meta(models_dir: Path, interval: str, payload: Dict[str, Any]) -> None:
    p = models_dir / f"lstm_meta_{interval}.json"
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    logger.info("[SAVE] LSTM meta saved: %s", p)


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM for a given interval")
    parser.add_argument("--symbol", type=str, default=None, help="e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default=None, help="e.g. 1m,3m,5m,15m,30m,1h")

    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)

    args = parser.parse_args()

    symbol = (
        (args.symbol or os.getenv("SYMBOL"))
        or ("BTCUSDT" if _CFG is None else getattr(_CFG, "SYMBOL", "BTCUSDT"))
    )
    symbol = str(symbol).strip().upper()

    interval = str(args.interval or os.getenv("INTERVAL", "5m")).strip().lower()

    # horizon / thr / window
    horizon = int(args.horizon if args.horizon is not None else _load_label_horizon(interval, default=1))
    thr = float(_load_label_thr(interval))

    if args.window is not None:
        window_size = int(args.window)
    else:
        # meta'da seq_len varsa onu kullanmayı tercih edelim
        try:
            meta = _load_meta(interval)
            window_size = int(meta.get("seq_len") or _interval_seq_len(interval, default=50))
        except Exception:
            window_size = _interval_seq_len(interval, default=50)

    if window_size < 2:
        raise ValueError(f"[ARGS] window_size too small: {window_size}")

    batch_size = int(args.batch if args.batch is not None else _env_int("LSTM_BATCH", 64))
    patience = int(args.patience if args.patience is not None else _env_int("LSTM_PATIENCE", 5))

    val_split = float(args.val_split if args.val_split is not None else _env_float("LSTM_VAL_SPLIT", 0.2))
    val_split = min(0.40, max(0.05, val_split))

    epochs = int(args.epochs if args.epochs is not None else _env_int("LSTM_EPOCHS", 20))
    limit_rows = int(args.limit_rows if args.limit_rows is not None else _env_int("LSTM_LIMIT_ROWS", 20000))

    logger.info(
        "[START] LSTM training | symbol=%s interval=%s horizon=%d window=%d thr=%.8f batch=%d epochs=%d val_split=%.3f limit_rows=%d pid=%d",
        symbol, interval, horizon, window_size, thr, batch_size, epochs, val_split, limit_rows, os.getpid()
    )

    X_seq, y_seq, scaler, schema_used, info = prepare_data(
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        window=window_size,
        thr=thr,
        limit_rows=limit_rows,
    )

    n_samples = int(X_seq.shape[0])
    if n_samples < 500:
        logger.warning("[WARN] Çok az örnek: n=%d (window=%d). Eğitim kalitesi düşük olabilir.", n_samples, window_size)

    # time-series split: last val_split for validation
    val_n = max(1, int(n_samples * float(val_split)))
    train_n = max(1, n_samples - val_n)

    X_train, X_val = X_seq[:train_n], X_seq[train_n:]
    y_train, y_val = y_seq[:train_n], y_seq[train_n:]

    logger.info("[SPLIT] train=%d val=%d", X_train.shape[0], X_val.shape[0])

    model = build_lstm_model(window=window_size, n_features=int(X_seq.shape[2]))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / f"lstm_{interval}_best_tmp_{os.getpid()}.h5"
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=max(1, patience // 2),
            verbose=1,
        ),
    ]

    logger.info("[FIT] Training starting...")
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # best checkpoint load
    if ckpt_path.exists():
        logger.info("[LOAD] Best checkpoint loading: %s", ckpt_path)
        best_model = tf.keras.models.load_model(str(ckpt_path), compile=False)
        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        logger.info("[LOAD] Best checkpoint yok, son modeli kullanıyorum.")
        best_model = model

    logger.info("[SAVE] Writing final artifacts for interval=%s", interval)
    _save_models(models_dir, interval, best_model)
    _save_scalers(models_dir, interval, scaler)

    # LSTM meta (debug + contract)
    last = hist.history
    def _last(v):
        try:
            return float(v[-1]) if isinstance(v, list) and v else None
        except Exception:
            return None

    lstm_meta = {
        "interval": interval,
        "symbol": symbol,
        "window": int(window_size),
        "horizon": int(horizon),
        "thr": float(thr),
        "feature_schema_used": list(schema_used),
        "n_features": int(X_seq.shape[2]),
        "n_sequences": int(X_seq.shape[0]),
        "val_split": float(val_split),
        "ydist": info.get("ydist"),
        "pos_ratio": info.get("pos_ratio"),
        "metrics_last": {
            "loss": _last(last.get("loss")),
            "auc": _last(last.get("auc")),
            "val_loss": _last(last.get("val_loss")),
            "val_auc": _last(last.get("val_auc")),
        },
        "feature_source": "train_lstm_for_interval::features.pipeline.make_matrix(meta_schema)",
        "meta_version": 2,
    }
    _save_lstm_meta(models_dir, interval, lstm_meta)

    logger.info("[DONE] LSTM training completed | interval=%s", interval)


if __name__ == "__main__":
    main()
