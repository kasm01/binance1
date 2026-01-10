#!/usr/bin/env python
import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# OPTIONAL CONFIG (patched)
try:
    from config.config import Config as _Config  # type: ignore
    _CFG = _Config()
except Exception:
    _CFG = None

# Projedeki feature/label fonksiyonlarını kullan
# NOT: build_labels artık thr zorunlu; bu dosyada thr yönetimi eklendi.
from features.fe_labels import build_features, build_labels  # type: ignore

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


def _load_meta_feature_cols(interval: str) -> List[str]:
    """
    models/model_meta_<interval>.json içinden SGD'nin kullandığı feature kolonlarını çeker.
    LSTM, SGD ile aynı feature düzenini kullanır.

    Not:
      - Yeni meta anahtarı: feature_schema
      - Eski/opsiyonel: feature_cols
    """
    meta = _load_meta(interval)
    feats = meta.get("feature_schema") or meta.get("feature_cols")
    if not feats or not isinstance(feats, list) or not all(isinstance(x, str) for x in feats):
        raise ValueError("[META] feature_schema/feature_cols meta içinde yok veya geçersiz.")

    logger.info("[META] feature columns loaded from models/model_meta_%s.json | n=%d", interval, len(feats))
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


def _ensure_feature_schema(feat_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    - Meta'daki feature sırasını birebir uygular.
    - Eksik feature varsa 0.0 ile oluşturur.
    """
    out = feat_df.copy()

    # Backward-compat alias map (projede bazı isimler değişmiş olabilir)
    alias_map = {
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }
    for old_col, new_col in alias_map.items():
        if old_col not in out.columns and new_col in out.columns:
            out[old_col] = out[new_col]

    missing = [c for c in feature_cols if c not in out.columns]
    if missing:
        logger.warning("[FE] Eksik feature kolonları (0 ile doldurulacak): %s", missing)
        for c in missing:
            out[c] = 0.0

    # Sadece meta şemasını al ve sıralamayı sabitle
    X_df = out[feature_cols].copy()

    # Her şey numeric olsun
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

    return X_df


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

    # Çok kolon varsa ilk 12'yi al
    if df.shape[1] > 12:
        df = df.iloc[:, :12].copy()

    # Header yoksa kolonlar 0..11 olur -> isimlendir
    if list(df.columns) == list(range(len(df.columns))) and df.shape[1] == 12:
        df = df.copy()
        df.columns = _KLINE_COLS

    # Kolon isimleri tam oturmamışsa ama 12 kolon varsa yine isimlendir
    if df.shape[1] == 12 and not set(_KLINE_COLS).issubset(set(df.columns)):
        df = df.copy()
        df.columns = _KLINE_COLS

    # Tipleri toparla
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

    # header belirsiz -> header=None ile okuyup normalize ediyoruz
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
    horizon: int = 1,
    window: int = 50,
    thr: float = 0.0,
    limit_rows: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    - Offline kline yükler
    - build_features ile feature'ları üretir
    - model_meta_<interval>.json'dan feature list alır (SGD ile aynı schema)
    - build_labels ile binary hedef üretir (thr zorunlu)
    - StandardScaler ile scale
    - LSTM için sekans dataset üretir
    """
    if window < 2:
        raise ValueError(f"[DATA] window çok küçük: window={window}")

    df_raw = load_offline_klines(symbol, interval, limit=limit_rows)

    # Feature engineering (offline_train_hybrid ile uyumlu)
    feat_df = build_features(df_raw)

    feature_cols = _load_meta_feature_cols(interval)
    X_df = _ensure_feature_schema(feat_df, feature_cols)

    # Label üret (feat_df üzerinden; build_labels'ın beklediği df bu)
    labels = build_labels(feat_df, horizon=horizon, thr=thr)

    # horizon > 0 ise en son horizon satırların label'ı invalid olur
    if horizon > 0:
        X_df2 = X_df.iloc[:-horizon].reset_index(drop=True)
        y_ser = labels.iloc[:-horizon].reset_index(drop=True)
    else:
        X_df2 = X_df.reset_index(drop=True)
        y_ser = labels.reset_index(drop=True)

    y = pd.to_numeric(y_ser, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=np.float32)
    X = X_df2.to_numpy(dtype=np.float32)

    if len(X) != len(y):
        n = min(len(X), len(y))
        logger.warning("[DATA] X/y length mismatch -> truncating to n=%d (X=%d y=%d)", n, len(X), len(y))
        X = X[:n]
        y = y[:n]

    logger.info("[DATA] After feature+label alignment: X shape=%s, y shape=%s | thr=%.8f", X.shape, y.shape, thr)

    if len(X) <= window + 5:
        raise ValueError(f"[DATA] Yetersiz örnek: N={len(X)} window={window} (sequence üretilemez)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_seq, y_seq = make_lstm_dataset(X_scaled, y, window)
    logger.info("[DATA] LSTM seq dataset: X_seq shape=%s, y_seq shape=%s (window=%d)", X_seq.shape, y_seq.shape, window)

    return X_seq, y_seq, scaler


# ---------------------------
# Model
# ---------------------------
def build_lstm_model(window: int, n_features: int) -> tf.keras.Model:
    """
    Basit ama güçlü bir LSTM mimarisi (binary classification).
    """
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
# Train + Save (atomic + validated)
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
    tmp = Path(str(path) + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        model.save(tmp)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _save_scalers(models_dir: Path, interval: str, scaler: StandardScaler) -> None:
    """
    Projede iki isim var:
      - lstm_scaler_{interval}.joblib
      - lstm_{interval}_scaler.joblib
    İkisini de yazarız.
    Atomik + doğrulamalı.
    """
    p1 = models_dir / f"lstm_scaler_{interval}.joblib"
    p2 = models_dir / f"lstm_{interval}_scaler.joblib"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        if scaler is None:
            raise ValueError("scaler is None")

        _atomic_joblib_dump(scaler, p1)
        _atomic_joblib_dump(scaler, p2)

        ok1 = p1.exists() and p1.stat().st_size > 0
        ok2 = p2.exists() and p2.stat().st_size > 0
        if not (ok1 and ok2):
            logger.warning("[SAVE] Scaler doğrulama FAIL | p1_ok=%s p2_ok=%s | %s | %s", ok1, ok2, p1, p2)
        else:
            logger.info("[SAVE] Scaler saved: %s (and alias %s)", p1, p2)

    except Exception as e:
        logger.exception("[SAVE] Scaler save failed | interval=%s | err=%s", interval, e)
        raise


def _save_models(models_dir: Path, interval: str, model: tf.keras.Model) -> None:
    """
    Hybrid tarafında hem long hem short isimleri var.
    Aynı modeli iki isimle kaydediyoruz (uyumluluk).
    Atomik + doğrulamalı.
    """
    long_path = models_dir / f"lstm_long_{interval}.h5"
    short_path = models_dir / f"lstm_short_{interval}.h5"

    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        if model is None:
            raise ValueError("model is None")

        _atomic_model_save(model, long_path)
        _atomic_model_save(model, short_path)

        ok1 = long_path.exists() and long_path.stat().st_size > 0
        ok2 = short_path.exists() and short_path.stat().st_size > 0
        if not (ok1 and ok2):
            logger.warning("[SAVE] Model doğrulama FAIL | long_ok=%s short_ok=%s | %s | %s", ok1, ok2, long_path, short_path)
        else:
            logger.info("[SAVE] Models saved: %s and %s", long_path, short_path)

    except Exception as e:
        logger.exception("[SAVE] Model save failed | interval=%s | err=%s", interval, e)
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM for a given interval")
    parser.add_argument("--symbol", type=str, default=None, help="e.g. BTCUSDT")
    parser.add_argument("--interval", type=str, default=None, help="e.g. 1m,3m,5m,15m,30m,1h")

    # Optional overrides (CLI > env > default)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--limit_rows", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)  # istersen manuel override

    args = parser.parse_args()

    # SYMBOL öncelik: CLI -> env -> config -> fallback
    symbol = (
        (args.symbol or os.getenv("SYMBOL"))
        or ("BTCUSDT" if _CFG is None else getattr(_CFG, "SYMBOL", "BTCUSDT"))
    )
    symbol = str(symbol).strip().upper()

    # INTERVAL öncelik: CLI -> env -> fallback
    interval = str(args.interval or os.getenv("INTERVAL", "5m")).strip()

    # Horizon / thr / window
    horizon = int(args.horizon if args.horizon is not None else _env_int("LSTM_HORIZON", 1))
    thr = _load_label_thr(interval)

    if args.window is not None:
        window_size = int(args.window)
    else:
        window_size = _interval_seq_len(interval, default=50)

    # Train params (CLI -> env -> default)
    batch_size = int(args.batch if args.batch is not None else _env_int("LSTM_BATCH", 64))
    patience = int(args.patience if args.patience is not None else _env_int("LSTM_PATIENCE", 5))
    val_split = float(args.val_split if args.val_split is not None else _env_float("LSTM_VAL_SPLIT", 0.2))
    epochs = int(args.epochs if args.epochs is not None else _env_int("LSTM_EPOCHS", 20))

    # Veri limiti
    limit_rows = int(args.limit_rows if args.limit_rows is not None else _env_int("LSTM_LIMIT_ROWS", 20000))

    logger.info(
        "[START] LSTM training | symbol=%s interval=%s horizon=%d window=%d thr=%.8f batch=%d epochs=%d val_split=%.3f limit_rows=%d",
        symbol, interval, horizon, window_size, thr, batch_size, epochs, val_split, limit_rows,
    )

    X_seq, y_seq, scaler = prepare_data(
        symbol=symbol,
        interval=interval,
        horizon=horizon,
        window=window_size,
        thr=thr,
        limit_rows=limit_rows,
    )

    n_samples = int(X_seq.shape[0])
    if n_samples < 500:
        logger.warning(
            "[WARN] Çok az örnek var: n=%d (window=%d). Eğitim kalitesi düşük olabilir.",
            n_samples, window_size
        )

    # Split (time-series): son val_split valid
    val_n = max(1, int(n_samples * float(val_split)))
    train_n = max(1, n_samples - val_n)

    X_train, X_val = X_seq[:train_n], X_seq[train_n:]
    y_train, y_val = y_seq[:train_n], y_seq[train_n:]

    logger.info("[SPLIT] train=%d val=%d", X_train.shape[0], X_val.shape[0])

    model = build_lstm_model(window=window_size, n_features=int(X_seq.shape[2]))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint (best auc)
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
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )

    # Best checkpoint varsa onu yükle
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

    _save_models(models_dir, interval, best_model)
    _save_scalers(models_dir, interval, scaler)

    logger.info("[DONE] LSTM training completed | interval=%s", interval)


if __name__ == "__main__":
    main()
