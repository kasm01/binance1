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
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _interval_seq_len(interval: str, default: int = 50) -> int:
    """
    .env:
      LSTM_SEQ_LEN_1m=50
      LSTM_SEQ_LEN_3m=50
      ...
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
    if not feats or not isinstance(feats, list):
        raise ValueError("[META] feature_schema/feature_cols meta içinde yok veya geçersiz.")

    logger.info("[META] feature columns loaded from models/model_meta_%s.json | n=%d", interval, len(feats))
    return feats


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
        if "label_thr" in meta and meta["label_thr"] is not None:
            return float(meta["label_thr"])
    except Exception:
        pass

    return 0.0


def _ensure_feature_schema(feat_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    - Meta'daki feature sırasını birebir uygular.
    - Eksik feature varsa 0.0 ile oluşturur (LSTM scaler mismatch çözümü).
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
def load_offline_klines(symbol: str, interval: str, limit: int = 20000) -> pd.DataFrame:
    """
    Offline cache'ten klines yükler:
      data/offline_cache/{symbol}_{interval}_6m.csv
    """
    path = Path("data/offline_cache") / f"{symbol}_{interval}_6m.csv"
    if not path.exists():
        raise FileNotFoundError(f"Offline cache yok: {path}")

    df = pd.read_csv(path)
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
    # i = window ... N-1
    for i in range(window, len(X)):
        xs.append(X[i - window : i])
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
    df_raw = load_offline_klines(symbol, interval, limit=limit_rows)

    # Feature engineering (offline_train_hybrid ile uyumlu)
    feat_df = build_features(df_raw)

    feature_cols = _load_meta_feature_cols(interval)
    X_df = _ensure_feature_schema(feat_df, feature_cols)

    # Label üret
    labels = build_labels(feat_df, horizon=horizon, thr=thr)

    # Son horizon bar label NaN olacağı için kes
    if horizon > 0:
        X_df = X_df.iloc[:-horizon].reset_index(drop=True)
        y_ser = labels.iloc[:-horizon]
    else:
        y_ser = labels

    y = pd.to_numeric(y_ser, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=np.float32)
    X = X_df.to_numpy(dtype=np.float32)

    logger.info("[DATA] After feature+label alignment: X shape=%s, y shape=%s | thr=%.8f", X.shape, y.shape, thr)

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
# Train + Save
# ---------------------------
def _save_scalers(models_dir: Path, interval: str, scaler: StandardScaler) -> None:
    """
    Projede farklı dönemlerde iki farklı isim kullanılmış:
      - lstm_scaler_{interval}.joblib
      - lstm_{interval}_scaler.joblib  (örn: lstm_5m_scaler.joblib)
    İkisini de yazarız -> loader hangi ismi ararsa bulsun.
    """
    p1 = models_dir / f"lstm_scaler_{interval}.joblib"
    p2 = models_dir / f"lstm_{interval}_scaler.joblib"
    joblib.dump(scaler, p1)
    joblib.dump(scaler, p2)
    logger.info("[SAVE] Scaler saved: %s (and alias %s)", p1, p2)


def _save_models(models_dir: Path, interval: str, model: tf.keras.Model) -> None:
    """
    Hybrid tarafında hem long hem short isimleri var.
    Aynı modeli iki isimle kaydediyoruz (uyumluluk).
    """
    long_path = models_dir / f"lstm_long_{interval}.h5"
    short_path = models_dir / f"lstm_short_{interval}.h5"
    model.save(long_path)
    model.save(short_path)
    logger.info("[SAVE] Models saved: %s and %s", long_path, short_path)


def main() -> None:
    # SYMBOL öncelik: env -> config -> fallback
    symbol = os.getenv("SYMBOL") or ("BTCUSDT" if _CFG is None else getattr(_CFG, "SYMBOL", "BTCUSDT"))

    # interval env üzerinden geliyor (sen döngü ile INTERVAL set ediyorsun)
    interval = os.getenv("INTERVAL", "5m")

    horizon = _env_int("LSTM_HORIZON", 1)
    window_size = _interval_seq_len(interval, default=50)

    # Label thr (build_labels zorunlu)
    thr = _load_label_thr(interval)

    # Train params (env ile uyumlu)
    batch_size = _env_int("LSTM_BATCH", 64)
    patience = _env_int("LSTM_PATIENCE", 5)
    val_split = _env_float("LSTM_VAL_SPLIT", 0.2)
    epochs = _env_int("LSTM_EPOCHS", 20)

    # Veri limiti (çok büyük olmasın)
    limit_rows = _env_int("LSTM_LIMIT_ROWS", 20000)

    logger.info(
        "[START] LSTM training | symbol=%s interval=%s horizon=%d window=%d thr=%.8f batch=%d epochs=%d val_split=%.3f",
        symbol, interval, horizon, window_size, thr, batch_size, epochs, val_split,
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
        logger.warning("[WARN] Çok az örnek var: n=%d (window=%d). Eğitim kalitesi düşük olabilir.", n_samples, window_size)

    # Split (time-series mantığıyla: son val_split kısmı valid)
    val_n = max(1, int(n_samples * float(val_split)))
    train_n = max(1, n_samples - val_n)

    X_train, X_val = X_seq[:train_n], X_seq[train_n:]
    y_train, y_val = y_seq[:train_n], y_seq[train_n:]

    logger.info("[SPLIT] train=%d val=%d", X_train.shape[0], X_val.shape[0])

    model = build_lstm_model(window=window_size, n_features=int(X_seq.shape[2]))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint (best auc)
    ckpt_path = models_dir / f"lstm_{interval}_best_tmp.h5"
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
        best_model = tf.keras.models.load_model(str(ckpt_path))
        ckpt_path.unlink(missing_ok=True)
    else:
        logger.info("[LOAD] Best checkpoint yok, son modeli kullanıyorum.")
        best_model = model

    _save_models(models_dir, interval, best_model)
    _save_scalers(models_dir, interval, scaler)

    logger.info("[DONE] LSTM training completed | interval=%s", interval)


if __name__ == "__main__":
    main()
