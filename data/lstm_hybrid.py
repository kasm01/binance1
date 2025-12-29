from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def _build_lstm_dataset(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N, F), y: (N,)
    returns:
      X_seq: (N - seq_len, seq_len, F)
      y_seq: (N - seq_len,)
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    N, F = X.shape
    if N <= seq_len:
        raise ValueError(f"Too few samples for seq_len={seq_len}: N={N}")

    X_seq = np.zeros((N - seq_len, seq_len, F), dtype=np.float32)
    y_seq = np.zeros((N - seq_len,), dtype=np.float32)

    for i in range(seq_len, N):
        X_seq[i - seq_len] = X[i - seq_len : i]
        y_seq[i - seq_len] = y[i]

    return X_seq, y_seq


def _build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    input_shape: (seq_len, n_features)
    """
    seq_len, n_features = input_shape

    inp = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    m = models.Model(inputs=inp, outputs=out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return m


def _prepare_X(features_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, list]:
    """
    - open_time/close_time/ignore drop
    - yalnızca numerik kolonlar
    - scaler fit + transform
    """
    df = features_df.copy()
    df = df.drop(columns=[c for c in ("open_time", "close_time", "ignore") if c in df.columns], errors="ignore")
    df = df.select_dtypes(include=["float", "int"])
    used_cols = list(df.columns)

    X = df.values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, scaler, used_cols


def train_lstm_hybrid(
    features_df: pd.DataFrame,
    y_long: pd.Series,
    y_short: pd.Series,
    interval: str,
    model_dir: str = "models",
    seq_len: int = 50,        # ✅ senin model beklentin 50, default 50
    test_size: float = 0.2,
    max_epochs: int = 20,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Kaydeder:
      models/lstm_{interval}_long.h5
      models/lstm_{interval}_short.h5
      models/lstm_{interval}_scaler.joblib
    """
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path_long = out_dir / f"lstm_{interval}_long.h5"
    model_path_short = out_dir / f"lstm_{interval}_short.h5"
    scaler_path = out_dir / f"lstm_{interval}_scaler.joblib"

    X_scaled, scaler, _used_cols = _prepare_X(features_df)
    joblib.dump(scaler, scaler_path)

    es = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=0,
    )

    # ---- LONG ----
    yL = y_long.astype(int).values
    Xtr, Xva, ytr, yva = train_test_split(X_scaled, yL, test_size=test_size, shuffle=False)

    Xtr_seq, ytr_seq = _build_lstm_dataset(Xtr, ytr, seq_len=seq_len)
    Xva_seq, yva_seq = _build_lstm_dataset(Xva, yva, seq_len=seq_len)

    m_long = _build_lstm_model((seq_len, Xtr_seq.shape[-1]))
    m_long.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    pva = m_long.predict(Xva_seq, verbose=0).ravel()
    try:
        auc_long = float(roc_auc_score(yva_seq, pva))
    except ValueError:
        auc_long = 0.5

    m_long.save(model_path_long)

    # ---- SHORT ----
    yS = y_short.astype(int).values
    Xtr, Xva, ytr, yva = train_test_split(X_scaled, yS, test_size=test_size, shuffle=False)

    Xtr_seq, ytr_seq = _build_lstm_dataset(Xtr, ytr, seq_len=seq_len)
    Xva_seq, yva_seq = _build_lstm_dataset(Xva, yva, seq_len=seq_len)

    m_short = _build_lstm_model((seq_len, Xtr_seq.shape[-1]))
    m_short.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xva_seq, yva_seq),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    pva = m_short.predict(Xva_seq, verbose=0).ravel()
    try:
        auc_short = float(roc_auc_score(yva_seq, pva))
    except ValueError:
        auc_short = 0.5

    m_short.save(model_path_short)

    return {"lstm_long_auc": auc_long, "lstm_short_auc": auc_short, "seq_len": float(seq_len)}


def load_lstm_bundle(interval: str, model_dir: str = "models"):
    """
    Runtime için: scaler, model_long, model_short
    """
    base = Path(model_dir)
    scaler_path = base / f"lstm_{interval}_scaler.joblib"
    long_path = base / f"lstm_{interval}_long.h5"
    short_path = base / f"lstm_{interval}_short.h5"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")
    if not long_path.exists():
        raise FileNotFoundError(f"Long LSTM modeli bulunamadı: {long_path}")
    if not short_path.exists():
        raise FileNotFoundError(f"Short LSTM modeli bulunamadı: {short_path}")

    scaler = joblib.load(scaler_path)
    model_long = tf.keras.models.load_model(long_path)
    model_short = tf.keras.models.load_model(short_path)
    return scaler, model_long, model_short


def load_lstm_for_inference(interval: str, model_dir: str = "models", logger=None, **kwargs):
    """
    HybridModel uyumlu wrapper.
    """
    if logger:
        logger.info("[LSTM] loading bundle interval=%s model_dir=%s", interval, model_dir)
    return load_lstm_bundle(interval=interval, model_dir=model_dir)
