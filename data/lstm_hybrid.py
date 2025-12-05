from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import joblib

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def _build_lstm_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N, F), y: (N,)
    seq_len: kaç bar'lık pencere
    Dönen:
      X_seq: (N - seq_len + 1, seq_len, F)
      y_seq: (N - seq_len + 1,)
    """
    N, F = X.shape
    if N <= seq_len:
        raise ValueError(f"Too few samples for seq_len={seq_len}: N={N}")

    X_seq = []
    y_seq = []
    for i in range(seq_len, N):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def _build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    input_shape: (seq_len, n_features)
    Basit ama güçlü bir LSTM binary classifier.
    """
    seq_len, n_features = input_shape

    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    return model


def train_lstm_hybrid(
    features_df: pd.DataFrame,
    y_long: pd.Series,
    y_short: pd.Series,
    interval: str,
    model_dir: str = "models",
    seq_len: int = 32,
    test_size: float = 0.2,
    random_state: int = 42,
    max_epochs: int = 20,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    SGD ile aynı feature'lar üzerinden LSTM eğitir.
    - features_df: clean_df (numerik kolonlar)
    - y_long, y_short: 0/1 seriler
    Kayıtlar:
      models/lstm_{interval}_long.h5
      models/lstm_{interval}_short.h5
      models/lstm_{interval}_scaler.joblib
    Dönen meta:
      {
        "lstm_long_auc": float,
        "lstm_short_auc": float,
        "seq_len": int
      }
    """
    model_path_long = Path(model_dir) / f"lstm_{interval}_long.h5"
    model_path_short = Path(model_dir) / f"lstm_{interval}_short.h5"
    scaler_path = Path(model_dir) / f"lstm_{interval}_scaler.joblib"

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Feature'ları hazırlama
    # -------------------------
    # open_time / close_time / ignore gibi kolonları at
    drop_cols = [c for c in ["open_time", "close_time", "ignore"] if c in features_df.columns]
    X_df = features_df.drop(columns=drop_cols, errors="ignore")

    # sadece numerik
    X_df = X_df.select_dtypes(include=["float", "int"])

    X = X_df.values.astype(np.float32)

    # scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Kaydet
    joblib.dump(scaler, scaler_path)

    # -------------------------
    # 2) Long LSTM
    # -------------------------
    y_long_arr = y_long.astype(int).values
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y_long_arr,
        test_size=test_size,
        shuffle=False,
    )

    X_train_seq, y_train_seq = _build_lstm_dataset(X_train, y_train, seq_len=seq_len)
    X_val_seq, y_val_seq = _build_lstm_dataset(X_val, y_val, seq_len=seq_len)

    model_long = _build_lstm_model((seq_len, X_train_seq.shape[-1]))

    es = callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    model_long.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    y_val_proba_long = model_long.predict(X_val_seq, verbose=0).ravel()
    try:
        long_auc = float(roc_auc_score(y_val_seq, y_val_proba_long))
    except ValueError:
        long_auc = 0.5

    model_long.save(model_path_long)

    # -------------------------
    # 3) Short LSTM
    # -------------------------
    y_short_arr = y_short.astype(int).values
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_scaled,
        y_short_arr,
        test_size=test_size,
        shuffle=False,
    )

    X_train_seq_s, y_train_seq_s = _build_lstm_dataset(X_train_s, y_train_s, seq_len=seq_len)
    X_val_seq_s, y_val_seq_s = _build_lstm_dataset(X_val_s, y_val_s, seq_len=seq_len)

    model_short = _build_lstm_model((seq_len, X_train_seq_s.shape[-1]))

    model_short.fit(
        X_train_seq_s,
        y_train_seq_s,
        validation_data=(X_val_seq_s, y_val_seq_s),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    y_val_proba_short = model_short.predict(X_val_seq_s, verbose=0).ravel()
    try:
        short_auc = float(roc_auc_score(y_val_seq_s, y_val_proba_short))
    except ValueError:
        short_auc = 0.5

    model_short.save(model_path_short)

    return {
        "lstm_long_auc": long_auc,
        "lstm_short_auc": short_auc,
        "seq_len": seq_len,
    }


def load_lstm_bundle(interval: str, model_dir: str = "models"):
    """
    HybridModel tarafından runtime'da kullanılmak üzere:
    - scaler
    - long lstm
    - short lstm
    döndürür.
    """
    model_path_long = Path(model_dir) / f"lstm_{interval}_long.h5"
    model_path_short = Path(model_dir) / f"lstm_{interval}_short.h5"
    scaler_path = Path(model_dir) / f"lstm_{interval}_scaler.joblib"

    if not (model_path_long.exists() and model_path_short.exists() and scaler_path.exists()):
        raise FileNotFoundError(f"LSTM bundle not found for interval={interval}")

    scaler = joblib.load(scaler_path)
    model_long = tf.keras.models.load_model(model_path_long)
    model_short = tf.keras.models.load_model(model_path_short)

    return scaler, model_long, model_short


def load_lstm_for_inference(interval: str, model_dir=None, logger=None, **kwargs):
    """
    Online inference için LSTM scaler + modellerini yükler.
    HybridModel bu fonksiyonu model_dir ve logger keyword arg'larıyla çağırdığı için
    imza buna göre tanımlandı. logger şimdilik sadece opsiyonel log için tutuluyor.
    """

    """
    Online inference için LSTM scaler + modellerini yükler.
    HybridModel bu fonksiyonu model_dir keyword arg ile çağırdığı için
    imza buna göre tanımlandı.
    """
    from pathlib import Path
    import joblib
    from keras.models import load_model

    base_dir = Path(model_dir) if model_dir is not None else Path("models")

    scaler_path = base_dir / f"lstm_{interval}_scaler.joblib"
    long_path = base_dir / f"lstm_{interval}_long.h5"
    short_path = base_dir / f"lstm_{interval}_short.h5"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")
    if not long_path.exists():
        raise FileNotFoundError(f"Long LSTM modeli bulunamadı: {long_path}")
    if not short_path.exists():
        raise FileNotFoundError(f"Short LSTM modeli bulunamadı: {short_path}")

    scaler = joblib.load(scaler_path)
    model_long = load_model(long_path)
    model_short = load_model(short_path)
    return scaler, model_long, model_short


    model_dir = Path("models")
    scaler_path = model_dir / f"lstm_{interval}_scaler.joblib"
    long_path = model_dir / f"lstm_{interval}_long.h5"
    short_path = model_dir / f"lstm_{interval}_short.h5"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler bulunamadı: {scaler_path}")
    if not long_path.exists():
        raise FileNotFoundError(f"Long LSTM modeli bulunamadı: {long_path}")
    if not short_path.exists():
        raise FileNotFoundError(f"Short LSTM modeli bulunamadı: {short_path}")

    scaler = joblib.load(scaler_path)
    model_long = load_model(long_path)
    model_short = load_model(short_path)
    return scaler, model_long, model_short
