# models/lstm_model.py
from __future__ import annotations

from typing import Tuple
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def _to_fixed_seq(X2d: np.ndarray, seq_len: int) -> np.ndarray:
    """
    X2d: (T, F)
    returns: (1, seq_len, F)
    - T >= seq_len: tail al
    - T <  seq_len: başa edge-pad ile tamamla
    """
    X2d = np.asarray(X2d)
    if X2d.ndim != 2:
        raise ValueError(f"_to_fixed_seq expects 2D, got ndim={X2d.ndim}")

    T, F = X2d.shape
    if T >= seq_len:
        Xw = X2d[-seq_len:, :]
    else:
        pad = seq_len - T
        # edge pad: ilk satırı tekrar et (0 pad yerine daha stabil)
        Xw = np.vstack([np.repeat(X2d[:1, :], pad, axis=0), X2d])

    return Xw[None, :, :]  # (1, seq_len, F)


def _ensure_model_window(X: np.ndarray, seq_len: int, n_feat: int) -> np.ndarray:
    """
    X:
      - (T,F) -> (1,seq_len,F)
      - (B,T,F) -> (B,seq_len,F)  (her batch için tail/pad)
    """
    X = np.asarray(X)

    if X.ndim == 2:
        # (T,F)
        if X.shape[1] != n_feat:
            raise ValueError(f"LSTM feature mismatch: got {X.shape[1]} expected {n_feat}")
        return _to_fixed_seq(X, seq_len)

    if X.ndim == 3:
        # (B,T,F)
        if X.shape[2] != n_feat:
            raise ValueError(f"LSTM feature mismatch: got {X.shape[2]} expected {n_feat}")

        B, T, F = X.shape
        if T == seq_len:
            return X.astype(np.float32, copy=False)

        # batch-wise fix
        out = np.zeros((B, seq_len, F), dtype=np.float32)
        for i in range(B):
            out[i] = _to_fixed_seq(X[i], seq_len)[0]
        return out

    raise ValueError(f"LSTM predict: unsupported X.ndim={X.ndim}")


class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int]):
        """
        input_shape: (timesteps, features)
        """
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def fit(self, X, y, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0,
        )

    def _predict_p1(self, X) -> np.ndarray:
        """
        X'i model.input_shape'a göre düzeltip p1 döndürür.
        model.input_shape: (None, seq_len, n_feat)
        """
        seq_len = int(self.model.input_shape[1])
        n_feat = int(self.model.input_shape[2])

        Xseq = _ensure_model_window(X, seq_len=seq_len, n_feat=n_feat)
        proba = self.model.predict(Xseq, verbose=0).reshape(-1)
        return np.clip(proba, 0.0, 1.0)

    def predict(self, X):
        p1 = self._predict_p1(X)
        return (p1 > 0.5).astype(int)

    def predict_proba(self, X):
        p1 = self._predict_p1(X)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
