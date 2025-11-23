# models/lstm_model.py
from typing import Tuple, Optional

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int]):
        """
        :param input_shape: (timesteps, features)
        """
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(
        self,
        X,
        y,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ):
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0,
        )

    def predict(self, X):
        # 0/1 sınıf etiketi döndür
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        """
        Sklearn benzeri interface: [p_class0, p_class1]
        """
        p1 = self.model.predict(X, verbose=0).flatten()
        p1 = np.clip(p1, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

