# models/ensemble_model.py

from __future__ import annotations
import os
import joblib
import numpy as np

# Modeller
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from models.fallback_model import FallbackModel


class EnsembleModel:
    """
    LightGBM + CatBoost + LSTM + Fallback model karışımı.
    Eksik model varsa fallback devreye girer.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir

        # Ana modeller
        self.lightgbm_model = None
        self.catboost_model = None
        self.lstm_model = None

        # Fallback model (daima mevcut)
        self.fallback_model = FallbackModel()

        self.load_models()

    # ------------------------------------------------------------
    # MODELLERİ YÜKLE
    # ------------------------------------------------------------
    def load_models(self):
        """Her model varsa yüklenir, yoksa atlanır."""

        # LightGBM
        try:
            self.lightgbm_model = LightGBMModel(os.path.join(self.model_dir, "lightgbm_model.pkl"))
        except Exception:
            self.lightgbm_model = None

        # CatBoost
        try:
            self.catboost_model = CatBoostModel(os.path.join(self.model_dir, "catboost_model.cbm"))
        except Exception:
            self.catboost_model = None

        # LSTM
       # try:
        #    self.lstm_model = LSTMModel(os.path.join(self.model_dir, "lstm_model.keras"))
        #except Exception:
         #   self.lstm_model = None

    # ------------------------------------------------------------
    # ANA TAHMİN
    # ------------------------------------------------------------
    def predict_proba(self, X: np.ndarray):
        """
        Modellerin ortalaması alınır.
        Hiçbiri çalışmazsa fallback model devreye girer.
        """
        predictions = []

        if self.lightgbm_model:
            try:
                predictions.append(self.lightgbm_model.predict_proba(X))
            except Exception:
                pass

        if self.catboost_model:
            try:
                predictions.append(self.catboost_model.predict_proba(X))
            except Exception:
                pass

        if self.lstm_model:
            try:
                predictions.append(self.lstm_model.predict_proba(X))
            except Exception:
                pass

        # Eğer hiç model tahmini yoksa -> fallback
        if not predictions:
            return self.fallback_model.predict_proba(X)

        # Ağırlıksız ortalama
        return float(np.mean(predictions))

    # ------------------------------------------------------------
    # TEK SATIR TAHMİN (p_buy)
    # ------------------------------------------------------------
    def predict(self, X_row):
        return self.predict_proba(np.array([X_row]))

