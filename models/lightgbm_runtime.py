from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np

class LightGBMRuntime:
    def __init__(self, bundle_path: str):
        self.bundle_path = str(bundle_path)
        self.bundle = joblib.load(self.bundle_path)
        # bundle yapısı: {"model": trained_model, "schema": [...], "scaler": optional}
        self.model = self.bundle.get("model", self.bundle)
        self.schema = self.bundle.get("schema", None)
        self.scaler = self.bundle.get("scaler", None)

    def predict_proba_p1(self, X: np.ndarray) -> np.ndarray:
        X2 = X
        if self.scaler is not None:
            X2 = self.scaler.transform(X2)
        proba = self.model.predict_proba(X2)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        # bazı modeller tek kolon dönebilir
        return proba.reshape(-1).astype(float)
