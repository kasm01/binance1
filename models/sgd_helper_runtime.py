from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import joblib

@dataclass
class SGDHelperRuntime:
    bundle_path: str
    model: Any = None
    scaler: Any = None
    cols: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        b = joblib.load(self.bundle_path)
        # bundle dict bekliyoruz; ama eski format gelirse fall back yap
        if isinstance(b, dict) and "model" in b and "scaler" in b:
            self.model = b["model"]
            self.scaler = b["scaler"]
            self.cols = b.get("cols")
            self.meta = b.get("meta")
        else:
            # fallback: doğrudan estimator gelmiş olabilir
            self.model = b
            self.scaler = None
            self.cols = None
            self.meta = None

    def predict_proba_p1(self, X: np.ndarray) -> np.ndarray:
        Xn = np.asarray(X, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(1, -1)
        if self.scaler is not None:
            Xn = self.scaler.transform(Xn)
        proba = self.model.predict_proba(Xn)
        return np.asarray(proba)[:, 1]
