# models/model_registry.py
import threading
from typing import Dict, Optional

from models.hybrid_inference import HybridModel

class ModelRegistry:
    """
    Interval bazlı HybridModel cache.
    Aynı interval için diskten bir kez yükler, sonra aynı instance'ı döner.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._hybrid: Dict[str, HybridModel] = {}

    def get_hybrid(self, interval: str) -> HybridModel:
        with self._lock:
            if interval in self._hybrid:
                return self._hybrid[interval]
            m = HybridModel(interval=interval)  # <-- senin mevcut constructor diskten yüklüyor
            self._hybrid[interval] = m
            return m

    def get_sgd_model(self, interval: str):
        # OnlineLearner için aynı SGD model objesini paylaşmak istersen:
        hm = self.get_hybrid(interval)
        # HybridModel içinde SGD modeli hangi attribute ise onu döndür:
        # örn: hm.sgd_model / hm.model / hm.clf vs.
        return getattr(hm, "sgd_model", None) or getattr(hm, "model", None)
