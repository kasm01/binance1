# models/model_registry.py
import threading
from typing import Dict, Optional

from models.hybrid_inference import HybridModel

# models/model_registry.py
import logging
from typing import Dict, Optional

from app_paths import MODELS_DIR
from models.hybrid_inference import HybridModel


class ModelRegistry:
    def __init__(self, model_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.model_dir = model_dir or MODELS_DIR
        self.logger = logger or logging.getLogger("system")
        self._hybrid: Dict[str, HybridModel] = {}

    def get_hybrid(self, interval: str) -> HybridModel:
        interval = str(interval).strip()
        if interval in self._hybrid:
            return self._hybrid[interval]

        # ✅ senin HybridModel imzasına uygun
        m = HybridModel(model_dir=self.model_dir, interval=interval, logger=self.logger)
        self._hybrid[interval] = m
        return m

    def get_sgd_model(self, interval: str):
        # OnlineLearner için aynı SGD model objesini paylaşmak istersen:
        hm = self.get_hybrid(interval)
        # HybridModel içinde SGD modeli hangi attribute ise onu döndür:
        # örn: hm.sgd_model / hm.model / hm.clf vs.
        return getattr(hm, "sgd_model", None) or getattr(hm, "model", None)
