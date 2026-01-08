# models/model_registry.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import logging

from app_paths import MODELS_DIR
from models.hybrid_inference import HybridModel
from data.online_learning import OnlineLearner

class ModelRegistry:
    def __init__(self, model_dir: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
        self.model_dir = model_dir or MODELS_DIR
        self.logger = logger or logging.getLogger("system")

        self._hybrid_cache: Dict[str, HybridModel] = {}
        self._online_cache: Dict[Tuple[str, str, int], OnlineLearner] = {}  # (interval, base, n_classes)

    def get_hybrid(
        self,
        interval: str,
        model_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> HybridModel:
        itv = str(interval)
        if itv in self._hybrid_cache:
            return self._hybrid_cache[itv]

        mdir = model_dir or self.model_dir
        lg = logger or self.logger
        m = HybridModel(model_dir=mdir, interval=itv, logger=lg)
        self._hybrid_cache[itv] = m
        return m

    def get_online(
        self,
        interval: str,
        base_model_name: str = "online_model",
        model_dir: Optional[str] = None,
        n_classes: int = 2,
        load_existing: bool = True,   # ✅ EKLENDİ
    ) -> OnlineLearner:
        itv = str(interval)
        base = str(base_model_name)
        key = (itv, base, int(n_classes), bool(load_existing))  # ✅ cache key'e dahil et

        if key in self._online_cache:
            return self._online_cache[key]

        mdir = model_dir or self.model_dir

        ol = OnlineLearner(
            model_dir=mdir,
            base_model_name=base,
            interval=itv,
            n_classes=int(n_classes),
            # ✅ OnlineLearner ctor'unda load_existing yoksa,
            # burada sadece "passthrough" yapmayacağız.
            # (Aşağıdaki notu oku)
        )

        self._online_cache[key] = ol
        return ol
