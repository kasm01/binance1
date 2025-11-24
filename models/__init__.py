# models/__init__.py

from .lstm_model import LSTMModel
from .lightgbm_model import LightGBMModel
from .ensemble_model import EnsembleModel
from .fallback_model import FallbackModel

# CatBoostModel opsiyonel: catboost paketi yoksa import hatasına düşmeyelim
try:
    from .catboost_model import CatBoostModel  # type: ignore
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostModel = None  # type: ignore
    CATBOOST_AVAILABLE = False

__all__ = [
    "LSTMModel",
    "LightGBMModel",
    "EnsembleModel",
    "FallbackModel",
    "CatBoostModel",
    "CATBOOST_AVAILABLE",
]

