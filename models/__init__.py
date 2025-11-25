# models/__init__.py

"""
Model package exports.

ÖNEMLİ:
- Cloud Run başlangıcında TensorFlow gibi ağır bağımlılıkları yüklememek için
  burada LSTMModel'i import ETMİYORUZ.
- LSTM'e ihtiyacın olursa doğrudan:
    from models.lstm_model import LSTMModel
  şeklinde import edebilirsin.

Bu sayede:
- Container start süresi kısalır
- TensorFlow kaynaklı olası crashler Cloud Run'da devre dışı kalır.
"""

from .lightgbm_model import LightGBMModel
from .ensemble_model import EnsembleModel
from .fallback_model import FallbackModel

# CatBoost opsiyonel: catboost paketi yoksa RuntimeError fırlatıyoruz,
# ama bu zaten Cloud Run'da kullanılmıyor.
try:
    from .catboost_model import CatBoostModel  # type: ignore
    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostModel = None  # type: ignore
    CATBOOST_AVAILABLE = False

__all__ = [
    "LightGBMModel",
    "EnsembleModel",
    "FallbackModel",
    "CatBoostModel",
    "CATBOOST_AVAILABLE",
]

