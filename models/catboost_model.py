# models/catboost_model.py
import os
from typing import Optional

from core.logger import system_logger

# catboost paketi opsiyonel: lokalde yoksa import patlamasın
try:
    from catboost import CatBoostClassifier  # type: ignore
    CATBOOST_AVAILABLE = True
except ImportError:
    CatBoostClassifier = None  # type: ignore
    CATBOOST_AVAILABLE = False
    system_logger.warning(
        "[CatBoostModel] 'catboost' paketi yüklü değil. "
        "Lokal ortamda CatBoostModel kullanılamayacak."
    )


class CatBoostModel:
    """
    CatBoost tabanlı model wrapper'ı.

    - Cloud Run / prod ortamında: catboost kurulu, model normal çalışır.
    - Lokal ortamda (Python 3.12, catboost kurulmamış): import başarısız olmaz,
      sadece bu sınıfı kullanmaya çalışırsan RuntimeError fırlatır.
    """

    def __init__(self, model_path: Optional[str] = None):
        if not CATBOOST_AVAILABLE:
            # Import aşamasında patlatmıyoruz; sadece gerçekten kullanmaya kalkınca hata veriyoruz.
            raise RuntimeError(
                "CatBoostModel kullanılamıyor çünkü 'catboost' paketi yüklü değil. "
                "Lütfen sadece LSTM / LightGBM / Fallback modellerini kullanın "
                "veya Python 3.11 ortamında catboost kurulmuş şekilde çalıştırın."
            )

        self.model = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
        )

        # Eğitimli model dosyası verilmişse yükle
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_model(model_path)
                system_logger.info(f"[CatBoostModel] Model yüklendi: {model_path}")
            except Exception as e:
                system_logger.error(f"[CatBoostModel] Model yüklenemedi: {e}")

    def fit(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoostModel.fit çağrıldı ama 'catboost' paketi yok.")

        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoostModel.predict_proba çağrıldı ama 'catboost' paketi yok.")

        return self.model.predict_proba(X)

    def predict(self, X):
        """
        Binary sınıflandırma için proba[1] > 0.5 ise 1, değilse 0 döner.
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

