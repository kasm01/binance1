import logging
import numpy as np

logger = logging.getLogger(__name__)


class FallbackModel:
    """
    ML modelleri hata verirse veya hazır değilse devreye giren basit yedek model.
    Interface:
      - predict_proba(X)
      - predict(X)
    """

    def __init__(self, default_proba: float = 0.5):
        self.default_proba = float(default_proba)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        try:
            if X is None:
                n = 1
            else:
                n = X.shape[0] if hasattr(X, "shape") else len(X)

            p_buy = np.clip(self.default_proba, 0.0, 1.0)
            p_sell = 1.0 - p_buy

            probs = np.tile(np.array([[p_sell, p_buy]]), (n, 1))
            return probs
        except Exception as e:
            logger.exception(f"[FallbackModel] predict_proba hatası: {e}")
            return np.tile(np.array([[0.5, 0.5]]), (1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        p_buy = probs[:, 1]
        return (p_buy > 0.5).astype(int)
