# trading/strategy_engine.py

import logging
from typing import Any, Optional

from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Model tabanlı sinyal üreten basit strateji motoru.

    Beklenen:
      - model.predict(features) -> p (0..1) dönebilir (EnsembleModel şu an float döndürüyor)
      - p >= 0.60 -> BUY
      - p <= 0.40 -> SELL
      - aksi -> HOLD
    """

    def __init__(self, model: Optional[EnsembleModel] = None) -> None:
        self.model = model or EnsembleModel()

    def generate_signal(self, features: Any) -> str:
        try:
            p = float(self.model.predict(features))  # 0..1

            if p >= 0.60:
                signal = "BUY"
            elif p <= 0.40:
                signal = "SELL"
            else:
                signal = "HOLD"

            logger.info("[StrategyEngine] p=%.4f -> signal=%s", p, signal)
            return signal

        except Exception as e:
            logger.error("[StrategyEngine] Error generating signal: %s", e, exc_info=True)
            return "HOLD"
