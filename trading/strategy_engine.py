# trading/strategy_engine.py

import logging
from typing import Any, Optional

from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Model tabanlı sinyal üreten basit strateji motoru.
    """

    def __init__(self, model: Optional[EnsembleModel] = None) -> None:
        self.model = model or EnsembleModel()

    def generate_signal(self, features: Any) -> str:
        """
        :param features: Modelin beklediği feature vektörü / dataframe satırı
        :return: 'BUY' / 'SELL' / 'HOLD'
        """
        try:
            # Modelin 0..1 arası p_buy verdiğini varsayıyoruz
            p = self.model.predict(features)

            # predict bazen array dönebilir
            if hasattr(p, "__len__"):
                p_val = float(p[0])
            else:
                p_val = float(p)

            if p_val >= 0.60:
                signal = "BUY"
            elif p_val <= 0.40:
                signal = "SELL"
            else:
                signal = "HOLD"

            logger.info("[StrategyEngine] p=%.4f -> signal=%s", p_val, signal)
            return signal

        except Exception as e:
            logger.error("[StrategyEngine] Error generating signal: %s", e, exc_info=True)
            return "HOLD"

