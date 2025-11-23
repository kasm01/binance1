import logging
from typing import Any

from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Model tabanlı sinyal üreten basit strateji motoru.

    Şu an:
      - EnsembleModel.predict(features) çağırıyor.
      - Hata durumunda 'HOLD' döner.
    """

    def __init__(self, model: EnsembleModel | None = None) -> None:
        self.model = model or EnsembleModel()

    def generate_signal(self, features: Any) -> str:
        """
        :param features: Modelin beklediği feature vektörü / dataframe satırı
        :return: 'BUY' / 'SELL' / 'HOLD' (model implementasyonuna göre)
        """
        try:
            signal = self.model.predict(features)
            logger.info(f"[StrategyEngine] Generated signal from model: {signal}")
            return signal
        except Exception as e:
            logger.error(f"[StrategyEngine] Error generating signal: {e}", exc_info=True)
            return "HOLD"
