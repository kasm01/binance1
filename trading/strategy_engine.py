# trading/strategy_engine.py

import logging
from typing import Any, Optional

from models.ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Model tabanlı sinyal üreten basit strateji motoru.

    Şu an:
      - EnsembleModel.predict(features) çağırıyor.
      - Hata durumunda 'HOLD' döner.
    """

    def __init__(self, model: Optional[EnsembleModel] = None) -> None:
        # Parametre gelmezse kendi içinde EnsembleModel oluşturur
        self.model = model or EnsembleModel()

    def generate_signal(self, features: Any) -> str:
        """
        :param features: Modelin beklediği feature vektörü / dataframe satırı
        :return: 'BUY' / 'SELL' / 'HOLD'
        """
        try:
            y_pred = self.model.predict(features)
            # y_pred 0/1 ise buradan sinyal haritalama yapabiliriz
            if hasattr(y_pred, "__len__"):
                pred = int(y_pred[0])
            else:
                pred = int(y_pred)

            signal = "BUY" if pred == 1 else "SELL"
            logger.info(
                f"[StrategyEngine] Generated signal from EnsembleModel: {signal}"
            )
            return signal
        except Exception as e:
            logger.error(
                f"[StrategyEngine] Error generating signal: {e}",
                exc_info=True,
            )
            return "HOLD"

