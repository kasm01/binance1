import logging
from typing import List, Dict, Any, Optional

from trading.trade_executor import execute_trade

logger = logging.getLogger(__name__)


class MultiTradeEngine:
    """
    Tek seferde birden çok işlemi yürüten basit motor.
    Örn:
      trades = [
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.001},
        {"symbol": "ETHUSDT", "side": "SELL", "qty": 0.01},
      ]
    """

    def __init__(self, max_trades: int = 3) -> None:
        self.max_trades = max_trades

    def execute_trades(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[Optional[Dict[str, Any]]]:
        """
        :param trades: [{'symbol':'BTCUSDT', 'side':'BUY', 'qty':0.01}, ...]
        :return: her trade denemesi için order sonucu listesi
        """
        results: List[Optional[Dict[str, Any]]] = []

        for i, trade in enumerate(trades):
            if i >= self.max_trades:
                logger.info(
                    f"[MultiTradeEngine] Max trades ({self.max_trades}) reached, remaining trades skipped."
                )
                break

            symbol = trade.get("symbol")
            side = trade.get("side")
            qty = trade.get("qty")

            if not symbol or not side or qty is None:
                logger.warning(f"[MultiTradeEngine] Invalid trade dict: {trade}")
                results.append(None)
                continue

            logger.info(
                f"[MultiTradeEngine] Executing trade {i+1}: "
                f"{symbol} {side} {qty}"
            )
            order = execute_trade(symbol, side, qty)
            results.append(order)

        return results
