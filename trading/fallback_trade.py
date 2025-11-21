import logging
from typing import Optional, Dict, Any

from trading.trade_executor import execute_trade

logger = logging.getLogger(__name__)


def fallback_trade(
    symbol: str,
    side: str,
    qty: float,
    retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Basit fallback trade fonksiyonu:
      - execute_trade başarısız olursa 'retries' kadar yeniden dener.
    """
    for attempt in range(retries):
        order = execute_trade(symbol, side, qty)
        if order:
            logger.info(
                f"[fallback_trade] Order success on attempt {attempt + 1} "
                f"for {symbol} {side} {qty}"
            )
            return order

        logger.warning(
            f"[fallback_trade] Retry {attempt + 1}/{retries} failed for {symbol} {side}"
        )

    logger.error(f"[fallback_trade] All retries failed for {symbol} {side}")
    return None

