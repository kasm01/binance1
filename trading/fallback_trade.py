from trading.trade_executor import execute_trade
from core.logger import system_logger

def fallback_trade(symbol, side, qty, retries=3):
    for attempt in range(retries):
        order = execute_trade(symbol, side, qty)
        if order:
            return order
        system_logger.warning(f"Retry {attempt+1} for {symbol} {side}")
    system_logger.error(f"All retries failed for {symbol} {side}")
    return None
