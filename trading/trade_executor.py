from binance.client import Client
from config.credentials import API_KEY, API_SECRET
from core.logger import system_logger

client = Client(API_KEY, API_SECRET)

def execute_trade(symbol, side, qty):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=qty
        )
        system_logger.info(f"Executed trade: {order}")
        return order
    except Exception as e:
        system_logger.error(f"Trade execution failed: {e}")
        return None
