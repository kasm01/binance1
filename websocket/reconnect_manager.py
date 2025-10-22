import time
from core.logger import system_logger
from .binance_ws import start_ws

def reconnect_ws(ws, retry_interval=5):
    """
    Websocket bağlantısı kapanırsa yeniden bağlanır.
    """
    while True:
        if not ws.sock or not ws.sock.connected:
            system_logger.warning("WebSocket disconnected. Reconnecting...")
            try:
                ws = start_ws()
            except Exception as e:
                system_logger.error(f"Reconnect failed: {e}")
        time.sleep(retry_interval)
