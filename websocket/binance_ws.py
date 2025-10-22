import websocket
import threading
from core.logger import system_logger
from .stream_handler import handle_message
from .reconnect_manager import reconnect_ws

BINANCE_WS_URL = "wss://fstream.binance.com/ws/btcusdt@trade"

def start_ws():
    ws = websocket.WebSocketApp(
        BINANCE_WS_URL,
        on_message=lambda ws, msg: handle_message(msg),
        on_error=lambda ws, err: system_logger.error(f"WebSocket error: {err}"),
        on_close=lambda ws: system_logger.info("WebSocket closed")
    )
    
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    system_logger.info("WebSocket started")
    
    return ws
