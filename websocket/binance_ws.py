# websocket/binance_ws.py

import threading
import websocket  # websocket-client paketi
from typing import Optional, Callable

from core.logger import system_logger
from config.settings import Settings
from .stream_handler import handle_message

# Binance Futures trade stream URL template
BINANCE_WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol_lower}@trade"


def _build_ws_url(symbol: Optional[str] = None) -> str:
    """
    SYMBOL env yoksa BTCUSDT kullan.
    """
    sym = (symbol or Settings.SYMBOL or "BTCUSDT").upper()
    return BINANCE_WS_URL_TEMPLATE.format(symbol_lower=sym.lower())


def _on_message(ws, msg: str) -> None:
    """
    Her yeni mesaj geldiğinde çağrılan callback.
    """
    handle_message(msg)


def _on_error(ws, err) -> None:
    system_logger.error(f"[WebSocket] Error: {err}")


def _on_close(ws, close_status_code, close_msg) -> None:
    system_logger.info(
        f"[WebSocket] Closed | code={close_status_code}, msg={close_msg}"
    )


def _on_open(ws) -> None:
    system_logger.info("[WebSocket] Connection opened.")


def create_ws_app(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    """
    Harici yerlerden (reconnect_manager) tekrar WebSocketApp üretmek için
    kullanılan fabrika fonksiyon.
    """
    url = _build_ws_url(symbol)
    system_logger.info(f"[WebSocket] Connecting to: {url}")

    ws_app = websocket.WebSocketApp(
        url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    return ws_app


def start_ws_in_thread(
    symbol: Optional[str] = None,
) -> websocket.WebSocketApp:
    """
    WebSocket bağlantısını ayrı bir daemon thread üzerinde başlatır.
    Cloud Run gibi ortamlarda ana thread'i bloklamaz.

    Dönüş değeri: WebSocketApp instance (reconnect_manager bunu takip edebilir).
    """
    ws_app = create_ws_app(symbol)

    wst = threading.Thread(target=ws_app.run_forever, daemon=True)
    wst.start()

    system_logger.info("[WebSocket] WebSocket thread started.")
    return ws_app

