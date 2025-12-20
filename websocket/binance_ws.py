# websocket/binance_ws.py

from __future__ import annotations

import threading
from typing import Optional

import websocket  # websocket-client paketi

from core.logger import system_logger
from config.settings import Settings
from .stream_handler import handle_message

BINANCE_WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol_lower}@trade"


def _build_ws_url(symbol: Optional[str] = None) -> str:
    sym = (symbol or getattr(Settings, "SYMBOL", None) or "BTCUSDT").upper()
    return BINANCE_WS_URL_TEMPLATE.format(symbol_lower=sym.lower())


def _on_message(ws, msg: str) -> None:
    handle_message(msg)


def _on_error(ws, err) -> None:
    system_logger.error("[WebSocket] Error: %s", err)


def _on_close(ws, close_status_code, close_msg) -> None:
    system_logger.info("[WebSocket] Closed | code=%s msg=%s", close_status_code, close_msg)


def _on_open(ws) -> None:
    system_logger.info("[WebSocket] Connection opened.")


def create_ws_app(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    url = _build_ws_url(symbol)
    system_logger.info("[WebSocket] Connecting to: %s", url)

    return websocket.WebSocketApp(
        url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )


def start_ws_in_thread(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    ws_app = create_ws_app(symbol)

    t = threading.Thread(target=ws_app.run_forever, daemon=True)
    t.start()

    system_logger.info("[WebSocket] WebSocket thread started.")
    return ws_app


class BinanceWS:
    """
    main.py tarafında kolay kullanım için wrapper.

    Kullanım:
        ws = BinanceWS(symbol="BTCUSDT")
        ws.run_background()
    """

    def __init__(self, symbol: Optional[str] = None) -> None:
        self.symbol = symbol
        self.ws_app: Optional[websocket.WebSocketApp] = None

    def run_background(self) -> None:
        self.ws_app = start_ws_in_thread(self.symbol)
