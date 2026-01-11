from __future__ import annotations

import threading
import time
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


def start_ws_in_thread(symbol: Optional[str] = None) -> tuple[websocket.WebSocketApp, threading.Thread]:
    ws_app = create_ws_app(symbol)

    # ping_interval/ping_timeout, reconnect vb. run_forever paramlarıyla yönetilebilir
    t = threading.Thread(
        target=lambda: ws_app.run_forever(ping_interval=20, ping_timeout=20),
        daemon=True,
        name="binance-ws",
    )
    t.start()

    system_logger.info("[WebSocket] WebSocket thread started.")
    return ws_app, t


class BinanceWS:
    """
    main.py tarafında kolay kullanım için wrapper.

    Kullanım:
        ws = BinanceWS(symbol="BTCUSDT")
        ws.run_background()
        ...
        ws.stop()
    """

    def __init__(self, symbol: Optional[str] = None) -> None:
        self.symbol = symbol
        self.ws_app: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def run_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_flag.clear()
        self.ws_app, self._thread = start_ws_in_thread(self.symbol)

    def stop(self, timeout: float = 5.0) -> None:
        """
        WS'i kapatır ve thread'i best-effort join eder.
        """
        self._stop_flag.set()

        if self.ws_app is not None:
            try:
                # websocket-client: close() thread-safe çağrılabilir
                self.ws_app.close()
            except Exception as e:
                system_logger.debug("[WebSocket] close() failed: %s", e)

        # thread join (daemon zaten ama clean çıkış için iyi)
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

        system_logger.info("[WebSocket] stop requested.")
