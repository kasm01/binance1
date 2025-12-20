# websocket/binance_ws.py

from __future__ import annotations

import threading
import time
import json
from typing import Optional

import websocket  # websocket-client

from core.logger import system_logger
from config.settings import Settings

# stream_handler.py içinde şunu bekliyoruz:
# def handle_message(raw_msg: str) -> None: ...
from .stream_handler import handle_message

# Binance Futures trade stream URL template
BINANCE_WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol_lower}@trade"


def _build_ws_url(symbol: Optional[str] = None) -> str:
    """
    SYMBOL verilmezse Settings.SYMBOL -> env -> default BTCUSDT kullanır.
    """
    sym = (symbol or getattr(Settings, "SYMBOL", None) or "BTCUSDT").upper()
    return BINANCE_WS_URL_TEMPLATE.format(symbol_lower=sym.lower())


def _safe_log_json_snippet(msg: str, max_len: int = 180) -> str:
    """
    Log şişmesin diye gelen mesajın kısa bir özetini döndürür.
    """
    s = msg.strip()
    if len(s) > max_len:
        s = s[:max_len] + "..."
    # JSON ise minify dene
    try:
        obj = json.loads(msg)
        s2 = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        if len(s2) > max_len:
            s2 = s2[:max_len] + "..."
        return s2
    except Exception:
        return s


def _on_message(ws, msg: str) -> None:
    """
    Her yeni mesaj geldiğinde çağrılan callback.
    """
    try:
        handle_message(msg)
    except Exception as e:
        # message parse/handler hataları WS'yi düşürmesin
        try:
            system_logger.error(
                "[WebSocket] handle_message error: %s | msg=%s",
                e,
                _safe_log_json_snippet(msg),
            )
        except Exception:
            pass


def _on_error(ws, err) -> None:
    try:
        system_logger.error("[WebSocket] Error: %s", err)
    except Exception:
        pass


def _on_close(ws, close_status_code, close_msg) -> None:
    try:
        system_logger.info(
            "[WebSocket] Closed | code=%s, msg=%s", close_status_code, close_msg
        )
    except Exception:
        pass


def _on_open(ws) -> None:
    try:
        system_logger.info("[WebSocket] Connection opened.")
    except Exception:
        pass


def create_ws_app(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    """
    Harici yerlerden (reconnect_manager) tekrar WebSocketApp üretmek için
    kullanılan fabrika fonksiyon.
    """
    url = _build_ws_url(symbol)

    try:
        system_logger.info("[WebSocket] Connecting to: %s", url)
    except Exception:
        pass

    ws_app = websocket.WebSocketApp(
        url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    return ws_app


def start_ws_in_thread(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    """
    WebSocket bağlantısını ayrı bir daemon thread üzerinde başlatır.
    Cloud Run gibi ortamlarda ana thread'i bloklamaz.

    Dönüş: WebSocketApp instance (istersen reconnect_manager bunu takip eder)
    """
    ws_app = create_ws_app(symbol)

    # run_forever kwargs
    ping_interval = int(getattr(Settings, "WS_PING_INTERVAL", 20))
    ping_timeout = int(getattr(Settings, "WS_PING_TIMEOUT", 10))

    def _runner() -> None:
        """
        İç thread runner.
        Not: reconnect için istersen bu fonksiyonun içine basit loop ekleyebilirsin.
        Şimdilik tek sefer run_forever.
        """
        try:
            ws_app.run_forever(
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
            )
        except Exception as e:
            try:
                system_logger.error("[WebSocket] run_forever crashed: %s", e)
            except Exception:
                pass

        # run_forever döndüyse (kapanmışsa) bilgi ver
        try:
            system_logger.info("[WebSocket] run_forever exited.")
        except Exception:
            pass

    wst = threading.Thread(target=_runner, daemon=True, name="binance-ws")
    wst.start()

    try:
        system_logger.info(
            "[WebSocket] WebSocket thread started | symbol=%s ping_interval=%s ping_timeout=%s",
            (symbol or getattr(Settings, "SYMBOL", "BTCUSDT")),
            ping_interval,
            ping_timeout,
        )
    except Exception:
        pass

    return ws_app

