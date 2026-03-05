# /home/kasm920/binance1/wsfeeds/binance_ws.py
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import websocket  # websocket-client
from core.logger import system_logger
from config.settings import Settings
from .stream_handler import handle_message

BINANCE_WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol_lower}@trade"


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "")
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, "")
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _build_ws_url(symbol: Optional[str] = None) -> str:
    sym = (symbol or getattr(Settings, "SYMBOL", None) or "BTCUSDT").upper()
    return BINANCE_WS_URL_TEMPLATE.format(symbol_lower=sym.lower())


class _WsSampler:
    """
    Per-message log spamını engellemek için örnekleme (sampling).
    WS_LOG_EVERY_SEC=0 => kapalı
    WS_LOG_EVERY_SEC=5 => 5 saniyede 1 log
    """

    def __init__(self) -> None:
        self.every_sec = _env_float("WS_LOG_EVERY_SEC", 0.0)
        self._last_ts = 0.0

    def should_log(self) -> bool:
        if self.every_sec <= 0:
            return False
        now = time.time()
        if (now - self._last_ts) >= self.every_sec:
            self._last_ts = now
            return True
        return False


def create_ws_app(symbol: Optional[str] = None) -> websocket.WebSocketApp:
    url = _build_ws_url(symbol)
    sampler = _WsSampler()

    if _env_bool("WS_DEBUG_IMPORT", False):
        system_logger.info(
            "[WS-DEBUG] websocket=%s WebSocketApp=%s",
            getattr(websocket, "__file__", str(websocket)),
            hasattr(websocket, "WebSocketApp"),
        )

    system_logger.info("[WebSocket] Connecting to: %s", url)

    def _on_open(ws) -> None:
        system_logger.info("[WebSocket] Connection opened.")

    def _on_message(ws, msg: str) -> None:
        # İstersen bu modül üzerinden “özet” log bas (spam yapmadan).
        if sampler.should_log():
            try:
                data = json.loads(msg)
                # Binance trade event: { "s": "BTCUSDT", "p": "71111.0", "q": "0.001", ... }
                sym = data.get("s") or "?"
                price = data.get("p") or data.get("price") or "?"
                qty = data.get("q") or data.get("qty") or "?"
                system_logger.info(
                    "[WebSocket] sample | symbol=%s price=%s qty=%s",
                    sym,
                    price,
                    qty,
                )
            except Exception:
                # JSON değilse sessiz geç
                pass

        # Asıl işleme
        handle_message(msg)

    def _on_error(ws, err) -> None:
        system_logger.error("[WebSocket] Error: %s", err)

    def _on_close(ws, close_status_code, close_msg) -> None:
        system_logger.warning("[WebSocket] Closed | code=%s msg=%s", close_status_code, close_msg)

    return websocket.WebSocketApp(
        url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )


def start_ws_in_thread(symbol: Optional[str] = None) -> tuple[websocket.WebSocketApp, threading.Thread]:
    ws_app = create_ws_app(symbol)

    ping_interval = _env_int("WS_PING_INTERVAL", 20)
    ping_timeout = _env_int("WS_PING_TIMEOUT", 10)

    reconnect_sec = _env_float("WS_RECONNECT_SEC", 5.0)
    max_reconnect_sec = _env_float("WS_MAX_RECONNECT_SEC", 60.0)

    def _run():
        backoff = max(1.0, reconnect_sec)
        while True:
            try:
                # run_forever bloklar; bağlantı kapanınca döner.
                ws_app.run_forever(ping_interval=ping_interval, ping_timeout=ping_timeout)
            except Exception as e:
                system_logger.exception("[WebSocket] run_forever crashed: %s", e)

            # Buraya geldiysek bağlantı kapandı veya hata aldı.
            system_logger.warning("[WebSocket] reconnecting in %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(max_reconnect_sec, backoff * 1.5)

    t = threading.Thread(target=_run, daemon=True, name="binance-ws")
    t.start()
    system_logger.info("[WebSocket] WebSocket thread started.")
    return ws_app, t


class BinanceWS:
    """
    main.py tarafında kolay kullanım için wrapper.
    """

    def __init__(self, symbol: Optional[str] = None) -> None:
        self.symbol = symbol
        self.ws_app: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None

    def run_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.ws_app, self._thread = start_ws_in_thread(self.symbol)

    def stop(self, timeout: float = 5.0) -> None:
        try:
            if self.ws_app is not None:
                self.ws_app.close()
        except Exception as e:
            system_logger.warning("[WebSocket] close failed: %s", e)

        try:
            if self._thread is not None:
                self._thread.join(timeout=timeout)
        except Exception:
            pass
