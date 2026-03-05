# /home/kasm920/binance1/wsfeeds/stream_handler.py

import json
import os
import time
from typing import Any, Dict, Optional

from core.logger import system_logger


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


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "")
    return str(v).strip() if v != "" else default


class _RateLimiter:
    """
    every_sec:
      - 0 veya negatif -> hiç izin verme (log tamamen kapalı)
      - pozitif -> en fazla every_sec'te 1 kez izin ver
    """

    def __init__(self, every_sec: float) -> None:
        self.every_sec = float(every_sec)
        self._last = 0.0

    def allow(self) -> bool:
        if self.every_sec <= 0:
            return False
        now = time.time()
        if (now - self._last) >= self.every_sec:
            self._last = now
            return True
        return False


# =========================================================
# Controls (env)
# =========================================================
# Tick (trade message) log spam kontrol:
#   WS_LOG_EVERY_SEC=0    -> tick log kapalı (default öneri)
#   WS_LOG_EVERY_SEC=5    -> 5 saniyede bir tick özeti logla
_WS_LOG_EVERY_SEC = _env_float("WS_LOG_EVERY_SEC", 0.0)

# Hangi seviyede yazsın: info|debug
_WS_LOG_LEVEL = _env_str("WS_LOG_LEVEL", "info").lower()

# "signal" alanı loglansın mı:
#   WS_SIGNAL_LOG=1 -> açık
#   WS_SIGNAL_LOG=0 -> kapalı
#
# Not: Binance trade stream normalde "signal" içermez.
_WS_SIGNAL_LOG = _env_bool("WS_SIGNAL_LOG", False)

# Parse/handler error log rate limit:
#   WS_ERR_LOG_EVERY_SEC=10 -> 10 saniyede bir hata logu (default)
_WS_ERR_LOG_EVERY_SEC = _env_float("WS_ERR_LOG_EVERY_SEC", 10.0)

_tick_limiter = _RateLimiter(_WS_LOG_EVERY_SEC)
_err_limiter = _RateLimiter(_WS_ERR_LOG_EVERY_SEC)


def _log_tick(symbol: str, price: float, qty: float) -> None:
    if not _tick_limiter.allow():
        return

    msg = f"[WebSocket] tick | symbol={symbol} price={price:.6f} qty={qty:.6f}"
    if _WS_LOG_LEVEL == "debug":
        system_logger.debug(msg)
    else:
        system_logger.info(msg)


def handle_message(msg: str) -> None:
    """
    Gelen websocket mesajını ayrıştırır.

    Binance Futures trade stream örnek mesaj:
      {
        "e": "trade",
        "E": 123456789,
        "s": "BTCUSDT",
        "t": 12345,
        "p": "71111.0",
        "q": "0.001",
        ...
      }

    NOT:
      - Binance trade stream normalde "signal" içermez.
        İleride kendi özel WS mesajların için kullanabilirsin.
    """
    try:
        data: Dict[str, Any] = json.loads(msg)

        symbol = str(data.get("s", "UNKNOWN"))

        # p/q bazen string, bazen boş/None olabilir -> güvenli parse
        try:
            price = float(data.get("p", 0.0) or 0.0)
        except Exception:
            price = 0.0

        try:
            qty = float(data.get("q", 0.0) or 0.0)
        except Exception:
            qty = 0.0

        # Tick log (rate-limited)
        _log_tick(symbol, price, qty)

        # Optional: external signal log
        signal: Optional[Any] = data.get("signal")
        if _WS_SIGNAL_LOG and signal and isinstance(signal, str):
            side = signal.upper()
            if side not in ("BUY", "SELL", "HOLD"):
                system_logger.warning(
                    "[WebSocket] Unknown signal value: %s (symbol=%s). Skipping.",
                    signal,
                    symbol,
                )
                return

            system_logger.info(
                "[WebSocket] external_signal | symbol=%s signal=%s qty=%.6f price=%.6f",
                symbol,
                side,
                qty,
                price,
            )

    except Exception as e:
        # Hata spamını kes (rate-limited)
        if _err_limiter.allow():
            system_logger.error("[WebSocket] Error handling message: %s", e, exc_info=True)
