# websocket/binance_depth_ws.py
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import websocket  # websocket-client

from core.logger import system_logger
from config.settings import Settings

BINANCE_FUTURES_WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol_lower}@depth5@100ms"


def _build_depth_ws_url(symbol: Optional[str] = None) -> str:
    sym = (symbol or getattr(Settings, "SYMBOL", None) or "BTCUSDT").upper()
    return BINANCE_FUTURES_WS_URL_TEMPLATE.format(symbol_lower=sym.lower())


@dataclass
class DepthSnapshot:
    ts: float
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    raw: Optional[Dict[str, Any]] = None


class BinanceDepthWS:
    """
    Binance Futures depth5@100ms WS wrapper.

    - best bid/ask + top5 levels burada var.
    - Order-book imbalance / spread / liquidity proxy üretmek için ideal.

    Kullanım:
        dws = BinanceDepthWS(symbol="BTCUSDT")
        dws.run_background()
        snap = dws.get_last_depth()
        ...
        dws.stop()
    """

    def __init__(self, symbol: Optional[str] = None) -> None:
        self.symbol = (symbol or getattr(Settings, "SYMBOL", None) or "BTCUSDT").upper()

        self.ws_app: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._last: Optional[DepthSnapshot] = None

    def run_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        url = _build_depth_ws_url(self.symbol)
        system_logger.info("[DEPTHWS] Connecting to: %s", url)

        self.ws_app = websocket.WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        def _run():
            # ping_interval/timeout ile bağlantı daha stabil olur
            self.ws_app.run_forever(ping_interval=20, ping_timeout=10)

        self._thread = threading.Thread(target=_run, daemon=True, name=f"binance-depth-ws-{self.symbol.lower()}")
        self._thread.start()
        system_logger.info("[DEPTHWS] thread started. symbol=%s", self.symbol)

    def stop(self, timeout: float = 5.0) -> None:
        try:
            if self.ws_app is not None:
                self.ws_app.close()
        except Exception as e:
            system_logger.warning("[DEPTHWS] close failed: %s", e)

        try:
            if self._thread is not None:
                self._thread.join(timeout=timeout)
        except Exception:
            pass

    def get_last_depth(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return asdict(self._last) if self._last else None

    # ---------------- handlers ----------------
    def _on_open(self, ws) -> None:
        system_logger.info("[DEPTHWS] Connection opened. symbol=%s", self.symbol)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        system_logger.info("[DEPTHWS] Closed | code=%s msg=%s symbol=%s", close_status_code, close_msg, self.symbol)

    def _on_error(self, ws, err) -> None:
        system_logger.error("[DEPTHWS] Error: %s | symbol=%s", err, self.symbol)

    def _on_message(self, ws, msg: str) -> None:
        # msg format: {"e":"depthUpdate","E":...,"b":[["price","qty"],...],"a":[...]}
        try:
            s = msg.strip()
            if not s:
                return
            j = json.loads(s)
            if not isinstance(j, dict):
                return

            bids_raw = j.get("b") or []
            asks_raw = j.get("a") or []
            if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                return

            def _parse_levels(x: list) -> List[Tuple[float, float]]:
                out: List[Tuple[float, float]] = []
                for it in x:
                    if not isinstance(it, (list, tuple)) or len(it) < 2:
                        continue
                    try:
                        p = float(it[0])
                        q = float(it[1])
                        out.append((p, q))
                    except Exception:
                        continue
                return out

            bids = _parse_levels(bids_raw)
            asks = _parse_levels(asks_raw)

            # Binance depth lists are best-first (bids desc, asks asc) by stream spec
            # yine de boşsa geç
            if not bids or not asks:
                return

            import time
            snap = DepthSnapshot(
                ts=time.time(),
                symbol=self.symbol,
                bids=bids[:5],
                asks=asks[:5],
                raw=j,
            )
            with self._lock:
                self._last = snap

        except Exception:
            # spam log yapmayalım
            return
