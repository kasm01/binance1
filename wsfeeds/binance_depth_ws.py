from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None  # type: ignore


logger = logging.getLogger("system")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


class BinanceDepthWS:
    """
    Binance Futures depth stream (default: depth5@100ms).
    - websockets ile çalışır (websocket-client ile isim çakışmasını önler).
    - last snapshot saklar
    - opsiyonel MarketMetaBuilder'a update_orderbook ile otomatik basar

    main.py:
        depth_ws = BinanceDepthWS(symbol=symbol, builder=market_meta_builder, tfs=list(mtf_intervals))
        depth_ws.run_background()
        snap = depth_ws.get_last_depth()
    """

    def __init__(
        self,
        *,
        symbol: str,
        url: Optional[str] = None,
        depth_stream: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
        builder: Any = None,
        tfs: Optional[List[str]] = None,
        store_every_ms: int = 50,
        reconnect_max_backoff_s: float = 30.0,
        ping_interval_s: float = 20.0,
        ping_timeout_s: float = 20.0,
    ) -> None:
        self.symbol = str(symbol).upper().strip()

        # fstream (USDT-M Futures)
        base = url or os.getenv("BINANCE_FUTURES_WS_URL", "wss://fstream.binance.com/ws")

        # depth5@100ms default
        stream = depth_stream or os.getenv("BINANCE_DEPTH_STREAM", f"{self.symbol.lower()}@depth5@100ms")
        self.ws_url = f"{base}/{stream}"

        self.logger = logger_ or logger

        self.builder = builder
        self.tfs = [str(x) for x in (tfs or [])]  # builder'a hangi TF'lere basılacak
        self.store_every_ms = int(max(0, store_every_ms))

        self.reconnect_max_backoff_s = float(max(1.0, reconnect_max_backoff_s))
        self.ping_interval_s = float(max(5.0, ping_interval_s))
        self.ping_timeout_s = float(max(5.0, ping_timeout_s))

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._lock = threading.Lock()
        self._last_depth: Optional[Dict[str, Any]] = None
        self._last_store_ts: float = 0.0

    # --------------------------
    # Public
    # --------------------------
    def run_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        if websockets is None:
            self.logger.warning("[DEPTHWS] websockets paketi yok. Kur: pip install websockets")
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._thread_main, name="binance-depth-ws", daemon=True)
        self._thread.start()
        self.logger.info("[DEPTHWS] thread started url=%s", self.ws_url)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_flag.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)
        except Exception:
            pass

    def get_last_depth(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_depth) if isinstance(self._last_depth, dict) else None

    # --------------------------
    # Internal
    # --------------------------
    def _thread_main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._runner())
        except Exception as e:
            self.logger.exception("[DEPTHWS] thread crashed: %s", e)

    async def _runner(self) -> None:
        backoff = 1.0
        while not self._stop_flag.is_set():
            try:
                await self._connect_once()
                backoff = 1.0
            except Exception as e:
                if self._stop_flag.is_set():
                    break
                self.logger.warning("[DEPTHWS] reconnect in %.1fs | err=%s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.7, self.reconnect_max_backoff_s)

    async def _connect_once(self) -> None:
        self.logger.info("[DEPTHWS] connecting %s", self.ws_url)

        async with websockets.connect(
            self.ws_url,
            ping_interval=self.ping_interval_s,
            ping_timeout=self.ping_timeout_s,
        ) as ws:
            while not self._stop_flag.is_set():
                raw = await ws.recv()
                if raw is None:
                    continue

                if isinstance(raw, (bytes, bytearray)):
                    try:
                        raw = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        continue

                s = str(raw).strip()
                if not s:
                    continue

                try:
                    msg = json.loads(s)
                except Exception:
                    continue

                if not isinstance(msg, dict):
                    continue

                # depth stream payload:
                # bids: msg["b"] = [["price","qty"],...]
                # asks: msg["a"] = [["price","qty"],...]
                bids_raw = msg.get("b")
                asks_raw = msg.get("a")
                if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                    continue

                now = time.time()
                if self.store_every_ms > 0:
                    if (now - self._last_store_ts) < (self.store_every_ms / 1000.0):
                        continue

                bids: List[Tuple[float, float]] = []
                asks: List[Tuple[float, float]] = []

                for x in bids_raw:
                    if isinstance(x, (list, tuple)) and len(x) >= 2:
                        p = _safe_float(x[0], 0.0)
                        q = _safe_float(x[1], 0.0)
                        if p > 0 and q >= 0:
                            bids.append((p, q))

                for x in asks_raw:
                    if isinstance(x, (list, tuple)) and len(x) >= 2:
                        p = _safe_float(x[0], 0.0)
                        q = _safe_float(x[1], 0.0)
                        if p > 0 and q >= 0:
                            asks.append((p, q))

                # sort best first (Binance genelde zaten sıralı gelir ama garanti edelim)
                bids.sort(key=lambda t: t[0], reverse=True)
                asks.sort(key=lambda t: t[0], reverse=False)

                snap: Dict[str, Any] = {
                    "ts": now,
                    "symbol": self.symbol,
                    "bids": bids,
                    "asks": asks,
                    "raw": None,
                }

                with self._lock:
                    self._last_depth = snap
                    self._last_store_ts = now

                # Builder'a otomatik bas
                if self.builder is not None:
                    try:
                        # TF listesi verildiyse hepsine güncelle; yoksa "5m" default gibi davran
                        tfs = self.tfs or ["5m"]
                        for tf in tfs:
                            # builder.update_orderbook(symbol, tf, bids=..., asks=..., ts_sec=...)
                            if hasattr(self.builder, "update_orderbook"):
                                self.builder.update_orderbook(
                                    self.symbol,
                                    str(tf),
                                    bids=bids,
                                    asks=asks,
                                    ts_sec=now,
                                )
                    except Exception:
                        pass
