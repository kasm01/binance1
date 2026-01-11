from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None

# MarketState opsiyonel: yoksa modül yine çalışsın
try:
    from monitoring.market_state import MarketState  # type: ignore
except Exception:  # pragma: no cover
    MarketState = None  # type: ignore


logger = logging.getLogger("system")


@dataclass
class OKXTick:
    ts: float
    instId: str
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    exch_ts: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class OKXWS:
    """
    OKX Public WebSocket (v5) - ticker-only izleme.

    Özellikler:
    - Background thread içinde çalışır (daemon)
    - Otomatik reconnect + backoff
    - Thread-safe last_tick saklar (get_last_tick())
    - İsteğe bağlı MarketState.set_okx_ticker(...) günceller

    ENV:
      OKX_WS_ENABLE=1
      OKX_WS_URL=wss://ws.okx.com:8443/ws/v5/public
      OKX_WS_INST_ID=BTC-USDT
      OKX_WS_CHANNEL=tickers
      OKX_WS_PING_INTERVAL_SEC=20
      OKX_WS_PING_TIMEOUT_SEC=20
      OKX_WS_RECONNECT_MAX_BACKOFF=30
      OKX_STORE_EVERY_SEC=0         # 0 => her mesaj, >0 => throttling
    """

    def __init__(
        self,
        url: Optional[str] = None,
        inst_id: Optional[str] = None,
        channel: Optional[str] = None,
        logger_: Optional[logging.Logger] = None,
        store_every_sec: Optional[float] = None,
        ping_interval_s: Optional[float] = None,
        ping_timeout_s: Optional[float] = None,
        reconnect_max_backoff_s: Optional[float] = None,
    ) -> None:
        self.url = url or os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")
        self.inst_id = inst_id or os.getenv("OKX_WS_INST_ID", "BTC-USDT")
        self.channel = channel or os.getenv("OKX_WS_CHANNEL", "tickers")

        self.logger = logger_ or logger

        self.store_every_sec = float(
            store_every_sec if store_every_sec is not None else (os.getenv("OKX_STORE_EVERY_SEC", "0") or "0")
        )

        self.ping_interval_s = float(
            ping_interval_s if ping_interval_s is not None else (os.getenv("OKX_WS_PING_INTERVAL_SEC", "20") or "20")
        )
        self.ping_timeout_s = float(
            ping_timeout_s if ping_timeout_s is not None else (os.getenv("OKX_WS_PING_TIMEOUT_SEC", "20") or "20")
        )

        self.reconnect_max_backoff_s = float(
            reconnect_max_backoff_s if reconnect_max_backoff_s is not None else (os.getenv("OKX_WS_RECONNECT_MAX_BACKOFF", "30") or "30")
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._lock = threading.Lock()
        self._last_tick: Optional[OKXTick] = None

        # shutdown cleanup için
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None  # websockets client protocol

    # --------------------------
    # Public API
    # --------------------------
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop flag set + aktif websocket varsa kapat + thread join.
        """
        self._stop_flag.set()

        # ws.close() coroutine olduğundan, loop varsa thread-safe schedule edelim
        loop = self._loop
        ws = self._ws
        if loop is not None and ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(ws.close(), loop)  # type: ignore
                try:
                    fut.result(timeout=timeout)
                except Exception:
                    pass
            except Exception:
                pass

        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

        self.logger.info("[OKXWS] stop requested.")

    def run_background(self) -> None:
        """
        Background thread başlatır. Zaten çalışıyorsa no-op.
        """
        if self._thread and self._thread.is_alive():
            return

        if websockets is None:
            self.logger.warning("[OKXWS] websockets paketi yok. 'pip install websockets' gerekli.")
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._thread_main, name="okx-ws", daemon=True)
        self._thread.start()
        self.logger.info("[OKXWS] background thread started. instId=%s channel=%s", self.inst_id, self.channel)

    def get_last_tick(self) -> Optional[Dict[str, Any]]:
        """
        Thread-safe son tick'i dict olarak döner (Telegram snapshot için ideal).
        """
        with self._lock:
            return asdict(self._last_tick) if self._last_tick else None

    # --------------------------
    # Internal
    # --------------------------
    def _set_last_tick(self, tick: OKXTick) -> None:
        with self._lock:
            self._last_tick = tick

    def _thread_main(self) -> None:
        loop = None
        try:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._runner())
        except Exception as e:
            self.logger.exception("[OKXWS] thread crashed: %s", e)
        finally:
            try:
                if loop is not None:
                    loop.stop()
                    loop.close()
            except Exception:
                pass
            self._loop = None
            self._ws = None

    async def _runner(self) -> None:
        backoff = 1.0
        while not self._stop_flag.is_set():
            try:
                await self._connect_once()
                backoff = 1.0
            except Exception as e:
                if self._stop_flag.is_set():
                    break
                self.logger.warning("[OKXWS] reconnect in %.1fs | err=%s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.7, self.reconnect_max_backoff_s)

    async def _connect_once(self) -> None:
        self.logger.info("[OKXWS] connecting url=%s instId=%s channel=%s", self.url, self.inst_id, self.channel)

        async with websockets.connect(self.url, ping_interval=self.ping_interval_s, ping_timeout=self.ping_timeout_s) as ws:
            self._ws = ws

            sub = {"op": "subscribe", "args": [{"channel": self.channel, "instId": self.inst_id}]}
            await ws.send(json.dumps(sub))

            last_store_ts = 0.0

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

                # OKX bazen "pong" döner
                if s.lower() == "pong":
                    continue

                try:
                    msg = json.loads(s)
                except Exception:
                    continue

                # event/error
                if isinstance(msg, dict) and msg.get("event"):
                    ev = msg.get("event")
                    if ev == "subscribe":
                        self.logger.info("[OKXWS] subscribed ok: %s", msg.get("arg"))
                    elif ev == "error":
                        self.logger.warning("[OKXWS] error event: %s", msg)
                    continue

                # data
                if not isinstance(msg, dict):
                    continue
                data = msg.get("data")
                if not isinstance(data, list) or not data:
                    continue

                now = time.time()
                if self.store_every_sec > 0 and (now - last_store_ts) < self.store_every_sec:
                    continue

                d0 = data[0] if isinstance(data[0], dict) else None
                if not d0:
                    continue

                def _f(x: Any) -> Optional[float]:
                    try:
                        return float(x)
                    except Exception:
                        return None

                inst = str(d0.get("instId") or self.inst_id)

                tick = OKXTick(
                    ts=now,
                    instId=inst,
                    last=_f(d0.get("last")),
                    bid=_f(d0.get("bidPx")),
                    ask=_f(d0.get("askPx")),
                    exch_ts=d0.get("ts"),
                    raw=d0,
                )

                # 1) thread-safe last_tick (Telegram snapshot / status için)
                self._set_last_tick(tick)

                # 2) opsiyonel MarketState store
                if MarketState is not None:
                    try:
                        MarketState.set_okx_ticker(
                            ticker={
                                "instId": tick.instId,
                                "last": d0.get("last"),
                                "bidPx": d0.get("bidPx"),
                                "askPx": d0.get("askPx"),
                                "ts": d0.get("ts"),
                                "raw": d0,
                            },
                            ts=now,
                        )
                    except Exception:
                        pass

                last_store_ts = now

        self._ws = None
