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
            reconnect_max_backoff_s
            if reconnect_max_backoff_s is not None
            else (os.getenv("OKX_WS_RECONNECT_MAX_BACKOFF", "30") or "30")
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self._lock = threading.Lock()
        self._last_tick: Optional[OKXTick] = None

    # --------------------------
    # Public API
    # --------------------------
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop flag set + thread join best-effort.
        """
        self._stop_flag.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=timeout)
        except Exception:
            pass

    def run_background(self) -> None:
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
        with self._lock:
            return asdict(self._last_tick) if self._last_tick else None

    # --------------------------
    # Internal
    # --------------------------
    def _set_last_tick(self, tick: OKXTick) -> None:
        with self._lock:
            self._last_tick = tick

    def _thread_main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._runner())
        except Exception as e:
            self.logger.exception("[OKXWS] thread crashed: %s", e)

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

                if s.lower() == "pong":
                    continue

                try:
                    msg = json.loads(s)
                except Exception:
                    continue

                if isinstance(msg, dict) and msg.get("event"):
                    ev = msg.get("event")
                    if ev == "subscribe":
                        self.logger.info("[OKXWS] subscribed ok: %s", msg.get("arg"))
                    elif ev == "error":
                        self.logger.warning("[OKXWS] error event: %s", msg)
                    continue

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

                self._set_last_tick(tick)

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

