import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

try:
    import websockets  # type: ignore
except Exception:  # pragma: no cover
    websockets = None

from monitoring.market_state import MarketState


class OKXWS:
    """
    OKX Public WebSocket (v5) - izleme amaçlı.
    Varsayılan: tickers channel.

    ENV:
      OKX_WS_ENABLE=1
      OKX_WS_URL=wss://ws.okx.com:8443/ws/v5/public
      OKX_WS_INST_ID=BTC-USDT
      OKX_WS_CHANNEL=tickers
    """

    def __init__(
        self,
        url: Optional[str] = None,
        inst_id: Optional[str] = None,
        channel: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.url = url or os.getenv("OKX_WS_URL", "wss://ws.okx.com:8443/ws/v5/public")
        self.inst_id = inst_id or os.getenv("OKX_WS_INST_ID", "BTC-USDT")
        self.channel = channel or os.getenv("OKX_WS_CHANNEL", "tickers")
        self.logger = logger or logging.getLogger("system")

        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def run_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._thread_main, name="okx-ws", daemon=True)
        self._thread.start()

    def _thread_main(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._runner())
        except Exception as e:
            self.logger.exception("[OKXWS] thread crashed: %s", e)

    async def _runner(self) -> None:
        if websockets is None:
            self.logger.warning("[OKXWS] websockets paketi yok. 'pip install websockets' gerekli.")
            return

        backoff = 1.0
        while not self._stop_flag.is_set():
            try:
                await self._connect_once()
                backoff = 1.0
            except Exception as e:
                self.logger.warning("[OKXWS] reconnect in %.1fs | err=%s", backoff, e)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.7, 30.0)

    async def _connect_once(self) -> None:
        self.logger.info("[OKXWS] connecting url=%s instId=%s channel=%s", self.url, self.inst_id, self.channel)

        async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
            sub = {
                "op": "subscribe",
                "args": [{"channel": self.channel, "instId": self.inst_id}],
            }
            await ws.send(json.dumps(sub))

            last_store_ts = 0.0
            store_every = float(os.getenv("OKX_STORE_EVERY_SEC", "0.0") or "0.0")  # 0 => her mesaj

            while not self._stop_flag.is_set():
                raw = await ws.recv()
                if raw is None:
                    continue

                # OKX bazen "pong" / event mesajları döndürebilir
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
                if store_every > 0 and (now - last_store_ts) < store_every:
                    continue

                # tickers: list[dict]
                d0 = data[0] if isinstance(data[0], dict) else None
                if not d0:
                    continue

                # Normalize minimal fields
                ticker = {
                    "instId": d0.get("instId"),
                    "last": d0.get("last"),
                    "bidPx": d0.get("bidPx"),
                    "askPx": d0.get("askPx"),
                    "ts": d0.get("ts"),
                    "raw": d0,
                }
                MarketState.set_okx_ticker(ticker=ticker, ts=now)
                last_store_ts = now
