from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PriceSnapshot:
    symbol: str
    bid: float
    ask: float
    mid: float
    ts: float  # time.time()


class PriceCache:
    """
    Thread-safe basit cache.
    - WS thread'leri set() yapar
    - main loop / executor get_mid() okur
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, PriceSnapshot] = {}

    def set_bid_ask(self, symbol: str, bid: float, ask: float, ts: Optional[float] = None) -> None:
        if bid <= 0 or ask <= 0:
            return

        if ask < bid:
            # bazen anlık terslik olabilir, düzeltelim
            bid, ask = min(bid, ask), max(bid, ask)

        now = time.time() if ts is None else float(ts)
        mid = (bid + ask) / 2.0

        with self._lock:
            self._data[symbol.upper()] = PriceSnapshot(
                symbol=symbol.upper(),
                bid=float(bid),
                ask=float(ask),
                mid=float(mid),
                ts=now,
            )

    def get(self, symbol: str) -> Optional[PriceSnapshot]:
        with self._lock:
            return self._data.get(symbol.upper())

    def get_mid(self, symbol: str, max_age_sec: float = 2.0) -> Optional[float]:
        snap = self.get(symbol)
        if not snap:
            return None

        if max_age_sec > 0 and (time.time() - snap.ts) > max_age_sec:
            return None

        return snap.mid
