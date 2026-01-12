from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from collections import deque
import time
import math

import numpy as np


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(lo)


def _hour_utc_from_ts(ts_sec: Optional[float]) -> Optional[int]:
    if ts_sec is None:
        return None
    try:
        # UTC hour
        return int(time.gmtime(float(ts_sec)).tm_hour)
    except Exception:
        return None


@dataclass
class _BookState:
    best_bid: float = 0.0
    best_ask: float = 0.0
    # store depth levels as (price, qty)
    bids: Optional[List[Tuple[float, float]]] = None
    asks: Optional[List[Tuple[float, float]]] = None
    ts_sec: Optional[float] = None


class MarketMetaBuilder:
    """
    Market meta üreticisi (tek modül):
      - spread_pct: (ask-bid)/mid
      - spread_z: spread_pct rolling z-score (shock ölçümü)
      - obi: order-book imbalance (topN qty imbalance)
      - liq_score: opsiyonel (basit proxy: depth toplamı)
      - hour_utc: time-of-day filter için

    Kullanım:
      builder = MarketMetaBuilder()
      builder.update_orderbook(symbol, tf, bids=[(p,q),...], asks=[(p,q),...], ts_sec=...)
      builder.update_best_bid_ask(symbol, tf, best_bid=..., best_ask=..., ts_sec=...)
      meta = builder.build_meta(symbol, tf)
    """

    def __init__(
        self,
        *,
        spread_window: int = 120,
        obi_levels: int = 10,
        min_mid: float = 1e-12,
    ) -> None:
        self.spread_window = int(max(10, spread_window))
        self.obi_levels = int(max(1, obi_levels))
        self.min_mid = float(min_mid)

        # state: (symbol, tf) -> bookstate
        self._book: Dict[Tuple[str, str], _BookState] = {}

        # spread history: (symbol, tf) -> deque[spread_pct]
        self._spread_hist: Dict[Tuple[str, str], deque] = {}

        # last meta cache: (symbol, tf) -> dict
        self._last_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _key(self, symbol: str, tf: str) -> Tuple[str, str]:
        return (str(symbol).upper(), str(tf).lower())

    # -------------------------
    # Updates (WS / REST tarafı burayı besler)
    # -------------------------
    def update_best_bid_ask(
        self,
        symbol: str,
        tf: str,
        *,
        best_bid: Any,
        best_ask: Any,
        ts_sec: Optional[float] = None,
    ) -> None:
        k = self._key(symbol, tf)
        st = self._book.get(k) or _BookState()
        st.best_bid = _safe_float(best_bid, 0.0)
        st.best_ask = _safe_float(best_ask, 0.0)
        st.ts_sec = ts_sec if ts_sec is not None else st.ts_sec
        self._book[k] = st

        # spread history güncelle
        self._push_spread(k, st.best_bid, st.best_ask)

    def update_orderbook(
        self,
        symbol: str,
        tf: str,
        *,
        bids: Optional[List[Tuple[Any, Any]]] = None,
        asks: Optional[List[Tuple[Any, Any]]] = None,
        best_bid: Any = None,
        best_ask: Any = None,
        ts_sec: Optional[float] = None,
    ) -> None:
        k = self._key(symbol, tf)
        st = self._book.get(k) or _BookState()

        if bids is not None:
            st.bids = [(float(_safe_float(p)), float(_safe_float(q))) for p, q in bids]
        if asks is not None:
            st.asks = [(float(_safe_float(p)), float(_safe_float(q))) for p, q in asks]

        if best_bid is not None:
            st.best_bid = _safe_float(best_bid, st.best_bid)
        elif st.bids:
            st.best_bid = float(st.bids[0][0])

        if best_ask is not None:
            st.best_ask = _safe_float(best_ask, st.best_ask)
        elif st.asks:
            st.best_ask = float(st.asks[0][0])

        st.ts_sec = ts_sec if ts_sec is not None else st.ts_sec
        self._book[k] = st

        self._push_spread(k, st.best_bid, st.best_ask)

    def _push_spread(self, k: Tuple[str, str], bid: float, ask: float) -> None:
        bid = float(bid)
        ask = float(ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        mid = (bid + ask) / 2.0
        if mid <= self.min_mid:
            return
        spread_pct = (ask - bid) / (mid + 1e-12)

        dq = self._spread_hist.get(k)
        if dq is None:
            dq = deque(maxlen=self.spread_window)
            self._spread_hist[k] = dq
        dq.append(float(spread_pct))

    # -------------------------
    # Feature builders
    # -------------------------
    def _compute_spread_meta(self, k: Tuple[str, str], bid: float, ask: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if bid <= 0 or ask <= 0 or ask < bid:
            out["spread_pct"] = 0.0
            out["spread_z"] = None
            return out

        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / (mid + 1e-12)
        out["spread_pct"] = float(max(0.0, spread_pct))

        dq = self._spread_hist.get(k)
        if dq is None or len(dq) < max(20, int(self.spread_window * 0.25)):
            out["spread_z"] = None
            return out

        arr = np.array(dq, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std()) if float(arr.std()) > 1e-12 else 1e-12
        z = (float(spread_pct) - mu) / sd
        out["spread_z"] = float(z)
        out["spread_mu"] = float(mu)
        out["spread_sd"] = float(sd)
        out["spread_n"] = int(len(dq))
        return out

    def _compute_obi_meta(self, bids: Optional[List[Tuple[float, float]]], asks: Optional[List[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        OBI (order-book imbalance):
          obi = (sum_bid_qty - sum_ask_qty) / (sum_bid_qty + sum_ask_qty)
          range: [-1..1]
        """
        out: Dict[str, Any] = {"obi": None, "liq_score": 0.0}
        if not bids or not asks:
            return out

        topn = int(self.obi_levels)
        bq = 0.0
        aq = 0.0
        for p, q in bids[:topn]:
            if q > 0:
                bq += float(q)
        for p, q in asks[:topn]:
            if q > 0:
                aq += float(q)

        denom = bq + aq
        if denom <= 1e-12:
            return out

        obi = (bq - aq) / denom
        out["obi"] = float(_clamp(obi, -1.0, 1.0))

        # liq_score: basit proxy (topN toplam qty)
        out["liq_score"] = float(max(0.0, denom))
        out["liq_bid_qty"] = float(bq)
        out["liq_ask_qty"] = float(aq)
        out["liq_topn"] = int(topn)
        return out

    # -------------------------
    # Public: build meta dict
    # -------------------------
    def build_meta(
        self,
        symbol: str,
        tf: str,
        *,
        fallback_ts_sec: Optional[float] = None,
        override_hour_utc: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Whale detector'a doğrudan verilecek market_meta dict'i üretir.
        """
        k = self._key(symbol, tf)
        st = self._book.get(k)

        if st is None:
            # hiçbir data yoksa cache döndür
            cached = self._last_meta.get(k)
            return dict(cached) if isinstance(cached, dict) else {}

        bid = float(st.best_bid or 0.0)
        ask = float(st.best_ask or 0.0)

        ts_sec = st.ts_sec if st.ts_sec is not None else fallback_ts_sec
        hour_utc = override_hour_utc if override_hour_utc is not None else _hour_utc_from_ts(ts_sec)

        meta: Dict[str, Any] = {}
        meta["hour_utc"] = hour_utc

        # spread
        meta.update(self._compute_spread_meta(k, bid, ask))

        # obi + liquidity
        meta.update(self._compute_obi_meta(st.bids, st.asks))

        # (ops) best bid/ask keep
        meta["best_bid"] = float(bid)
        meta["best_ask"] = float(ask)
        meta["ts_sec"] = float(ts_sec) if ts_sec is not None else None

        self._last_meta[k] = dict(meta)
        return meta

    def build_meta_by_tf(
        self,
        symbol: str,
        tfs: List[str],
        *,
        fallback_ts_sec: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for tf in tfs:
            out[str(tf)] = self.build_meta(symbol, tf, fallback_ts_sec=fallback_ts_sec)
        return out
