# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
        return s
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, tuple):
        return [str(t) for t in list(x)]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return [str(x)]
@dataclass
class SignalEvent:
    """
    Worker -> Aggregator standart event formatı.

    Notlar:
      - Worker "trade aç" kararı vermez. Sadece aday sinyal üretir.
      - side_candidate: "long" | "short" | "none"
      - score_edge: model edge / alpha (0..1)
      - confidence: model confidence (0..1)
      - atr_pct: örn 0.012 (%1.2)
      - spread_pct: örn 0.0003 (%0.03)
      - liq_score: 0..1
      - whale_score: 0..1 (opsiyonel)
      - meta: debug/ek alanlar (p_used, fast_model_score, micro_score, vb.)
    """
    ts_utc: str
    symbol: str
    interval: str
    source: str

    side_candidate: str = "none"
    score_edge: float = 0.0
    confidence: float = 0.0

    atr_pct: float = 0.0
    spread_pct: float = 0.0
    liq_score: float = 0.0

    whale_score: float = 0.0
    whale_dir: str = "none"

    dedup_key: str = ""

    meta: Dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> "SignalEvent":
        self.ts_utc = _safe_str(self.ts_utc, _now_utc_iso()) or _now_utc_iso()
        self.symbol = _safe_str(self.symbol, "").upper()
        self.interval = _safe_str(self.interval, "5m")
        self.source = _safe_str(self.source, "w?")

        sc = _safe_str(self.side_candidate, "none").strip().lower()
        if sc in ("buy", "long"):
            sc = "long"
        elif sc in ("sell", "short"):
            sc = "short"
        elif sc in ("", "hold", "none", "flat", "neutral"):
            sc = "none"
        self.side_candidate = sc

        self.score_edge = float(_clamp(_safe_float(self.score_edge, 0.0), 0.0, 1.0))
        self.confidence = float(_clamp(_safe_float(self.confidence, 0.0), 0.0, 1.0))

        self.atr_pct = float(_clamp(_safe_float(self.atr_pct, 0.0), 0.0, 1.0))
        self.spread_pct = float(_clamp(_safe_float(self.spread_pct, 0.0), 0.0, 1.0))
        self.liq_score = float(_clamp(_safe_float(self.liq_score, 0.0), 0.0, 1.0))

        self.whale_score = float(_clamp(_safe_float(self.whale_score, 0.0), 0.0, 1.0))
        wd = _safe_str(self.whale_dir, "none").strip().lower()
        if wd in ("buy", "long", "in", "inflow"):
            wd = "long"
        elif wd in ("sell", "short", "out", "outflow"):
            wd = "short"
        elif wd in ("", "none", "neutral"):
            wd = "none"
        self.whale_dir = wd

        # dedup_key otomatik üret
        if not self.dedup_key:
            self.dedup_key = self.make_dedup_key(self.symbol, self.interval, self.side_candidate)

        if not isinstance(self.meta, dict):
            self.meta = {}

        return self

    @staticmethod
    def make_dedup_key(symbol: str, interval: str, side_candidate: str) -> str:
        sym = _safe_str(symbol, "").upper()
        itv = _safe_str(interval, "")
        side = _safe_str(side_candidate, "none").lower()
        return f"{sym}|{itv}|{side}"
    def to_dict(self) -> Dict[str, Any]:
        self.normalize()
        return {
            "ts_utc": self.ts_utc,
            "symbol": self.symbol,
            "interval": self.interval,
            "source": self.source,
            "side_candidate": self.side_candidate,
            "score_edge": float(self.score_edge),
            "confidence": float(self.confidence),
            "atr_pct": float(self.atr_pct),
            "spread_pct": float(self.spread_pct),
            "liq_score": float(self.liq_score),
            "whale_score": float(self.whale_score),
            "whale_dir": self.whale_dir,
            "dedup_key": self.dedup_key,
            "meta": self.meta if isinstance(self.meta, dict) else {},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], default_source: str = "w?") -> "SignalEvent":
        if not isinstance(d, dict):
            d = {}
        ev = SignalEvent(
            ts_utc=_safe_str(d.get("ts_utc"), _now_utc_iso()),
            symbol=_safe_str(d.get("symbol"), ""),
            interval=_safe_str(d.get("interval"), "5m"),
            source=_safe_str(d.get("source"), default_source or "w?"),
            side_candidate=_safe_str(d.get("side_candidate"), _safe_str(d.get("side"), "none")),
            score_edge=_safe_float(d.get("score_edge", d.get("score", 0.0)), 0.0),
            confidence=_safe_float(d.get("confidence", d.get("conf", 0.0)), 0.0),
            atr_pct=_safe_float(d.get("atr_pct", 0.0), 0.0),
            spread_pct=_safe_float(d.get("spread_pct", 0.0), 0.0),
            liq_score=_safe_float(d.get("liq_score", 0.0), 0.0),
            whale_score=_safe_float(d.get("whale_score", 0.0), 0.0),
            whale_dir=_safe_str(d.get("whale_dir", "none"), "none"),
            dedup_key=_safe_str(d.get("dedup_key", ""), ""),
            meta=d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {},
        )
        return ev.normalize()


@dataclass
class CandidateTrade:
    """
    Aggregator -> TopSelector standart formatı.
    score_total tek sayı; MasterExecutor skor/whale/risk işini kendi yapıyor ama
    burada iyi bir 'aday sıralama' gerekir.
    """
    ts_utc: str
    symbol: str
    interval: str
    side: str
    score_total: float

    recommended_leverage: int = 5
    recommended_notional_pct: float = 0.05

    confidence: float = 0.0
    atr_pct: float = 0.0
    spread_pct: float = 0.0
    liq_score: float = 0.0

    whale_score: float = 0.0
    whale_dir: str = "none"

    risk_tags: List[str] = field(default_factory=list)
    reason_codes: List[str] = field(default_factory=list)

    raw: Dict[str, Any] = field(default_factory=dict)

    def normalize(self) -> "CandidateTrade":
        self.ts_utc = _safe_str(self.ts_utc, _now_utc_iso()) or _now_utc_iso()
        self.symbol = _safe_str(self.symbol, "").upper()
        self.interval = _safe_str(self.interval, "5m")
        s = _safe_str(self.side, "long").strip().lower()
        if s in ("buy", "long"):
            s = "long"
        elif s in ("sell", "short"):
            s = "short"
        self.side = s

        self.score_total = float(_clamp(_safe_float(self.score_total, 0.0), 0.0, 1.0))
        self.recommended_leverage = int(_clamp(float(_safe_int(self.recommended_leverage, 5)), 3.0, 30.0))
        self.recommended_notional_pct = float(_clamp(_safe_float(self.recommended_notional_pct, 0.05), 0.0, 1.0))

        self.confidence = float(_clamp(_safe_float(self.confidence, 0.0), 0.0, 1.0))
        self.atr_pct = float(_clamp(_safe_float(self.atr_pct, 0.0), 0.0, 1.0))
        self.spread_pct = float(_clamp(_safe_float(self.spread_pct, 0.0), 0.0, 1.0))
        self.liq_score = float(_clamp(_safe_float(self.liq_score, 0.0), 0.0, 1.0))

        self.whale_score = float(_clamp(_safe_float(self.whale_score, 0.0), 0.0, 1.0))
        wd = _safe_str(self.whale_dir, "none").strip().lower()
        if wd in ("buy", "long", "in", "inflow"):
            wd = "long"
        elif wd in ("sell", "short", "out", "outflow"):
            wd = "short"
        elif wd in ("", "none", "neutral"):
            wd = "none"
        self.whale_dir = wd

        self.risk_tags = [str(x) for x in _as_list_str(self.risk_tags)]
        self.reason_codes = [str(x) for x in _as_list_str(self.reason_codes)]
        if not isinstance(self.raw, dict):
            self.raw = {}

        return self
    def to_dict(self) -> Dict[str, Any]:
        self.normalize()
        return {
            "ts_utc": self.ts_utc,
            "symbol": self.symbol,
            "interval": self.interval,
            "side": self.side,
            "score_total": float(self.score_total),
            "recommended_leverage": int(self.recommended_leverage),
            "recommended_notional_pct": float(self.recommended_notional_pct),
            "confidence": float(self.confidence),
            "atr_pct": float(self.atr_pct),
            "spread_pct": float(self.spread_pct),
            "liq_score": float(self.liq_score),
            "whale_score": float(self.whale_score),
            "whale_dir": self.whale_dir,
            "risk_tags": list(self.risk_tags or []),
            "reason_codes": list(self.reason_codes or []),
            "raw": self.raw if isinstance(self.raw, dict) else {},
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CandidateTrade":
        if not isinstance(d, dict):
            d = {}
        c = CandidateTrade(
            ts_utc=_safe_str(d.get("ts_utc"), _now_utc_iso()),
            symbol=_safe_str(d.get("symbol"), ""),
            interval=_safe_str(d.get("interval"), "5m"),
            side=_safe_str(d.get("side"), "long"),
            score_total=_safe_float(d.get("score_total", d.get("_score_total_final", 0.0)), 0.0),
            recommended_leverage=_safe_int(d.get("recommended_leverage", 5), 5),
            recommended_notional_pct=_safe_float(d.get("recommended_notional_pct", 0.05), 0.05),
            confidence=_safe_float(d.get("confidence", 0.0), 0.0),
            atr_pct=_safe_float(d.get("atr_pct", 0.0), 0.0),
            spread_pct=_safe_float(d.get("spread_pct", 0.0), 0.0),
            liq_score=_safe_float(d.get("liq_score", 0.0), 0.0),
            whale_score=_safe_float(d.get("whale_score", 0.0), 0.0),
            whale_dir=_safe_str(d.get("whale_dir", "none"), "none"),
            risk_tags=_as_list_str(d.get("risk_tags")),
            reason_codes=_as_list_str(d.get("reason_codes", d.get("reasons"))),
            raw=d.get("raw", {}) if isinstance(d.get("raw", {}), dict) else {},
        )
        return c.normalize()
