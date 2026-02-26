from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: Any) -> float:
    v = _safe_float(x, 0.0)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class SignalEvent:
    # producer metadata
    event_id: str
    producer_id: str  # e.g. "w0", "w1"
    ts_utc: str = field(default_factory=utcnow_iso)

    # market
    symbol: str = ""
    interval: str = "5m"

    # candidate decision (NOT final)
    side_candidate: str = "none"  # "long" | "short" | "none"

    # model outputs
    p_used: float = 0.0
    p_single: float = 0.0
    confidence: float = 0.0  # 0-1
    source: str = "HYBRID"

    # risk/market quality hints
    whale_dir: str = "none"
    whale_score: float = 0.0
    price: float = 0.0
    atr_pct: float = 0.0
    spread_pct: float = 0.0
    vol_score: float = 0.0

    # dedupe
    cooldown_key: str = ""  # e.g. "XRPUSDT|5m|long"
    dedup_key: str = ""     # standard

    # freeform
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def normalize(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort normalization for incoming dict payloads.

        Accepts aliases:
          - producer_id or source (w1..w8)
          - dedup_key or cooldown_key

        Ensures:
          - symbol upper
          - interval default "5m"
          - side_candidate in {"long","short","none"}
          - ts_utc exists
          - price / scores are safe floats
        """
        if not isinstance(d, dict):
            return {}

        out = dict(d)

        # producer_id alias
        if not out.get("producer_id") and out.get("source"):
            out["producer_id"] = str(out.get("source"))

        # ts
        if not out.get("ts_utc"):
            out["ts_utc"] = utcnow_iso()

        # symbol/interval
        sym = str(out.get("symbol") or "").upper().strip()
        if sym:
            out["symbol"] = sym
        itv = str(out.get("interval") or "").strip() or "5m"
        out["interval"] = itv

        # side_candidate normalize (fail-closed)
        sc = str(out.get("side_candidate") or "none").strip().lower()
        if sc in ("buy", "long"):
            sc = "long"
        elif sc in ("sell", "short"):
            sc = "short"
        elif sc not in ("long", "short"):
            sc = "none"
        out["side_candidate"] = sc

        # dedupe key
        dk = str(out.get("dedup_key") or "").strip()
        if not dk:
            dk = str(out.get("cooldown_key") or "").strip()
        if not dk and sym:
            dk = f"{sym}|{itv}|{sc}"
        out["dedup_key"] = dk

        # keep old field for backward compat
        if not out.get("cooldown_key"):
            out["cooldown_key"] = dk

        # normalize numerics (safe)
        out["price"] = max(0.0, _safe_float(out.get("price", 0.0), 0.0))
        out["confidence"] = _clamp01(out.get("confidence", out.get("conf", 0.0)))
        out["whale_score"] = _clamp01(out.get("whale_score", 0.0))

        out["p_used"] = _clamp01(out.get("p_used", 0.0))
        out["p_single"] = _clamp01(out.get("p_single", 0.0))

        out["atr_pct"] = max(0.0, _safe_float(out.get("atr_pct", 0.0), 0.0))
        out["spread_pct"] = max(0.0, _safe_float(out.get("spread_pct", 0.0), 0.0))
        out["vol_score"] = _clamp01(out.get("vol_score", 0.0))

        # whale_dir normalize
        wd = str(out.get("whale_dir") or "none").strip().lower()
        out["whale_dir"] = wd or "none"

        return out


@dataclass
class CandidateTrade:
    candidate_id: str
    ts_utc: str = field(default_factory=utcnow_iso)

    symbol: str = ""
    interval: str = "5m"
    side: str = "long"  # "long" | "short"

    # IMPORTANT: price passthrough (critical for downstream)
    price: float = 0.0

    score_total: float = 0.0

    # provenance / dedupe
    source: str = ""
    dedup_key: str = ""

    # whale passthrough (optional)
    whale_score: float = 0.0
    whale_dir: str = "none"
    reasons: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)

    # sizing hints
    recommended_notional_pct: float = 0.0
    recommended_leverage: int = 3

    # optional debug
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # normalize numeric fields defensively
        d["price"] = max(0.0, _safe_float(d.get("price", 0.0), 0.0))
        d["score_total"] = _clamp01(d.get("score_total", 0.0))
        d["whale_score"] = _clamp01(d.get("whale_score", 0.0))
        d["recommended_notional_pct"] = max(0.0, _safe_float(d.get("recommended_notional_pct", 0.0), 0.0))
        d["recommended_leverage"] = max(0, _safe_int(d.get("recommended_leverage", 3), 3))
        return d


@dataclass
class TopKBatchEvent:
    """
    top5_stream payload standard.

    Produced by TopSelector:
      {"ts_utc": "...", "topk": N, "items": [CandidateTrade dicts], ...}

    This schema keeps extra selector metadata for debugging/ops.
    """
    ts_utc: str = field(default_factory=utcnow_iso)
    topk: int = 0
    items: List[Dict[str, Any]] = field(default_factory=list)

    # selector metadata (optional)
    window_sec: int = 0
    selector_id: str = "topsel"
    min_score: float = 0.0
    w_min: float = 0.0

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["topk"] = max(0, int(d.get("topk", 0) or 0))
        d["window_sec"] = max(0, int(d.get("window_sec", 0) or 0))
        d["min_score"] = float(d.get("min_score", 0.0) or 0.0)
        d["w_min"] = float(d.get("w_min", 0.0) or 0.0)
        if not d.get("ts_utc"):
            d["ts_utc"] = utcnow_iso()
        return d
