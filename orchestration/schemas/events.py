from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    # dedupe (standard)
    dedup_key: str = ""  # e.g. "XRPUSDT|5m|long"

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

        # side_candidate normalize
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

        # normalize price (optional)
        try:
            out["price"] = float(out.get("price", 0.0) or 0.0)
        except Exception:
            out["price"] = 0.0

        # keep old field for backward compat
        if not out.get("cooldown_key"):
            out["cooldown_key"] = dk

        return out


@dataclass
class CandidateTrade:
    candidate_id: str
    ts_utc: str = field(default_factory=utcnow_iso)

    symbol: str = ""
    interval: str = "5m"
    side: str = "long"  # "long" | "short"

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
        return asdict(self)


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
        return asdict(self)
