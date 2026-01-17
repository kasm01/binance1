from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
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
    atr_pct: float = 0.0
    spread_pct: float = 0.0
    vol_score: float = 0.0

    # dedupe
    cooldown_key: str = ""  # e.g. "XRPUSDT|5m|long"

    # freeform
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # keep payload json-safe
        return d


@dataclass
class CandidateTrade:
    candidate_id: str
    ts_utc: str = field(default_factory=utcnow_iso)

    symbol: str = ""
    interval: str = "5m"
    side: str = "long"  # "long" | "short"

    score_total: float = 0.0
    reasons: List[str] = field(default_factory=list)
    risk_tags: List[str] = field(default_factory=list)

    # sizing hints (0..1 notional ratio suggestion)
    recommended_notional_pct: float = 0.0
    recommended_leverage: int = 3

    # optional debug
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
