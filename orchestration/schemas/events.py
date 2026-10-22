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
    model_score: float = 0.0  # side-specific canonical MODEL score; SHORT => 1-p_buy
    source: str = "HYBRID"

    # risk/market quality hints
    whale_dir: str = "none"
    whale_score: float = 0.0
    price: float = 0.0
    atr_pct: float = 0.0
    spread_pct: float = 0.0
    vol_score: float = 0.0

    # dedupe
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
          - dedup_key

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
        if not dk and sym:
            dk = f"{sym}|{itv}|{sc}"
        out["dedup_key"] = dk

        # normalize numerics (safe)
        out["price"] = max(0.0, _safe_float(out.get("price", 0.0), 0.0))
        out["whale_score"] = _clamp01(out.get("whale_score", 0.0))

        out["model_score"] = _clamp01(out.get("model_score", 0.0))

        out["atr_pct"] = max(0.0, _safe_float(out.get("atr_pct", 0.0), 0.0))
        out["spread_pct"] = max(0.0, _safe_float(out.get("spread_pct", 0.0), 0.0))
        out["vol_score"] = _clamp01(out.get("vol_score", 0.0))

        # whale_dir normalize
        wd = str(out.get("whale_dir") or "none").strip().lower()
        out["whale_dir"] = wd or "none"

        return out


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

    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["topk"] = max(0, int(d.get("topk", 0) or 0))
        d["window_sec"] = max(0, int(d.get("window_sec", 0) or 0))
        if not d.get("ts_utc"):
            d["ts_utc"] = utcnow_iso()
        return d
