from __future__ import annotations

from typing import Any, Dict, List, Tuple


def side_from_p(p_used: float, buy_thr: float = 0.60, sell_thr: float = 0.40) -> str:
    if p_used >= buy_thr:
        return "long"
    if p_used <= sell_thr:
        return "short"
    return "none"


def compute_score(evt: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """
    Returns: (score_total, reasons, risk_tags)
    score_total higher is better.
    """
    reasons: List[str] = []
    risk_tags: List[str] = []

    p_used = float(evt.get("p_used", 0.0) or 0.0)
    confidence = float(evt.get("confidence", 0.0) or 0.0)
    spread_pct = float(evt.get("spread_pct", 0.0) or 0.0)
    atr_pct = float(evt.get("atr_pct", 0.0) or 0.0)
    whale_score = float(evt.get("whale_score", 0.0) or 0.0)
    whale_dir = str(evt.get("whale_dir", "none") or "none")

    side = str(evt.get("side_candidate", "none") or "none")

    # edge: 0..1
    if side == "long":
        edge = max(0.0, (p_used - 0.5) * 2.0)
    elif side == "short":
        edge = max(0.0, (0.5 - p_used) * 2.0)
    else:
        edge = 0.0

    # penalties
    vol_pen = min(1.0, atr_pct / 0.03) if atr_pct > 0 else 0.0  # 3% ATR -> 1.0 penalty
    spr_pen = min(1.0, spread_pct / 0.001) if spread_pct > 0 else 0.0  # 0.10% spread -> 1.0 penalty

    # whale alignment
    whale_bonus = 0.0
    if whale_dir in ("buy", "long") and side == "long":
        whale_bonus = min(0.5, whale_score)
        reasons.append("whale_align_long")
    if whale_dir in ("sell", "short") and side == "short":
        whale_bonus = min(0.5, whale_score)
        reasons.append("whale_align_short")

    # base score
    score = (
        1.40 * edge +
        0.60 * confidence +
        0.30 * whale_bonus -
        0.60 * spr_pen -
        0.40 * vol_pen
    )

    if spr_pen > 0.7:
        risk_tags.append("wide_spread")
    if vol_pen > 0.7:
        risk_tags.append("high_vol")

    if edge > 0.3:
        reasons.append("edge_ok")
    if confidence > 0.5:
        reasons.append("conf_ok")

    return float(score), reasons, risk_tags


def pass_quality_gates(evt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Hard filters to avoid garbage.
    """
    side = str(evt.get("side_candidate", "none") or "none")
    if side not in ("long", "short"):
        return False, "side_none"

    spread_pct = float(evt.get("spread_pct", 0.0) or 0.0)
    if spread_pct > 0.0020:  # 0.20%
        return False, "spread_too_high"

    atr_pct = float(evt.get("atr_pct", 0.0) or 0.0)
    if atr_pct > 0.06:  # 6% ATR on short TF => too wild
        return False, "atr_too_high"

    return True, "ok"
