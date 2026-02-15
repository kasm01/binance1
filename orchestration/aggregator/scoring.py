from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        fx = float(x)
    except Exception:
        fx = lo
    return max(lo, min(hi, fx))


def compute_score(evt: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """
    SignalEvent -> score_total(0..1), reasons, risk_tags

    Primary: score_edge + confidence + liq_score (+ whale_score optional)
    Penalize: spread_pct, atr_pct (vol)
    """
    reasons: List[str] = []
    risk_tags: List[str] = []

    # --- weights (env) ---
    W_EDGE = _env_float("SCORE_W_EDGE", 0.55)
    W_CONF = _env_float("SCORE_W_CONF", 0.25)
    W_LIQ = _env_float("SCORE_W_LIQ", 0.25)
    W_WHALE = _env_float("SCORE_W_WHALE", 0.15)

    W_SPREAD_PEN = _env_float("SCORE_W_SPREAD_PEN", 0.35)
    W_VOL_PEN = _env_float("SCORE_W_VOL_PEN", 0.25)

    # normalize refs
    ATR_REF = _env_float("SCORE_ATR_REF", 0.030)   # 3%
    SPR_REF = _env_float("SCORE_SPR_REF", 0.0010)  # 0.10%

    # tagging thresholds
    WIDE_SPREAD_TAG = _env_float("TAG_WIDE_SPREAD_AT", 0.0009)
    HIGH_VOL_TAG = _env_float("TAG_HIGH_VOL_ATR_AT", 0.030)
    LOW_LIQ_TAG = _env_float("TAG_LOW_LIQ_AT", 0.25)
    WHALE_STRONG_TAG = _env_float("TAG_WHALE_STRONG_AT", 0.35)

    # --- inputs ---
    edge = float(evt.get("score_edge", 0.0) or 0.0)
    conf = float(evt.get("confidence", 0.0) or 0.0)
    liq = float(evt.get("liq_score", 0.0) or 0.0)
    whale = float(evt.get("whale_score", 0.0) or 0.0)

    spread = float(evt.get("spread_pct", 0.0) or 0.0)
    atr = float(evt.get("atr_pct", 0.0) or 0.0)

    # penalties 0..1
    spr_pen = min(1.0, spread / max(SPR_REF, 1e-9)) if spread > 0 else 0.0
    vol_pen = min(1.0, atr / max(ATR_REF, 1e-9)) if atr > 0 else 0.0

    score_raw = (
        W_EDGE * edge +
        W_CONF * conf +
        W_LIQ * liq +
        W_WHALE * whale -
        (W_SPREAD_PEN * spr_pen) -
        (W_VOL_PEN * vol_pen)
    )

    # reasons
    if edge >= 0.55:
        reasons.append("edge_strong")
    elif edge >= 0.35:
        reasons.append("edge_ok")

    if conf >= 0.70:
        reasons.append("conf_strong")
    elif conf >= 0.55:
        reasons.append("conf_ok")

    if liq >= 0.60:
        reasons.append("liq_ok")

    if whale >= 0.20:
        reasons.append("whale_seen")
    if whale >= WHALE_STRONG_TAG:
        reasons.append("whale_strong")

    if spr_pen <= 0.30:
        reasons.append("tight_spread")
    if vol_pen <= 0.50:
        reasons.append("vol_ok")

    # risk tags
    if spread >= WIDE_SPREAD_TAG:
        risk_tags.append("wide_spread")
    if atr >= HIGH_VOL_TAG:
        risk_tags.append("high_vol")
    if liq <= LOW_LIQ_TAG:
        risk_tags.append("low_liq")
    if whale >= WHALE_STRONG_TAG:
        risk_tags.append("whale_strong")

    score_total = _clamp(score_raw, 0.0, 1.0)
    return float(score_total), reasons, risk_tags


def pass_quality_gates(evt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Hard filters (SignalEvent)
      - side must be long/short
      - spread/atr under max
      - min confidence
      - optional: min liq
    """
    side = str(evt.get("side_candidate", "none") or "none").lower()
    if side not in ("long", "short"):
        return False, "side_none"

    spread = float(evt.get("spread_pct", 0.0) or 0.0)
    max_spread = _env_float("GATE_MAX_SPREAD_PCT", 0.0015)
    if spread > max_spread:
        return False, "spread_too_high"

    atr = float(evt.get("atr_pct", 0.0) or 0.0)
    max_atr = _env_float("GATE_MAX_ATR_PCT", 0.060)
    if atr > max_atr:
        return False, "atr_too_high"

    conf = float(evt.get("confidence", 0.0) or 0.0)
    min_conf = _env_float("GATE_MIN_CONF", 0.35)
    if conf < min_conf:
        return False, "conf_too_low"

    # optional liq gate (fail-open by default with low threshold)
    liq = float(evt.get("liq_score", 0.0) or 0.0)
    min_liq = _env_float("GATE_MIN_LIQ", 0.0)
    if liq < min_liq:
        return False, "liq_too_low"

    return True, "ok"

