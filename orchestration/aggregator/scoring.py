from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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

    Aggressive scalping tuned:
      - edge + confidence are primary
      - whale alignment boosts strongly
      - whale CONTRA applies hard penalty (or veto via quality gates)
      - spread/atr are penalized, but not overly (aggressive mode)
    """
    reasons: List[str] = []
    risk_tags: List[str] = []

    # --- Tunables (env) ---
    W_EDGE = _env_float("SCORE_W_EDGE", 1.55)
    W_CONF = _env_float("SCORE_W_CONF", 0.55)
    W_WHALE = _env_float("SCORE_W_WHALE", 0.55)

    W_SPREAD_PEN = _env_float("SCORE_W_SPREAD_PEN", 0.55)
    W_VOL_PEN = _env_float("SCORE_W_VOL_PEN", 0.30)

    # contra whale penalty (very important for "nokta atışı")
    WHALE_CONTRA_PEN = _env_float("SCORE_WHALE_CONTRA_PEN", 0.70)

    # Normalize thresholds
    ATR_REF = _env_float("SCORE_ATR_REF", 0.030)      # 3% ATR => vol_pen ~ 1
    SPR_REF = _env_float("SCORE_SPR_REF", 0.0010)     # 0.10% spread => spr_pen ~ 1

    # Risk tagging thresholds
    WIDE_SPREAD_TAG = _env_float("TAG_WIDE_SPREAD_AT", 0.0009)  # 0.09%
    HIGH_VOL_TAG = _env_float("TAG_HIGH_VOL_ATR_AT", 0.030)     # 3%

    # --- Inputs ---
    p_used = float(evt.get("p_used", 0.0) or 0.0)
    confidence = float(evt.get("confidence", 0.0) or 0.0)
    spread_pct = float(evt.get("spread_pct", 0.0) or 0.0)
    atr_pct = float(evt.get("atr_pct", 0.0) or 0.0)

    whale_score = float(evt.get("whale_score", 0.0) or 0.0)
    whale_dir = str(evt.get("whale_dir", "none") or "none").lower()

    side = str(evt.get("side_candidate", "none") or "none").lower()

    # --- edge: 0..1 ---
    if side == "long":
        edge = max(0.0, (p_used - 0.5) * 2.0)
    elif side == "short":
        edge = max(0.0, (0.5 - p_used) * 2.0)
    else:
        edge = 0.0

    # --- penalties (0..1) ---
    vol_pen = min(1.0, atr_pct / max(ATR_REF, 1e-9)) if atr_pct > 0 else 0.0
    spr_pen = min(1.0, spread_pct / max(SPR_REF, 1e-9)) if spread_pct > 0 else 0.0

    # --- whale alignment / contra ---
    whale_bonus = 0.0
    whale_contra = 0.0

    whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
    whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")

    if whale_is_buy and side == "long":
        whale_bonus = _clamp(whale_score, 0.0, 0.7)
        reasons.append("whale_align_long")
    elif whale_is_sell and side == "short":
        whale_bonus = _clamp(whale_score, 0.0, 0.7)
        reasons.append("whale_align_short")
    elif (whale_is_buy and side == "short") or (whale_is_sell and side == "long"):
        # CONTRA whale = big danger for scalping
        whale_contra = _clamp(whale_score, 0.0, 1.0)
        reasons.append("whale_contra")

    # --- base score ---
    score = (
        W_EDGE * edge +
        W_CONF * confidence +
        W_WHALE * whale_bonus -
        (WHALE_CONTRA_PEN * whale_contra) -
        W_SPREAD_PEN * spr_pen -
        W_VOL_PEN * vol_pen
    )

    # Risk tags
    if spread_pct >= WIDE_SPREAD_TAG:
        risk_tags.append("wide_spread")
    if atr_pct >= HIGH_VOL_TAG:
        risk_tags.append("high_vol")
    if whale_contra > 0.15:
        risk_tags.append("whale_contra")

    # Reasons
    if edge >= 0.30:
        reasons.append("edge_ok")
    if confidence >= 0.55:
        reasons.append("conf_ok")
    if whale_bonus >= 0.20:
        reasons.append("whale_strong")

    return float(score), reasons, risk_tags


def pass_quality_gates(evt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Hard filters to avoid garbage.
    Aggressive scalping gates:
      - no side_none
      - spread too high => drop
      - atr too high => drop (very wild)
      - whale contra strong => drop (key!)
      - optional: edge/conf minimum
    """
    side = str(evt.get("side_candidate", "none") or "none").lower()
    if side not in ("long", "short"):
        return False, "side_none"

    spread_pct = float(evt.get("spread_pct", 0.0) or 0.0)
    max_spread = _env_float("GATE_MAX_SPREAD_PCT", 0.0015)  # default 0.15%
    if spread_pct > max_spread:
        return False, "spread_too_high"

    atr_pct = float(evt.get("atr_pct", 0.0) or 0.0)
    max_atr = _env_float("GATE_MAX_ATR_PCT", 0.060)  # 6%
    if atr_pct > max_atr:
        return False, "atr_too_high"

    # Whale contra veto (agresif hedef için kritik)
    whale_score = float(evt.get("whale_score", 0.0) or 0.0)
    whale_dir = str(evt.get("whale_dir", "none") or "none").lower()
    whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
    whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")

    contra = (whale_is_buy and side == "short") or (whale_is_sell and side == "long")
    contra_thr = _env_float("GATE_WHALE_CONTRA_THR", 0.25)
    if contra and whale_score >= contra_thr:
        return False, "whale_contra_veto"

    # Optional minimum edge/conf (spam keser)
    p_used = float(evt.get("p_used", 0.0) or 0.0)
    conf = float(evt.get("confidence", 0.0) or 0.0)
    min_conf = _env_float("GATE_MIN_CONF", 0.25)
    if conf < min_conf:
        return False, "conf_too_low"

    # edge min
    if side == "long":
        edge = max(0.0, (p_used - 0.5) * 2.0)
    else:
        edge = max(0.0, (0.5 - p_used) * 2.0)
    min_edge = _env_float("GATE_MIN_EDGE", 0.10)
    if edge < min_edge:
        return False, "edge_too_low"

    return True, "ok"

