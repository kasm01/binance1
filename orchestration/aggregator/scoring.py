from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        fx = float(x)
    except Exception:
        fx = lo
    return max(lo, min(hi, fx))


def _meta_get(evt: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        meta = evt.get("meta") or {}
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
    except Exception:
        pass
    return default


def _norm_dir(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long", "in", "inflow"):
        return "long"
    if s in ("sell", "short", "out", "outflow"):
        return "short"
    return "none"


def _get_whale_score(evt: Dict[str, Any]) -> float:
    ws = _safe_float(evt.get("whale_score", 0.0), 0.0)
    if ws > 0:
        return float(_clamp(ws, 0.0, 1.0))
    ws2 = _safe_float(_meta_get(evt, "whale_score", 0.0), 0.0)
    return float(_clamp(ws2, 0.0, 1.0))


def _get_whale_dir(evt: Dict[str, Any]) -> str:
    wd = evt.get("whale_dir", "")
    if wd:
        return _norm_dir(wd)
    return _norm_dir(_meta_get(evt, "whale_dir", ""))


def _get_side(evt: Dict[str, Any]) -> str:
    s = str(evt.get("side_candidate", evt.get("side", "none")) or "none").strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return "none"


def _get_trend_score(evt: Dict[str, Any], tf: str, side: str) -> float:
    """
    meta içinden trend / p_used / tf score okumaya çalışır.
    Beklenen yerler:
      meta.per_tf.15m.p_used
      meta.per_tf.15m.score
      meta.tf_scores.15m
      meta.trend_15m
    """
    meta = evt.get("meta") if isinstance(evt.get("meta"), dict) else {}
    if not isinstance(meta, dict):
        return 0.5

    try:
        per_tf = meta.get("per_tf") or {}
        if isinstance(per_tf, dict):
            row = per_tf.get(tf) or {}
            if isinstance(row, dict):
                for k in ("p_used", "score", "p", "prob"):
                    if k in row:
                        return float(_clamp(_safe_float(row.get(k), 0.5), 0.0, 1.0))
    except Exception:
        pass

    try:
        tf_scores = meta.get("tf_scores") or {}
        if isinstance(tf_scores, dict) and tf in tf_scores:
            return float(_clamp(_safe_float(tf_scores.get(tf), 0.5), 0.0, 1.0))
    except Exception:
        pass

    try:
        key = f"trend_{tf}"
        if key in meta:
            return float(_clamp(_safe_float(meta.get(key), 0.5), 0.0, 1.0))
    except Exception:
        pass

    return 0.5


def _trend_veto_hit(evt: Dict[str, Any]) -> Tuple[bool, str]:
    side = _get_side(evt)
    if side not in ("long", "short"):
        return False, "trend_skip"

    p15 = _get_trend_score(evt, "15m", side)
    p30 = _get_trend_score(evt, "30m", side)
    p1h = _get_trend_score(evt, "1h", side)

    if side == "long":
        t15 = _env_float("TREND_VETO_LONG_15M", 0.48)
        t30 = _env_float("TREND_VETO_LONG_30M", 0.48)
        t1h = _env_float("TREND_VETO_LONG_1H", 0.47)
        if p15 < t15:
            return True, "trend_veto_15m"
        if p30 < t30:
            return True, "trend_veto_30m"
        if p1h < t1h:
            return True, "trend_veto_1h"
        return False, "ok"

    t15 = _env_float("TREND_VETO_SHORT_15M", 0.52)
    t30 = _env_float("TREND_VETO_SHORT_30M", 0.52)
    t1h = _env_float("TREND_VETO_SHORT_1H", 0.53)
    if p15 > t15:
        return True, "trend_veto_15m"
    if p30 > t30:
        return True, "trend_veto_30m"
    if p1h > t1h:
        return True, "trend_veto_1h"
    return False, "ok"


def compute_score(evt: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    """
    SignalEvent -> (score_total 0..1, reasons, risk_tags)

    Whale-first overlay:
      - aligned whale => bonus
      - contra whale => penalty
      - strong contra whale => reason/tag
    """
    reasons: List[str] = []
    risk_tags: List[str] = []

    W_EDGE = _env_float("SCORE_W_EDGE", 0.52)
    W_CONF = _env_float("SCORE_W_CONF", 0.22)
    W_LIQ = _env_float("SCORE_W_LIQ", 0.18)
    W_WHALE = _env_float("SCORE_W_WHALE", 0.18)

    W_SPREAD_PEN = _env_float("SCORE_W_SPREAD_PEN", 0.30)
    W_VOL_PEN = _env_float("SCORE_W_VOL_PEN", 0.20)

    ATR_REF = _env_float("SCORE_ATR_REF", 0.030)
    SPR_REF = _env_float("SCORE_SPR_REF", 0.0010)

    WIDE_SPREAD_TAG = _env_float("TAG_WIDE_SPREAD_AT", 0.0009)
    HIGH_VOL_TAG = _env_float("TAG_HIGH_VOL_ATR_AT", 0.030)
    LOW_LIQ_TAG = _env_float("TAG_LOW_LIQ_AT", 0.25)
    WHALE_STRONG_TAG = _env_float("TAG_WHALE_STRONG_AT", 0.35)

    WHALE_CONFIRM_THR = _env_float("WHALE_CONFIRM_THR", 0.60)
    WHALE_VETO_THR = _env_float("WHALE_VETO_THR", 0.70)
    WHALE_TREND_BONUS = _env_float("WHALE_TREND_BONUS", 1.15)
    WHALE_RANGE_PENALTY = _env_float("WHALE_RANGE_PENALTY", 0.65)
    WHALE_XCONF_BONUS = _env_float("WHALE_XCONF_BONUS", 1.10)

    edge = _clamp(_safe_float(evt.get("score_edge", 0.0), 0.0), 0.0, 1.0)
    conf = _clamp(_safe_float(evt.get("confidence", 0.0), 0.0), 0.0, 1.0)
    liq = _clamp(_safe_float(evt.get("liq_score", 0.0), 0.0), 0.0, 1.0)
    whale = _get_whale_score(evt)

    spread = max(0.0, _safe_float(evt.get("spread_pct", 0.0), 0.0))
    atr = max(0.0, _safe_float(evt.get("atr_pct", 0.0), 0.0))

    spr_pen = min(1.0, spread / max(SPR_REF, 1e-9)) if spread > 0 else 0.0
    vol_pen = min(1.0, atr / max(ATR_REF, 1e-9)) if atr > 0 else 0.0

    score_raw = (
        (W_EDGE * edge)
        + (W_CONF * conf)
        + (W_LIQ * liq)
        + (W_WHALE * whale)
        - (W_SPREAD_PEN * spr_pen)
        - (W_VOL_PEN * vol_pen)
    )

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
    elif liq >= 0.40:
        reasons.append("liq_mid")

    if whale >= 0.20:
        reasons.append("whale_seen")
    if whale >= WHALE_STRONG_TAG:
        reasons.append("whale_strong")

    if spr_pen <= 0.30:
        reasons.append("tight_spread")
    if vol_pen <= 0.50:
        reasons.append("vol_ok")

    side = _get_side(evt)
    wdir = _get_whale_dir(evt)

    if whale >= WHALE_STRONG_TAG and side in ("long", "short") and wdir in ("long", "short"):
        if wdir == side:
            reasons.append(f"whale_align_{side}")
            score_raw *= WHALE_XCONF_BONUS
            if whale >= WHALE_CONFIRM_THR:
                reasons.append("whale_confirm")
        else:
            reasons.append(f"whale_contra_{side}")
            risk_tags.append("whale_contra")
            score_raw *= 0.82
            if whale >= WHALE_VETO_THR:
                reasons.append("whale_hard_contra")

    regime = str(evt.get("market_regime", _meta_get(evt, "market_regime", "")) or "").strip().lower()
    if regime == "trend" and whale >= WHALE_CONFIRM_THR and wdir == side and side in ("long", "short"):
        score_raw *= WHALE_TREND_BONUS
        reasons.append("trend_whale_bonus")
    elif regime == "range" and whale >= WHALE_CONFIRM_THR:
        score_raw *= WHALE_RANGE_PENALTY
        reasons.append("range_whale_penalty")
        risk_tags.append("range_regime")

    if spread >= WIDE_SPREAD_TAG:
        risk_tags.append("wide_spread")
    if atr >= HIGH_VOL_TAG:
        risk_tags.append("high_vol")
    if liq <= LOW_LIQ_TAG:
        risk_tags.append("low_liq")
    if whale >= WHALE_STRONG_TAG:
        risk_tags.append("whale_strong")

    score_total = _clamp(score_raw, 0.0, 1.0)
    return float(score_total), list(dict.fromkeys(reasons)), list(dict.fromkeys(risk_tags))


def pass_quality_gates(evt: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Hard filters
      - side must be long/short
      - spread/atr under max
      - min confidence
      - optional min liq / min whale
      - trend veto
      - strong contra whale veto
    """
    side = _get_side(evt)
    if side not in ("long", "short"):
        return False, "side_none"

    spread = float(_safe_float(evt.get("spread_pct", 0.0), 0.0))
    max_spread = _env_float("GATE_MAX_SPREAD_PCT", 0.0015)
    if spread > max_spread:
        return False, "spread_too_high"

    atr = float(_safe_float(evt.get("atr_pct", 0.0), 0.0))
    max_atr = _env_float("GATE_MAX_ATR_PCT", 0.060)
    if atr > max_atr:
        return False, "atr_too_high"

    conf = float(_safe_float(evt.get("confidence", 0.0), 0.0))
    min_conf = _env_float("GATE_MIN_CONF", 0.35)
    if conf < min_conf:
        return False, "conf_too_low"

    liq = float(_safe_float(evt.get("liq_score", 0.0), 0.0))
    min_liq = _env_float("GATE_MIN_LIQ", 0.0)
    if liq < min_liq:
        return False, "liq_too_low"

    min_whale = _env_float("GATE_MIN_WHALE_SCORE", 0.0)
    if min_whale > 0.0:
        ws = _get_whale_score(evt)
        if ws < min_whale:
            return False, "whale_too_low"

    trend_hit, trend_reason = _trend_veto_hit(evt)
    if trend_hit:
        return False, trend_reason

    whale_score = _get_whale_score(evt)
    whale_dir = _get_whale_dir(evt)

    if whale_dir in ("long", "short") and side in ("long", "short") and whale_dir != side:
        veto_thr = _env_float("WHALE_VETO_THR", 0.70)
        if whale_score >= veto_thr:
            return False, "whale_contra_veto"

    return True, "ok"
