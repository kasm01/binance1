from __future__ import annotations

from typing import Any, Dict


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def classify_confirmed_position_build(
    *,
    oi_delta_15m: Any,
    price_return_15m: Any,
    money_flow_15m: Any,
) -> Dict[str, Any]:
    """Canonical OI/price/taker-flow regime classifier.

    This function classifies market state only. It never decides whether to
    reject a trade or how large the position should be.
    """
    oi15 = _safe_float(oi_delta_15m, 0.0)
    ret15 = _safe_float(price_return_15m, 0.0)
    flow15 = _safe_float(money_flow_15m, 0.0)

    confirmed = bool(
        oi15 > 0.0
        and ret15 != 0.0
        and flow15 != 0.0
        and ((ret15 > 0.0) == (flow15 > 0.0))
    )
    direction = (
        "long" if confirmed and ret15 > 0.0
        else "short" if confirmed and ret15 < 0.0
        else "none"
    )
    return {
        "name": "confirmed_build" if confirmed else "normal",
        "build_direction": direction,
        "confirmed_build": confirmed,
        "oi_delta_15m": oi15,
        "price_return_15m": ret15,
        "money_flow_15m": flow15,
    }
