from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from orchestration.context.market_context_contract import classify_confirmed_position_build


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def completed_1m_metrics(
    rows: Any,
    *,
    now_ms: int | None = None,
    confirmation_open_ms: int | None = None,
) -> Dict[str, Any]:
    """Build V23-style metrics from completed 1m Binance kline rows.

    Expected Binance row indexes:
      0 open_time, 4 close, 6 close_time, 7 quote volume,
      10 taker-buy quote volume.

    The currently forming minute is excluded. 16 completed closes are needed
    for a 15-minute close-to-close return; the last 15 completed bars are used
    for taker-flow.
    """
    if not isinstance(rows, list):
        return {"available": False, "reason": "rows_not_list"}
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    valid: List[Any] = [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 11]
    if not valid:
        return {"available": False, "reason": "no_valid_rows"}

    explicit_cutoff = _i(confirmation_open_ms, 0)
    if explicit_cutoff > 0:
        # Shadow can run after ORDER without moving the measurement window.
        # Keep only candles whose open time is strictly before the entry
        # confirmation minute captured by the executor.
        forming_open_ms = explicit_cutoff
        valid = [r for r in valid if _i(r[0], 0) < explicit_cutoff]
    else:
        forming_open_ms = 0
        last = valid[-1]
        last_close_ms = _i(last[6], 0)
        if last_close_ms >= now:
            forming_open_ms = _i(last[0], 0)
            valid = valid[:-1]
        else:
            forming_open_ms = (now // 60_000) * 60_000

    if len(valid) < 16:
        return {
            "available": False,
            "reason": "insufficient_completed_rows",
            "completed_rows": len(valid),
            "confirmation_open_ms": forming_open_ms,
        }

    latest_close = _f(valid[-1][4], 0.0)
    prior_close = _f(valid[-16][4], 0.0)
    if latest_close <= 0.0 or prior_close <= 0.0:
        return {"available": False, "reason": "invalid_close"}

    qsum = 0.0
    netsum = 0.0
    for row in valid[-15:]:
        quote = _f(row[7], 0.0)
        taker_buy_quote = _f(row[10], 0.0)
        if quote < 0.0 or taker_buy_quote < 0.0:
            continue
        qsum += quote
        netsum += (2.0 * taker_buy_quote) - quote
    if qsum <= 0.0:
        return {"available": False, "reason": "invalid_quote_volume"}

    return {
        "available": True,
        "reason": "completed_1m_context",
        "confirmation_open_ms": forming_open_ms,
        "last_completed_open_ms": _i(valid[-1][0], 0),
        "price_return_15m": float(latest_close / prior_close - 1.0),
        "money_flow_15m": float(netsum / qsum),
    }


def strictly_prior_oi_snapshot(oi_context: Any, *, cutoff_open_ms: int) -> Dict[str, Any]:
    """Select an OI snapshot strictly older than the confirmation candle open."""
    ctx = oi_context if isinstance(oi_context, dict) else {}
    cutoff = int(cutoff_open_ms or 0)
    latest_ts = _i(ctx.get("source_ts_ms"), 0)
    prev_ts = _i(ctx.get("previous_source_ts_ms"), 0)

    if latest_ts > 0 and (cutoff <= 0 or latest_ts < cutoff):
        return {
            "available": True,
            "source_ts_ms": latest_ts,
            "sum_open_interest": _f(ctx.get("sum_open_interest"), 0.0),
            "oi_delta_5m": _f(ctx.get("oi_delta_5m"), 0.0),
            "oi_delta_15m": _f(ctx.get("oi_delta_15m"), 0.0),
            "selection": "latest_strictly_prior",
        }
    if prev_ts > 0 and (cutoff <= 0 or prev_ts < cutoff):
        return {
            "available": True,
            "source_ts_ms": prev_ts,
            "sum_open_interest": _f(ctx.get("previous_sum_open_interest"), 0.0),
            "oi_delta_5m": _f(ctx.get("previous_oi_delta_5m"), 0.0),
            "oi_delta_15m": _f(ctx.get("previous_oi_delta_15m"), 0.0),
            "selection": "previous_boundary_fallback",
        }
    return {"available": False, "selection": "no_strictly_prior_snapshot"}


def build_entry_oi_regime(*, one_minute: Any, oi_context: Any) -> Dict[str, Any]:
    """Combine completed 1m metrics and causal OI into a context-only regime."""
    m = one_minute if isinstance(one_minute, dict) else {}
    if not m.get("available"):
        return {"available": False, "reason": str(m.get("reason") or "missing_1m_context")}
    oi = strictly_prior_oi_snapshot(
        oi_context,
        cutoff_open_ms=_i(m.get("confirmation_open_ms"), 0),
    )
    if not oi.get("available"):
        return {"available": False, "reason": "missing_causal_oi", "open_interest": oi}
    regime = classify_confirmed_position_build(
        oi_delta_15m=oi.get("oi_delta_15m", 0.0),
        price_return_15m=m.get("price_return_15m", 0.0),
        money_flow_15m=m.get("money_flow_15m", 0.0),
    )
    return {
        "available": True,
        "schema_version": "entry_market_context.v1",
        "snapshot_stage": "preorder_completed_1m",
        "confirmation_open_ms": _i(m.get("confirmation_open_ms"), 0),
        "price_response": {"return_15m": _f(m.get("price_return_15m"), 0.0)},
        "money_flow": {"flow_15m": _f(m.get("money_flow_15m"), 0.0)},
        "open_interest": oi,
        "regime": {
            "name": regime["name"],
            "build_direction": regime["build_direction"],
        },
    }


def _decode_json_dict(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def build_entry_context_from_rows(
    *,
    redis_client: Any,
    symbol: str,
    rows: Any,
    confirmation_open_ms: int,
) -> Dict[str, Any]:
    """Build canonical entry_market_context.v1 with no market-data I/O.

    `rows` must be the exact 1m rows already used by ENTRY CONFIRMATION.
    OI is read from the central Redis context collector. Missing data is
    represented explicitly and remains neutral at the Risk Master layer.
    """
    one = completed_1m_metrics(
        rows,
        confirmation_open_ms=int(confirmation_open_ms or 0),
    )
    raw_oi = None
    try:
        if redis_client is not None:
            raw_oi = redis_client.get(f"context:oi:{str(symbol or '').upper().strip()}")
    except Exception:
        raw_oi = None
    oi_context = _decode_json_dict(raw_oi)
    out = build_entry_oi_regime(one_minute=one, oi_context=oi_context)
    if isinstance(out, dict):
        out["symbol"] = str(symbol or "").upper().strip()
        out["source"] = "entry_confirmation_reuse"
    return out
