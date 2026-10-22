from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict

from orchestration.context.entry_context_snapshot import (
    build_entry_oi_regime,
    completed_1m_metrics,
)
from orchestration.risk.risk_master import resolve_market_context_multiplier


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


def _decode_json(raw: Any) -> Dict[str, Any]:
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


def _regime(context: Any) -> Dict[str, str]:
    ctx = context if isinstance(context, dict) else {}
    reg = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
    return {
        "name": str(reg.get("name") or "normal").strip().lower(),
        "build_direction": str(reg.get("build_direction") or "none").strip().lower(),
    }


def compare_contexts(master_context: Any, entry_context: Any) -> Dict[str, Any]:
    """Compare classification only; never alter execution state."""
    m = _regime(master_context)
    ectx = entry_context if isinstance(entry_context, dict) else {}
    if not bool(ectx.get("available")):
        return {
            "available": False,
            "changed": None,
            "transition": "entry_context_unavailable",
            "master_regime": m,
            "entry_regime": {"name": "unavailable", "build_direction": "none"},
        }
    e = _regime(ectx)
    mn, en = m["name"], e["name"]
    md, ed = m["build_direction"], e["build_direction"]
    if mn == en and (mn != "confirmed_build" or md == ed):
        transition = "same_confirmed_build" if mn == "confirmed_build" else "same_normal"
        changed = False
    elif mn == "normal" and en == "confirmed_build":
        transition = "normal_to_confirmed_build"
        changed = True
    elif mn == "confirmed_build" and en == "normal":
        transition = "confirmed_build_to_normal"
        changed = True
    elif mn == "confirmed_build" and en == "confirmed_build" and md != ed:
        transition = "confirmed_build_direction_flip"
        changed = True
    else:
        transition = f"{mn}_to_{en}"
        changed = mn != en or md != ed
    return {
        "available": True,
        "changed": bool(changed),
        "transition": transition,
        "master_regime": m,
        "entry_regime": e,
    }


def build_shadow_event(
    *,
    client: Any,
    redis_client: Any,
    symbol: str,
    master_context: Any,
    confirmation_open_ms: int,
    model_score: float = 0.0,
    intent_id: str = "",
    entry_path: str = "direct",
    oi_policy_multiplier: float = 0.50,
    kline_rows: Any = None,
    entry_context_override: Any = None,
) -> Dict[str, Any]:
    """Fetch entry-time completed context after order/attempt using a frozen cutoff.

    The function is telemetry-only. Exceptions are returned as unavailable events
    rather than raised into the execution pipeline.
    """
    sym = str(symbol or "").upper().strip()
    event: Dict[str, Any] = {
        "schema_version": "market_context_shadow.v1",
        "event_id": str(uuid.uuid4()),
        "ts": float(time.time()),
        "symbol": sym,
        "intent_id": str(intent_id or ""),
        "entry_path": str(entry_path or "direct"),
        "model_score": float(model_score or 0.0),
        "confirmation_open_ms": _i(confirmation_open_ms, 0),
        "master_context": master_context if isinstance(master_context, dict) else {},
    }
    try:
        override = entry_context_override if isinstance(entry_context_override, dict) else {}
        if bool(override.get("available")):
            entry_context = dict(override)
            one = {}
            event["entry_context_source"] = "preorder_exact_snapshot"
            event["kline_rows_source"] = "confirmation_reuse" if isinstance(kline_rows, list) else "not_required"
            event["kline_rows_count"] = len(kline_rows) if isinstance(kline_rows, list) else 0
        else:
            rows = list(kline_rows) if isinstance(kline_rows, list) else None
            rows_source = "confirmation_reuse"
            if not isinstance(rows, list) or len(rows) < 17:
                rows = client.futures_klines(symbol=sym, interval="1m", limit=20)
                rows_source = "postorder_fallback_fetch"
            one = completed_1m_metrics(
                rows,
                confirmation_open_ms=_i(confirmation_open_ms, 0),
            )
            event["entry_context_source"] = "shadow_rebuilt"
            event["kline_rows_source"] = rows_source
            event["kline_rows_count"] = len(rows) if isinstance(rows, list) else 0
            raw_oi = redis_client.get(f"context:oi:{sym}") if redis_client is not None else None
            oi = _decode_json(raw_oi)
            entry_context = build_entry_oi_regime(one_minute=one, oi_context=oi)
        comparison = compare_contexts(master_context, entry_context)

        master_mult, _ = resolve_market_context_multiplier(
            market_context=master_context,
            oi_confirmed_build_enable=True,
            oi_confirmed_build_multiplier=oi_policy_multiplier,
        )
        entry_mult, _ = resolve_market_context_multiplier(
            market_context=entry_context,
            oi_confirmed_build_enable=True,
            oi_confirmed_build_multiplier=oi_policy_multiplier,
        )
        event.update({
            "available": bool(entry_context.get("available")),
            "one_minute": one,
            "entry_context": entry_context,
            "comparison": comparison,
            "hypothetical_risk": {
                "oi_confirmed_build_multiplier": float(oi_policy_multiplier),
                "master_context_multiplier": float(master_mult),
                "entry_context_multiplier": float(entry_mult),
                "sizing_signal_changed": bool(abs(float(master_mult) - float(entry_mult)) > 1e-12),
            },
        })
    except Exception as exc:
        event.update({
            "available": False,
            "error": str(exc)[:240],
            "comparison": {
                "available": False,
                "changed": None,
                "transition": "shadow_exception",
            },
        })
    return event


def publish_shadow_event(
    *,
    redis_client: Any,
    event: Dict[str, Any],
    stream: str = "telemetry:market_context_shadow",
    maxlen: int = 20000,
) -> str | None:
    if redis_client is None or not isinstance(event, dict):
        return None
    try:
        return redis_client.xadd(
            str(stream),
            {"json": json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)},
            maxlen=max(1000, int(maxlen)),
            approximate=True,
        )
    except Exception:
        return None
