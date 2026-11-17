"""Conservative V6->SHORT1 Master admission audit, with NO order authorization.

V6 economic score has a DIFFERENT target to the historic Master threshold.
Missing Master inputs remain missing, never silently normalized to neutral zero.
"""
from __future__ import annotations
import math
from typing import Any

PROVIDER="short_arm_quality_logistic_v6"
REQUIRED_MASTER_INPUTS=("price","whale_dir","whale_score","spread_pct",
                        "atr_pct","liq_score")

def _valid_num(v:Any)->bool:
    try:return math.isfinite(float(v))
    except (TypeError,ValueError):return False

def audit(record:dict[str,Any], *, now_ms:int)->dict[str,Any]:
    base=dict(symbol="INJUSDT",strategy_id="SHORT1",can_authorize_executor=False)
    if (not isinstance(record,dict) or record.get("routing")!="OFF" or
        record.get("symbol")!="INJUSDT" or record.get("strategy_id")!="SHORT1" or
        record.get("interval")!="3m" or record.get("model_provider")!=PROVIDER or
        not record.get("event_id") or not record.get("model_feature_hash")):
        return dict(base,state="REJECT_SOURCE",reason="source_identity_or_routing_invalid")
    try:
        close_ms=int(record["completed_close_time"])
        score=float(record["economic_probability"])
        threshold=float(record["reference_threshold"])
    except (ValueError,TypeError,KeyError):
        return dict(base,state="REJECT_MODEL",reason="invalid_model_values")
    if (not (0<=now_ms-close_ms<=8*60000) or
        not (0<=score<=1 and 0<threshold<=1) or
        not math.isfinite(score) or not math.isfinite(threshold)):
        return dict(base,state="REJECT_MODEL",reason="stale_or_invalid_economic_probability")
    if score<threshold:
        return dict(base,state="REJECT_MODEL",reason="v6_own_economic_threshold",score=score,threshold=threshold)
    ctx=record.get("entry_market_context")
    if not isinstance(ctx,dict) or ctx.get("available") is not True or ctx.get("schema_version")!="entry_market_context.v1":
        return dict(base,state="BLOCK_CONTEXT",reason="missing_entry_context")
    price_resp=ctx.get("price_response") if isinstance(ctx.get("price_response"),dict) else {}
    flow=ctx.get("money_flow") if isinstance(ctx.get("money_flow"),dict) else {}
    oi=ctx.get("open_interest") if isinstance(ctx.get("open_interest"),dict) else {}
    if not _valid_num(price_resp.get("return_15m")) or not _valid_num(flow.get("flow_15m")) or not _valid_num(record.get("money_flow_60m")):
        return dict(base,state="BLOCK_CONTEXT",reason="missing_15_60m_flow_or_price")
    # The existing Master uses these fields in veto/risk decisions.
    # Their absence MUST NOT be turned into whale_dir='none', spread=0, etc.
    missing=[key for key in REQUIRED_MASTER_INPUTS if record.get(key) is None]
    if missing:
        return dict(base,state="BLOCK_CONTEXT",reason="missing_master_inputs",missing=missing,
                    v6_score=score,v6_threshold=threshold,source_event_id=record["event_id"],
                    completed_close_ms=close_ms,
                    flow15=float(flow["flow_15m"]),flow60=float(record["money_flow_60m"]),
                    price_return15=float(price_resp["return_15m"]),oi_available=bool(oi.get("available")))
    return dict(base,state="AUDIT_ONLY_CONTEXT_COMPLETE",reason="requires_canonical_master_and_executor",
                v6_score=score,v6_threshold=threshold,
                source_event_id=record["event_id"],completed_close_ms=close_ms)
