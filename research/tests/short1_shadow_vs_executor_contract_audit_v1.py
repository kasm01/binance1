"""Read-only direct test of actual SHORT executor stages on live V6 shadow output.

No new production intent, no fake upstream market_context_passed, no order.
"""
import json,redis
from collections import Counter
from core.trade_executor_short import TradeExecutorShort

r=redis.Redis(decode_responses=True,socket_timeout=2)
rows=r.xrevrange("telemetry:short1:inj:model_shadow_v6",count=480)
executor=TradeExecutorShort.__new__(TradeExecutorShort)
results=Counter()
for sid,raw in rows:
    e=json.loads(raw["json"])
    assert e["symbol"]=="INJUSDT" and e["routing"]=="OFF"
    normal=TradeExecutorShort._normalize_short_signal_stage(
        executor,symbol="INJUSDT",signal="short")
    assert normal["state"]=="READY"
    # There is no production MasterExecutor authorization in the shadow event.
    extra=dict(execution_source="shadow",model_provider=e["model_provider"],
               model_score=e["economic_probability"],
               model_score_gate_threshold=e["reference_threshold"],
               model_score_gate_passed=e["economic_probability"]>=e["reference_threshold"],
               model_score_gate_state="READY")
    context=TradeExecutorShort._build_short_market_context_stage(
        executor,probs={},extra=extra)
    gate=TradeExecutorShort._evaluate_short_model_score_gate_stage(
        executor,symbol="INJUSDT",market_context=context)
    results[gate["reason"]]+=1
    assert gate["state"]=="REJECT"
    assert gate["reason"]=="upstream_model_score_gate_required"
    # Even an unauthenticated 'orchestrated' tag can't bypass Market Context.
    adversarial=TradeExecutorShort._build_short_market_context_stage(
        executor,probs={},extra=dict(extra,execution_source="orchestrated"))
    assert adversarial["state"]=="REJECT" and adversarial["reason"]=="upstream_market_context_gate_missing"
print("SHORT_EXECUTOR_REAL_STAGE_AUDIT",len(rows),dict(results))
print("CONCLUSION: live shadow models have no actual orchestrated Master risk/context authority")
