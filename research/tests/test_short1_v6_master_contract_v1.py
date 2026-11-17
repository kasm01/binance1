"""Source completeness, model semantics and no spoofed executor authorization."""
from orchestration.shadow.short1_v6_master_contract_v1 import audit
t=1794606479999
def source():
 return dict(routing="OFF",symbol="INJUSDT",strategy_id="SHORT1",interval="3m",
   event_id="sig-real",model_provider="short_arm_quality_logistic_v6",
   model_feature_hash="hash",completed_close_time=t,economic_probability=.83,
   reference_threshold=.808,
   entry_market_context=dict(available=True,schema_version="entry_market_context.v1",
     price_response=dict(return_15m=-.002),money_flow=dict(flow_15m=-.13),
     open_interest=dict(available=True)),
   money_flow_60m=-.07)
a=source()
x=audit(a,now_ms=t+3000)
assert x["state"]=="BLOCK_CONTEXT" and x["reason"]=="missing_master_inputs"
assert {"price","whale_dir","whale_score","spread_pct","atr_pct","liq_score"}.issubset(x["missing"])
assert x["can_authorize_executor"] is False
assert audit(dict(a,economic_probability=.6),now_ms=t+3000)["state"]=="REJECT_MODEL"
assert audit(dict(a,model_provider="other"),now_ms=t+3000)["state"]=="REJECT_SOURCE"
assert audit(a,now_ms=t+9*60000)["state"]=="REJECT_MODEL"
assert audit(dict(a,entry_market_context=None),now_ms=t+3000)["state"]=="BLOCK_CONTEXT"
assert audit(dict(a,money_flow_60m=None),now_ms=t+3000)["state"]=="BLOCK_CONTEXT"
b=dict(a,**dict(price=15,whale_dir="none",whale_score=.2,
    spread_pct=.001,atr_pct=.01,liq_score=.8))
x=audit(b,now_ms=t+3000)
assert x["state"]=="AUDIT_ONLY_CONTEXT_COMPLETE" and not x["can_authorize_executor"]
print("PASS 7 conservative V6-to-Master admission cases, no routing/order permission")
