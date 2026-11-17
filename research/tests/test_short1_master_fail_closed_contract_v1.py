"""SHORT1 shadow->Master hard denial contract, no Binance calls or Redis writes."""
from core.trade_executor_short import TradeExecutorShort
from orchestration.shadow.short1_v6_master_contract_v1 import audit

e=TradeExecutorShort.__new__(TradeExecutorShort)
base=dict(execution_source='shadow',model_provider='short_arm_quality_logistic_v6',
    model_score=.91,model_score_gate_threshold=.81,
    model_score_gate_passed=True,model_score_gate_state='READY',
    market_context_passed=True)
# Fake READY and fake market_context_passed may not authorize a shadow source.
ctx=e._build_short_market_context_stage(probs={},extra=base)
result=e._evaluate_short_model_score_gate_stage(symbol='INJUSDT',market_context=ctx)
assert result['state']=='REJECT' and result['reason']=='upstream_model_score_gate_required'
# If someone changes ONLY execution_source, upstream pass can be asserted by
# arbitrary dictionaries: strict provenance MUST be implemented before promotion.
spoof=dict(base,execution_source='orchestrated')
ctx2=e._build_short_market_context_stage(probs={},extra=spoof)
result2=e._evaluate_short_model_score_gate_stage(symbol='INJUSDT',market_context=ctx2)
assert ctx2['state']=='READY' and result2['state']=='READY'
# This is a finding: these methods trust upstream flags; they are NOT a
# complete authenticity validator. Do not invoke with user/provider payload.
# Incomplete orchestrated market context must reject.
ctx3=e._build_short_market_context_stage(probs={},extra=dict(spoof,market_context_passed=False))
assert ctx3['state']=='REJECT'
def src():
    return dict(routing='OFF',symbol='INJUSDT',strategy_id='SHORT1',interval='3m',
                model_provider='short_arm_quality_logistic_v6',
                event_id='test',model_feature_hash='testhash',
                completed_close_time=1794650400000,
                economic_probability=.91,reference_threshold=.81,
                money_flow_60m=-.12,
                entry_market_context=dict(schema_version='entry_market_context.v1',
                    available=True,price_response=dict(return_15m=-.002),
                    money_flow=dict(flow_15m=-.01)))
s=src()
a=audit(s,now_ms=s['completed_close_time']+7000)
assert a['state']=='BLOCK_CONTEXT' and a['reason']=='missing_master_inputs'
assert not a['can_authorize_executor']
assert audit(s,now_ms=s['completed_close_time']+9*60000)['state']=='REJECT_MODEL'
assert audit(dict(s,economic_probability=.3),now_ms=s['completed_close_time']+7000)['state']=='REJECT_MODEL'
assert audit(dict(s,model_provider='unknown'),now_ms=s['completed_close_time']+7000)['state']=='REJECT_SOURCE'
print('PASS 8 fail-closed/source/provenance checks')
print('SECURITY_FINDING: spoofed orchestrated upstream pass flags are accepted by downstream stage methods; upstream caller must be trusted/authenticated before any production routing.')
