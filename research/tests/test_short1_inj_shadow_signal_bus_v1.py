"""Pure tests: no real Redis, no Binance, no production stream writes."""
import os
from unittest.mock import patch
from orchestration.shadow.short1_inj_shadow_signal_bus_v1 import process,input_valid,construct_model_candidate,check_shadow_only
bar=1794603959999
def event(version,close,ev,score,confirm='NOT_DOWN'):
 return {
 'event_id':ev,'strategy_id':'SHORT1','symbol':'INJUSDT','interval':'3m',
 'routing':'OFF','completed_close_time':close,
 'model_provider':'short_arm_quality_logistic_v6' if version=='v6' else 'short_entry_quality_fresh_logistic_v5',
 'model_feature_hash':'hash-'+version,'economic_probability':score,
 'reference_threshold':0.8 if version=='v6' else .52,
 'sniper_progress':.20,'sniper_min':.1,'sniper_max':.42,
 'confirmation_state':confirm}
v5=event('v5',bar,'v5a',.54)
v6=event('v6',bar,'v6a',.82)
assert input_valid(v6,version='v6') and input_valid(v5,version='v5')
a,s=process(v6,v5,{},now_ms=bar+5000)
assert a['decision']=='SHADOW_PENDING' and s['pending']['source_signal_event_id']=='v6a'
assert a['target_routing']=='SHADOW_ONLY'
v5b=event('v5',bar+180000,'v5b',.55)
v6b=event('v6',bar+180000,'v6b',.84,confirm='CONFIRM')
b,s2=process(v6b,v5b,s,now_ms=bar+180000+5000)
assert b['decision']=='SHADOW_READY_FRESH_MODEL' and s2['pending'] is None
assert a['intent_id']!=b['intent_id']
bad=event('v6',bar+180000,'v6a',.84,confirm='CONFIRM')
c,s3=process(bad,v5b,s,now_ms=bar+180000+5000)
assert c['decision']=='SHADOW_PENDING' and s3['pending']
low6=event('v6',bar,'v6low',.7)
r,_=process(low6,v5,{},now_ms=bar+5000)
assert r['decision']=='SHADOW_REJECT'
d,_=process(v6b,v5b,{},now_ms=bar+180000+5000)
assert d['decision']=='SHADOW_READY_DIRECT'
try:process(v6b,v5,s,now_ms=bar+180000+5000)
except ValueError as x:assert 'mismatch' in str(x)
else:raise AssertionError('mixed-model closed bars')
try:process(v6b,v5b,s,now_ms=bar+180000+15*60000)
except ValueError as x:assert 'fresh' in str(x)
else:raise AssertionError('stale model never accepted')
with patch.dict(os.environ,{'DRY_RUN':'0','ARMED':'0','SYMBOL':'INJUSDT','STRATEGY_ID':'SHORT1'}):
 try:check_shadow_only()
 except RuntimeError:pass
 else:raise AssertionError('LIVE mode must be prohibited')
with patch.dict(os.environ,{'DRY_RUN':'1','ARMED':'1','SYMBOL':'INJUSDT','STRATEGY_ID':'SHORT1'}):
 try:check_shadow_only()
 except RuntimeError:pass
 else:raise AssertionError('ARMED mode must be prohibited')
print('PASS 9 SHORT shadow FSM safety scenarios: unique fresh model, direct, stale, crossed bars, no live/armed')
