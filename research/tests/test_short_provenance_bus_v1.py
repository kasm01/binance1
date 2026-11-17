"""No Redis write, mock only: event -> candidate -> intent -> exec event provenance."""
import json
from orchestration.aggregator.aggregator import Aggregator
from orchestration.executor.master_executor import MasterExecutor,TradeIntent
from orchestration.executor.intent_bridge import IntentBridge

a=Aggregator.__new__(Aggregator)
a.include_raw=False
signal=dict(event_id='sig-model-2026',symbol='INJUSDT',interval='3m',
  side_candidate='short',price=15.5,model_score=.577,model_provider='short_v6',
  meta=dict(completed_close_time=1790000000000))
candidate=a._build_candidate_payload(signal)
assert candidate['source_signal_event_id']=='sig-model-2026'
assert candidate['source_completed_close_ms']==1790000000000
class FakeRedis:
 def __init__(self):self.written=[]
 def xadd(self,stream,payload,**kw):
  obj=json.loads(payload['json']);self.written.append((stream,obj));return '123-0'
r=FakeRedis()
m=MasterExecutor.__new__(MasterExecutor)
m.publish_allowed=True;m._last_published_source_id='';m.r=r;m.out_stream='intent_mock'
m.min_trade_score=.555;m.short_soft_score_min=.53
it=TradeIntent(intent_id='intent-new',ts_utc='now',symbol='INJUSDT',interval='3m',
   side='short',price=15.5,score=.577,ranking_score=.577,
   reasons=[],risk_tags=[],raw={**candidate,'model_score_gate_passed':True,
   'model_score_gate_threshold':.555,'model_score_gate_state':'READY'})
assert m._publish_intents('top-1',[it])=='123-0'
item=r.written[-1][1]['items'][0]
assert item['source_signal_event_id']=='sig-model-2026'
assert item['source_completed_close_ms']==1790000000000
b=IntentBridge.__new__(IntentBridge)
b.publish_allowed=True;b.out_stream='exec_mock';b.r=r;b.pass_raw=False;b.out_maxlen=5000;b.out_approx=True
b._norm_side=lambda i:'short'
b._price_of=lambda i:float(i.get('price') or 0)
b.short_micro_gate_fail_open=True
# New strict SHORT1 gate refuses old or research-only model packets.
assert b._publish_exec('intent-1',item) is None
assert len(r.written)==1
# Non-SHORT1 bridge transport still preserves provenance.
other=dict(item,symbol='SOLUSDT')
assert b._publish_exec('intent-1',other)=='123-0'
exec_item=r.written[-1][1]['items'][0]
assert exec_item['source_signal_event_id']=='sig-model-2026'
assert exec_item['source_completed_close_ms']==1790000000000
assert exec_item['intent_id']=='intent-new'
print('PASS SHORT1 unauthorised intent blocked, unrelated transport origin preserved')
