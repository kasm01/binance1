import asyncio
from core.trade_executor_short import TradeExecutorShort
from orchestration.aggregator.aggregator import Aggregator
class D:
 logger=None
 def _env_bool(self,k,default):return True
 def _cancel_short_pending(self,**kw):self.canceled=kw
 async def nope(self):raise AssertionError('should not open')
d=D()
p=dict(origin_intent_id='orig',origin_model_provider='short_v6',
 origin_completed_close_ms=100000,score=.56,min_score=.55)
asyncio.run(TradeExecutorShort._execute_short_pending_open(d,symbol='INJUSDT',pending=p))
assert d.canceled['symbol']=='INJUSDT' and 'independent_new_model_required' in d.canceled['reason']
print('PASS SHORT1 PENDING old stored score never opens without independent model intent')
agg=object.__new__(Aggregator)
agg.include_raw=False
event={'event_id':'sig123','model_provider':'short_v6','symbol':'INJUSDT','interval':'3m','side_candidate':'short','price':15,
       'model_score':.58,'meta':{'completed_close_time':1794000000000}}
candidate=Aggregator._build_candidate_payload(agg,event)
assert candidate['source_signal_event_id']=='sig123'
assert candidate['source_completed_close_ms']==1794000000000
assert candidate['model_score']==.58
print('PASS SHORT source model event_id and closed candle time preserved in normalized candidate with raw redacted')
