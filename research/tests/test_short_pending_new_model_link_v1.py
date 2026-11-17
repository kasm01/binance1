from unittest.mock import patch
from core.trade_executor_short import TradeExecutorShort
from core.short_pending_freshness_contract import check_independent_model_revalidation

ORIGIN=1000000
class D:
    logger=None
    _env_bool=lambda self,k,d:True
    def __init__(self):self._pending_short_entries={'INJUSDT':dict(
      origin_intent_id='orig1',origin_source_event_id='sig1',
      origin_completed_close_ms=ORIGIN,origin_model_provider='short_v6',
      origin_created_ms=1002000,score=.58,min_score=.555)}
    def _pop_short_pending(self,sym):return self._pending_short_entries.pop(sym,None)
    def _cancel_short_pending(self,**kw):
      self._pending_short_entries.pop(kw['symbol'],None);self.cancelled=kw
fresh=dict(intent_id='new2',source_signal_event_id='sig2',source_completed_close_ms=ORIGIN+180000,
  model_provider='short_v6',model_score=.59,model_score_gate_threshold=.555,
  model_score_gate_passed=True)
with patch('core.trade_executor_short.time.time',return_value=(ORIGIN+190000)/1000):
 d=D();bad=TradeExecutorShort._resolve_short_pending_by_fresh_model(d,symbol='INJUSDT',upstream=dict(fresh,intent_id='orig1'),score_state='READY')
 assert bad['state']=='PENDING' and 'INJUSDT' in d._pending_short_entries
 d=D();bad=TradeExecutorShort._resolve_short_pending_by_fresh_model(d,symbol='INJUSDT',upstream=dict(fresh,source_signal_event_id='sig1'),score_state='READY')
 assert bad['state']=='PENDING' and 'INJUSDT' in d._pending_short_entries
 d=D();bad=TradeExecutorShort._resolve_short_pending_by_fresh_model(d,symbol='INJUSDT',upstream=fresh,score_state='PENDING_CONTEXT')
 assert bad['state']=='PENDING'
 d=D();ok=TradeExecutorShort._resolve_short_pending_by_fresh_model(d,symbol='INJUSDT',upstream=fresh,score_state='READY')
 assert ok['state']=='READY_NEW_INTENT' and not d._pending_short_entries
 d=D();bad=TradeExecutorShort._resolve_short_pending_by_fresh_model(d,symbol='INJUSDT',upstream=dict(fresh,model_score=.50),score_state='READY')
 assert bad['state']=='PENDING'
print('PASS short pending native entry linkage (same intent/model event, soft state, fresh ready, below-score)')
