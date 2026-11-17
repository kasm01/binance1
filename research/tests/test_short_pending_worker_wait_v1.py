import asyncio
from unittest.mock import patch
from core.trade_executor_short import TradeExecutorShort
class D:
 _env_bool=lambda self,n,default:True
 sniper_entry_interval='1m'
 sniper_entry_max_candle_progress=.42
 def _get_candle_id(self,*x):return 100
 def _calc_candle_progress(self,*x):return .20
 def _cancel_short_pending(self,**kw):
  self.cancelled=kw
  self._pending_short_entries.pop(kw['symbol'],None)
 async def _process_single_short_pending_entry(self,**kw):
  raise AssertionError('legacy pending worker must not use old model score')
for elapsed,expected in ((300000,False),(1000000,True)):
 d=D();d._pending_short_entries={'INJUSDT':{'origin_created_ms':1000000,'score':.57,'min_score':.55}}
 d.cancelled=None
 with patch('core.trade_executor_short.time.time',return_value=(1000000+elapsed)/1000):
  asyncio.run(TradeExecutorShort._process_short_pending_entries(d))
 assert bool(d.cancelled)==expected
print('PASS: old PENDING worker waits for fresh model, expires after 15min, never opens old score')
