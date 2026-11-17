#!/usr/bin/env python3
"""SHORT1 close proof: never cancel own stop before Binance flat confirmation."""
from unittest.mock import patch
from core.trade_executor_short import TradeExecutorShort
class Dummy:
    dry_run=False;logger=None;central_risk_client=None
    def _validate_short_close_request(self,*,symbol):
        return {'ok':True,'position':{'entry_price':15.0,'native_stop_order_id':42,'qty':2},'qty':2}
    def _resolve_price(self,*,symbol,price):return 14.9
    def _exchange_close_market(self,*,symbol,qty):return {'status':'NEW','orderId':101}
    def _cancel_native_short_stop(self,*,symbol,position):
        self.canceled+=1
    def _finalize_short_close(self,**kw):
        self.finalized+=1
        return {'status':'closed'}
    def _verify_short_exchange_flat(self,symbol):
        return self.confirm_flat
for flat in (False,True):
 o=Dummy();o.canceled=o.finalized=0;o.confirm_flat=flat
 with patch('core.trade_executor_short.time.sleep') as sleep:
  result=TradeExecutorShort.close_position(o,symbol='INJUSDT')
 assert (o.canceled,o.finalized)==((1,1) if flat else (0,0))
 assert result['status']==('closed' if flat else 'pending_exchange_flat_verification')
 print('PASS_CLOSE', 'flat_confirmed_then_cancel' if flat else 'still_open_stop_retained', 'sleep_calls',sleep.call_count)
with patch('core.trade_executor_short._futures_v3_position_risk',return_value=[
 {'symbol':'INJUSDT','positionSide':'SHORT','positionAmt':'-2'}]):
 assert not TradeExecutorShort._verify_short_exchange_flat(Dummy(),'INJUSDT')
with patch('core.trade_executor_short._futures_v3_position_risk',return_value=[
 {'symbol':'INJUSDT','positionSide':'SHORT','positionAmt':'0'}]):
 assert TradeExecutorShort._verify_short_exchange_flat(Dummy(),'INJUSDT')
with patch('core.trade_executor_short._futures_v3_position_risk',side_effect=RuntimeError('timeout')):
 assert not TradeExecutorShort._verify_short_exchange_flat(Dummy(),'INJUSDT')
print('ALL_SHORT_CLOSE_PROTECTION_TESTS_OK')
