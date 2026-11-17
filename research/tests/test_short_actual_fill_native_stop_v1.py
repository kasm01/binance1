from unittest.mock import patch
from core.trade_executor_short import TradeExecutorShort
class D:logger=None
real=dict(symbol='INJUSDT',positionSide='SHORT',positionAmt='-6.5',entryPrice='15.025')
with patch('core.trade_executor_short._futures_v3_position_risk',return_value=[real]) as x:
 o=TradeExecutorShort._verify_short_exchange_open_fill(D(),symbol='INJUSDT')
 assert o==dict(qty=6.5,entry_price=15.025),o
 assert x.call_count==1
with patch('core.trade_executor_short._futures_v3_position_risk',return_value=[
 dict(real,positionAmt='0')]),patch('core.trade_executor_short.time.sleep') as sleep:
 try:TradeExecutorShort._verify_short_exchange_open_fill(D(),symbol='INJUSDT')
 except RuntimeError as e:assert 'not_confirmed' in str(e)
 else:raise AssertionError('unfilled market order must NOT be treated as filled')
 assert sleep.call_count==4
try:TradeExecutorShort._verify_short_exchange_open_fill(D(),symbol='NEARUSDT')
except RuntimeError as e:assert 'unassigned_coin' in str(e)
else:raise AssertionError('wrong symbol accepted')
with patch('core.trade_executor_short._futures_v3_position_risk',side_effect=RuntimeError('api down')),patch('core.trade_executor_short.time.sleep'):
 try:TradeExecutorShort._verify_short_exchange_open_fill(D(),symbol='INJUSDT')
 except RuntimeError as e:assert 'not_confirmed' in str(e)
 else:raise AssertionError('failed exchange API accepted')
print('PASS: SHORT actual Binance fill qty+entry, unfilled/unknown/error fail-closed')
