from unittest.mock import patch
from decimal import Decimal
from scripts import short1_native_stop_reconciler as s
from core import short_native_algo_stop as a

def pos(qty='-5',entry='15',mark='14',lev='3'):
 return dict(symbol='INJUSDT',positionSide='SHORT',positionAmt=qty,entryPrice=entry,markPrice=mark,leverage=lev)
class Resp:
 def json(self):return dict(symbols=[dict(symbol='INJUSDT',filters=[dict(filterType='PRICE_FILTER',tickSize='0.001')])])
assert s.expected_native_stop('15','3','0.001','-0.0897')==Decimal('15.448')
# Backwards-compatible historical calculation check, not current production setting.
assert s.expected_native_stop('15','3','0.001','-0.12')==Decimal('15.600')
with patch('scripts.short1_native_stop_reconciler.position_risk',return_value=[]),\
 patch('scripts.short1_native_stop_reconciler.get_open',return_value=[]):
 assert s.state()['status']=='FLAT_CLEAN'
with patch('scripts.short1_native_stop_reconciler.position_risk',return_value=[pos()]),\
 patch('scripts.short1_native_stop_reconciler.get_open',return_value=[]),\
 patch('scripts.short1_native_stop_reconciler.requests.get',return_value=Resp()),\
 patch('scripts.short1_native_stop_reconciler.place') as post:
 assert s.state()['status']=='MISSING_STOP' and not post.called
try:s.state(apply=True)
except RuntimeError as e:assert 'DRY_RUN_ACTIVE' in str(e)
else:raise AssertionError('dry run apply must block before write')
valid=dict(symbol='INJUSDT',side='BUY',positionSide='SHORT',orderType='STOP_MARKET',closePosition=True,triggerPrice='15.448',clientAlgoId='b1s1_a1',algoId=123,algoStatus='NEW')
with patch('scripts.short1_native_stop_reconciler.position_risk',return_value=[pos()]),\
 patch('scripts.short1_native_stop_reconciler.get_open',return_value=[valid]),\
 patch('scripts.short1_native_stop_reconciler.requests.get',return_value=Resp()),\
 patch('scripts.short1_native_stop_reconciler.place') as post:
 assert s.state()['status']=='PROTECTED' and not post.called
with patch('scripts.short1_native_stop_reconciler.position_risk',return_value=[pos()]),\
 patch('scripts.short1_native_stop_reconciler.get_open',return_value=[dict(valid,clientAlgoId='foreign')]),\
 patch('scripts.short1_native_stop_reconciler.requests.get',return_value=Resp()),\
 patch('scripts.short1_native_stop_reconciler.place') as post:
 assert s.state()['status']=='MANUAL_REVIEW_ALGO_EXISTS' and not post.called
print('PASS 5 SHORT1 native stop reconciler tests: flat/missing/dry safety/protected/foreign')
