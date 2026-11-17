"""Broker exposure reconciliation: explicit unknown/blocked/flat scenarios."""
from orchestration.risk.exchange_flat_readonly_reconcile_v1 import evaluate_symbol,read_only_snapshot
def p(n,side='BOTH'):
 return [dict(symbol='NEARUSDT',positionSide=side,positionAmt=str(n))]
def order():
 return [dict(symbol='NEARUSDT',orderId=123,clientOrderId='abc')]
def algo():
 return [dict(symbol='NEARUSDT',algoId=456,clientAlgoId='b1l1_abc',orderType='STOP_MARKET')]
assert evaluate_symbol('NEARUSDT',p(0),[],[])['state']=='FLAT_CANDIDATE'
assert evaluate_symbol('NEARUSDT',p(1),[],[])['state']=='BLOCK_RELEASE'
assert evaluate_symbol('NEARUSDT',p(0),order(),[])['state']=='BLOCK_RELEASE'
assert evaluate_symbol('NEARUSDT',p(0),[],algo())['state']=='BLOCK_RELEASE'
assert evaluate_symbol('NEARUSDT',p(0),order(),algo())['state']=='BLOCK_RELEASE'
assert evaluate_symbol('NEARUSDT',p(float('nan')),[],[])['state']=='UNKNOWN'
assert evaluate_symbol('NEARUSDT',[dict(symbol='NEARUSDT',positionAmt='0')],[],[])['state']=='UNKNOWN'
assert evaluate_symbol('NEARUSDT',p(0),{},[])['state']=='UNKNOWN'
assert evaluate_symbol('NEARUSDT',p(0),[],[dict(symbol='NEARUSDT')])['state']=='UNKNOWN'
assert evaluate_symbol('NEARUSDT',[dict(symbol='SOLUSDT',positionAmt='0',positionSide='BOTH')],[],[])['state']=='UNKNOWN'
assert evaluate_symbol('SOLUSDT',[],[],[])['state']=='UNKNOWN'
def pos(s):return [dict(symbol=s,positionAmt='0',positionSide='BOTH')]
def empty(s):return []
summary=read_only_snapshot(pos,empty,empty)
assert summary['overall_state']=='FLAT_CANDIDATE' and not summary['safe_to_auto_release']
def fail(s):raise ConnectionError('simulated exchange outage')
assert read_only_snapshot(pos,empty,fail)['overall_state']=='BLOCK_OR_UNKNOWN'
assert read_only_snapshot(pos,empty,fail)['symbols']['INJUSDT']['state']=='UNKNOWN'
def short_position(s):
 return [dict(symbol=s,positionAmt='-3' if s=='INJUSDT' else '0',positionSide='SHORT' if s=='INJUSDT' else 'BOTH')]
assert read_only_snapshot(short_position,empty,empty)['overall_state']=='BLOCK_OR_UNKNOWN'
print('PASS 15 read-only broker-flat, pending order, algo, malformed, outage, unauthorized release tests')
