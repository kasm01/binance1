"""Research-only fault-matrix: full exact client ID/order fill/stop match."""
from orchestration.risk.bound_exchange_reconcile_v1 import bind
res=dict(request_id='i1',reservation_id='g1',strategy_id='SHORT1',symbol='INJUSDT')
led=dict(intent_id='i1',reservation_id='g1',owner='SHORT1',
    symbol='INJUSDT',side='SELL',client_order_id='cid1',
    stage='FILLED',original_qty='2',filled_qty='2',
    client_algo_id='b1s1_stop1')
order=dict(symbol='INJUSDT',clientOrderId='cid1',orderId=778,side='SELL',
    status='FILLED',origQty='2',executedQty='2')
fills=[dict(symbol='INJUSDT',orderId=778,id=1,qty='1'),
       dict(symbol='INJUSDT',orderId=778,id=2,qty='1')]
pos=[dict(symbol='INJUSDT',positionSide='SHORT',positionAmt='-2')]
stop=[dict(symbol='INJUSDT',clientAlgoId='b1s1_stop1',algoId=1001,
    side='BUY',positionSide='SHORT',orderType='STOP_MARKET',
    closePosition=True,workingType='MARK_PRICE',algoStatus='NEW')]
def trial(*,rr=None,ll=None,oo=None,ff=None,pp=None,normal=None,ss=None,fail=''):
 def read_order(sym,cid):
  if fail=='order':raise TimeoutError('test')
  return order if oo is None else oo
 def read_fills(sym,oid):
  if fail=='fill':raise TimeoutError('test')
  return fills if ff is None else ff
 def read_pos(sym):
  if fail=='position':raise TimeoutError('test')
  return pos if pp is None else pp
 def read_normal(sym):
  if fail=='openOrders':raise TimeoutError('test')
  return [] if normal is None else normal
 def read_algo(sym):
  if fail=='algo':raise TimeoutError('test')
  return stop if ss is None else ss
 return bind(reservation=res if rr is None else rr,
     ledger=led if ll is None else ll,read_order=read_order,
     read_fills=read_fills,read_positions=read_pos,
     read_open_orders=read_normal,read_open_algo=read_algo)
good=trial()
assert good['state']=='PROTECTED_EVIDENCE_ONLY'
assert not any(good[k] for k in ('can_commit','can_submit','can_release'))
assert good['source_binding_verified']
assert trial(ss=[])['state']=='FREEZE_UNPROTECTED'
assert trial(ss=[dict(stop[0],clientAlgoId='foreign')])['state']=='FREEZE_UNPROTECTED'
assert trial(ss=[dict(stop[0],workingType='CONTRACT_PRICE')])['state']=='FREEZE_UNPROTECTED'
assert trial(pp=[dict(pos[0],positionAmt='-1')])['state']=='HOLD_RECONCILIATION'
assert trial(pp=[])['state']=='HOLD_RECONCILIATION'
assert trial(normal=[dict(symbol='INJUSDT',orderId=89)])['state']=='HOLD_RECONCILIATION'
assert trial(ll=dict(led,reservation_id='g2'))['state']=='UNKNOWN_FREEZE'
assert trial(ll=dict(led,client_order_id='other'))['state']=='UNKNOWN_FREEZE'
assert trial(oo=dict(order,clientOrderId='other'))['state']=='UNKNOWN_FREEZE'
assert trial(oo=dict(order,status='PARTIALLY_FILLED',executedQty='1'))['state']=='HOLD_RECONCILIATION'
assert trial(ff=fills[:1])['state']=='UNKNOWN_FREEZE'
assert trial(ll=dict(led,filled_qty='1'))['state']=='UNKNOWN_FREEZE'
assert trial(ll=dict(led,stage='UNKNOWN',filled_qty='0'))['state']=='RECOVERED_FILL_PENDING_JOURNAL'
assert trial(fail='order')['state']=='UNKNOWN_FREEZE'
assert trial(fail='fill')['state']=='UNKNOWN_FREEZE'
assert trial(fail='position')['state']=='UNKNOWN_FREEZE'
assert trial(fail='openOrders')['state']=='UNKNOWN_FREEZE'
assert trial(fail='algo')['state']=='UNKNOWN_FREEZE'
assert trial(oo=dict(order,status='CANCELED',executedQty='0'))['state']=='HOLD_RECONCILIATION'
assert trial(pp=[],ss=[],ll=dict(led,stage='FLAT_RECONCILED',
  verified_exit_order_id='exit1',verified_exit_qty='2'))['state']=='FLAT_EVIDENCE_ONLY'
assert not trial(pp=[],ss=[],ll=dict(led,stage='FLAT_RECONCILED',
  verified_exit_order_id='exit1',verified_exit_qty='2'))['can_release']
assert trial(pp=[],ss=[],ll=dict(led,stage='FLAT_RECONCILED',
  verified_exit_order_id='exit1',verified_exit_qty='1'))['state']=='HOLD_RECONCILIATION'
print('PASS 23 exact order-fill-stop-identity/failure/closed-state cases; no broker mutation')
