"""Actual isolated Redis reservation+intent and separately mocked signed GETs.

Never submits/commits/releases, never contacts real Binance or production Redis.
"""
import json,os
from tempfile import TemporaryDirectory
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.bound_exchange_reconcile_v1 import bind
with TemporaryDirectory(prefix='risk_bound_exchange_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  m=IsolatedAtomicReservePrepare(r)
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  v=m.act('RESERVE_PREPARE',request_id='test-i',generation='g11',
    strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='b1s1-test-i',now_ms=1000,equity=100,
    original_qty=2,requested_notional=46)
  assert v['state']=='READY'
  reservation=json.loads(r.hget(m.keys[1],'SHORT1'))
  stored=json.loads(r.hget(m.keys[4],'test-i'))
  # Mock actual Binance position/order/fills/Algo REST shapes.
  order=dict(symbol='INJUSDT',clientOrderId='b1s1-test-i',orderId=12,
      side='SELL',origQty='2',executedQty='2',status='FILLED')
  fills=[dict(symbol='INJUSDT',orderId=12,id=1,qty='1'),
         dict(symbol='INJUSDT',orderId=12,id=2,qty='1')]
  pos=[dict(symbol='INJUSDT',positionAmt='-2',positionSide='SHORT')]
  algo=[dict(symbol='INJUSDT',clientAlgoId='b1s1-stop-g11',
    algoId=99,side='BUY',positionSide='SHORT',orderType='STOP_MARKET',
    closePosition=True,workingType='MARK_PRICE',algoStatus='NEW')]
  def audit(l,p=pos,a=algo):
    return bind(reservation=reservation,ledger=l,
      read_order=lambda sym,client:order,read_fills=lambda sym,order_id:fills,
      read_positions=lambda sym:p,read_open_orders=lambda sym:[],
      read_open_algo=lambda sym:a)
  # Model is PREPARED: can never imply a broker submission happened.
  assert audit(stored)['state']=='UNKNOWN_FREEZE'
  recovered=dict(stored,stage='UNKNOWN')
  assert audit(recovered)['state']=='RECOVERED_FILL_PENDING_JOURNAL'
  protected=dict(stored,stage='FILLED',filled_qty='2',client_algo_id='b1s1-stop-g11')
  assert audit(protected)['state']=='PROTECTED_EVIDENCE_ONLY'
  assert audit(protected,a=[])['state']=='FREEZE_UNPROTECTED'
  assert audit(protected,p=[])['state']=='HOLD_RECONCILIATION'
  # The actual isolated risk reservation still exists; no automatic state
  # update occurs from the read-only evidence chain.
  assert r.hlen(m.keys[1])==1
  assert json.loads(r.hget(m.keys[4],'test-i'))['stage']=='PREPARED'
  print(json.dumps(dict(result='PASS',atomic_reserve=v['state'],
    stored_stage='PREPARED',unknown_broker_filled='RECOVERED_FILL_PENDING_JOURNAL',
    protected_evidence='PROTECTED_EVIDENCE_ONLY',
    missing_stop='FREEZE_UNPROTECTED',
    flat_without_exit_history='HOLD_RECONCILIATION',
    reservation_still_held=True,automatic_commits=0,automatic_releases=0,
    real_binance_calls=0,production_redis_calls=0),indent=2))
 finally:stop(proc,r)
