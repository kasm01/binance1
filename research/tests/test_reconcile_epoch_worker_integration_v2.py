"""End-to-end isolated epoch worker / simulated Binance events, no live access."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json,os
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_epoch_v2 import IsolatedEpochAudit
class FakeBroker:
 def __init__(self,*,stop_exists=True,fail=False):
  self.stop_exists=stop_exists;self.fail=fail
 def get_order(self,s,client):
  if self.fail:raise TimeoutError('simulated')
  return dict(symbol=s,clientOrderId=client,orderId=444,side='SELL',
     status='FILLED',origQty='2',executedQty='2')
 def get_fills(self,s,oid):
  return [dict(symbol=s,orderId=444,id=1,qty='1'),
          dict(symbol=s,orderId=444,id=2,qty='1')]
 def get_positions(self,s):
  return [dict(symbol=s,positionSide='SHORT',positionAmt='-2')]
 def get_open_orders(self,s):return []
 def get_open_algos(self,s):
  if not self.stop_exists:return []
  return [dict(symbol=s,clientAlgoId='b1s1_stop_g1',algoId=111,side='BUY',
   positionSide='SHORT',orderType='STOP_MARKET',closePosition=True,
   workingType='MARK_PRICE',algoStatus='NEW')]
with TemporaryDirectory(prefix='risk_epoch_worker_v2_') as d:
 os.chmod(d,0o700)
 proc,r=boot(d)
 try:
  master=IsolatedAtomicReservePrepare(r)
  assert master.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert master.act('RESERVE_PREPARE',request_id='i1',generation='g1',
    strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='cid1',now_ms=1000,equity=100,original_qty=2,
    requested_notional=46)['state']=='READY'
  # Simulated exchange journal state only within isolated test Redis:
  row=json.loads(r.hget(master.keys[4],'i1'))
  row.update(stage='FILLED',filled_qty='2',client_algo_id='b1s1_stop_g1')
  r.hset(master.keys[4],'i1',json.dumps(row))
  audit=IsolatedEpochAudit(r,master.keys)
  ok=audit.inspect_and_persist(intent_id='i1',event_id='broker-success-1',
     broker=FakeBroker())
  assert ok['evidence']['state']=='PROTECTED_EVIDENCE_ONLY'
  assert ok['audit']['reason']=='new_epoch_archived'
  assert master.act('STATUS')['reason']=='CLOSED'
  hazard=audit.inspect_and_persist(intent_id='i1',event_id='broker-stop-missing-2',
     broker=FakeBroker(stop_exists=False))
  assert hazard['evidence']['state']=='FREEZE_UNPROTECTED'
  assert hazard['audit']['reason']=='new_epoch_archived'
  unknown=audit.inspect_and_persist(intent_id='i1',event_id='broker-down-3',
     broker=FakeBroker(fail=True))
  assert unknown['evidence']['state']=='UNKNOWN_FREEZE'
  assert r.xlen(audit.keys[6])==3
  assert r.xlen(audit.keys[7])==2
  current=json.loads(r.hget(audit.keys[4],'i1'))
  assert current['label']=='UNKNOWN_FREEZE' and current['max_severity']==5
  later=audit.inspect_and_persist(intent_id='i1',event_id='broker-success-again-4',
     broker=FakeBroker())
  assert later['evidence']['state']=='PROTECTED_EVIDENCE_ONLY'
  assert json.loads(r.hget(audit.keys[4],'i1'))['max_severity']==5
  assert r.hlen(master.keys[1])==1
  assert not any(ok[z] for z in ('order_authorized','commit_authorized','release_authorized'))
  stop(proc,r);proc=None
  proc,r=boot(d)
  master=IsolatedAtomicReservePrepare(r);audit=IsolatedEpochAudit(r,master.keys)
  assert master.act('STATUS')['reason']=='CLOSED'
  assert r.xlen(audit.keys[6])==4 and r.xlen(audit.keys[7])==2
  assert json.loads(r.hget(audit.keys[4],'i1'))['max_severity']==5
  print(json.dumps(dict(result='PASS',stages=[
   'PROTECTED_EVIDENCE_ONLY','FREEZE_UNPROTECTED','UNKNOWN_FREEZE','PROTECTED_EVIDENCE_ONLY'],
   sticky_severity=5,persisted_audit_events=4,
   durable_hazard_events=2,gate_after_restart='CLOSED',
   pending_reservation_held=True,all_order_commit_release_authority=False,
   synthetic_binance=True,production_redis_writes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
