"""AOF isolated risk->intent->mock broker->audit freeze worker tests."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json,os
from concurrent.futures import ThreadPoolExecutor
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_worker_v1 import IsolatedReadOnlyReconcileWorker

class Broker:
 def __init__(self,*,network_fail=False,missing_stop=False):
  self.network_fail=network_fail;self.missing_stop=missing_stop
 def get_order(self,s,c):
  if self.network_fail:raise TimeoutError('simulated')
  return dict(symbol=s,clientOrderId=c,orderId=987,side='SELL',
              status='FILLED',origQty='2',executedQty='2')
 def get_fills(self,s,oid):
  return [dict(symbol=s,orderId=987,id=1,qty='1'),dict(symbol=s,orderId=987,id=2,qty='1')]
 def get_positions(self,s):
  return [dict(symbol=s,positionSide='SHORT',positionAmt='-2')]
 def get_open_orders(self,s):return []
 def get_open_algos(self,s):
  return [] if self.missing_stop else [dict(symbol=s,clientAlgoId='b1s1-native1',
    algoId=123,side='BUY',positionSide='SHORT',orderType='STOP_MARKET',
    closePosition=True,workingType='MARK_PRICE',algoStatus='NEW')]
with TemporaryDirectory(prefix='isolated_reconcile_worker_') as d:
 os.chmod(d,0o700)
 proc,r=boot(d)
 try:
  m=IsolatedAtomicReservePrepare(r)
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  a=m.act('RESERVE_PREPARE',request_id='i1',generation='g1',
    strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='b1s1_i1',now_ms=1000,equity=100,
    original_qty=2,requested_notional=46)
  assert a['state']=='READY'
  # Research-only stage update: authentic fill journal writer does not exist.
  rec=json.loads(r.hget(m.keys[4],'i1'))
  rec.update(stage='FILLED',filled_qty='2',client_algo_id='b1s1-native1')
  r.hset(m.keys[4],'i1',json.dumps(rec))
  w=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker())
  first=w.inspect_and_persist('i1')
  assert first['audit']['state']=='READY'
  assert first['evidence']['state']=='PROTECTED_EVIDENCE_ONLY'
  assert m.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==1
  assert r.hlen(w.audit_keys[1])==1
  second=w.inspect_and_persist('i1')
  assert second['audit']['reason']=='idempotent_audit_replay'
  assert r.xlen(w.audit_keys[0])==1
  # The same identity with now missing stop cannot overwrite audit.
  changed=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker(missing_stop=True))
  got=changed.inspect_and_persist('i1')
  assert got['evidence']['state']=='FREEZE_UNPROTECTED'
  assert got['audit']['reason']=='conflicting_evidence_needs_new_epoch'
  # 30 concurrent workers cannot append duplicate audit evidence.
  with ThreadPoolExecutor(max_workers=15) as pool:
   responses=list(pool.map(lambda _:w.inspect_and_persist('i1'),range(30)))
  assert all(z['audit']['reason']=='idempotent_audit_replay' for z in responses)
  assert r.xlen(w.audit_keys[0])==1
  # AOF restart: audit + reservation persist; gate remains CLOSED.
  stop(proc,r);proc=None
  proc,r=boot(d);m=IsolatedAtomicReservePrepare(r)
  w=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker())
  assert m.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==1
  assert r.hlen(w.audit_keys[1])==1
  assert r.xlen(w.audit_keys[0])==1
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_intent_or_exposure'
  replay=w.inspect_and_persist('i1')
  assert replay['audit']['reason']=='idempotent_audit_replay'
  assert r.xlen(w.audit_keys[0])==1
  # Separate new observation, network failed before first audit. Never opens.
  # keep same journal, do not clear old event or authorization.
  assert not replay['can_commit'] and not replay['can_release'] and not replay['can_submit']
  fail_worker=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker(network_fail=True))
  failure=fail_worker.inspect_and_persist('i1')
  assert failure['evidence']['state']=='UNKNOWN_FREEZE'
  assert failure['audit']['reason']=='conflicting_evidence_needs_new_epoch'
  assert m.act('STATUS')['reason']=='CLOSED'
  # Independent isolated ledger: FIRST ever broker read times out. Its
  # UNKNOWN must be persisted once and must close the gate immediately.
  stop(proc,r);proc=None
  second_dir=__import__('pathlib').Path(d)/'initial_outage'
  second_dir.mkdir(mode=0o700)
  proc,r=boot(second_dir)
  m=IsolatedAtomicReservePrepare(r)
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert m.act('RESERVE_PREPARE',request_id='i2',generation='g2',
     strategy_id='SHORT1',symbol='INJUSDT',side='short',
     client_order_id='b1s1_i2',now_ms=1000,equity=100,
     original_qty=2,requested_notional=46)['state']=='READY'
  rec=json.loads(r.hget(m.keys[4],'i2'))
  rec['stage']='UNKNOWN'
  r.hset(m.keys[4],'i2',json.dumps(rec))
  fw=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker(network_fail=True))
  initial_failure=fw.inspect_and_persist('i2')
  assert initial_failure['evidence']['state']=='UNKNOWN_FREEZE'
  assert initial_failure['audit']['reason']=='audit_persisted_gate_closed'
  assert m.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==1
  assert r.xlen(fw.audit_keys[0])==1
  stop(proc,r);proc=None
  proc,r=boot(second_dir)
  m=IsolatedAtomicReservePrepare(r)
  fw=IsolatedReadOnlyReconcileWorker(r,m.keys,Broker(network_fail=True))
  assert m.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==1
  assert r.xlen(fw.audit_keys[0])==1
  print(json.dumps(dict(result='PASS',
    first_broker_result=first['evidence']['state'],risk_gate_after_audit='CLOSED',
    duplicate_audit_events=0,concurrent_replays=30,
    changes_require_new_epoch=True,
    simulated_stop_failure='FREEZE_UNPROTECTED',
    simulated_network_failure='UNKNOWN_FREEZE',
    restart_aof_keeps_reservation=True,
    restart_aof_keeps_audit=True,
    first_broker_lookup_failure_persisted_freeze=True,
    actual_binance_calls=0,production_redis_writes=0,
    live_orders=0,automatic_commit=False,automatic_release=False),indent=2))
 finally:
  if proc is not None:stop(proc,r)
