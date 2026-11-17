from tempfile import TemporaryDirectory
import os,json
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_epoch_v2 import IsolatedEpochAudit
from orchestration.risk.isolated_hazard_outbox_consumer_v1 import IsolatedHazardConsumer
from orchestration.risk.isolated_central_control_hazard_gate_v1 import inspect
with TemporaryDirectory(prefix='central_control_hazard_test_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  m=IsolatedAtomicReservePrepare(r)
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert m.act('RESERVE_PREPARE',request_id='i1',generation='g1',
      strategy_id='SHORT1',symbol='INJUSDT',side='short',
      client_order_id='cid1',now_ms=1000,equity=100,
      original_qty=2,requested_notional=46)['state']=='READY'
  e=IsolatedEpochAudit(r,m.keys)
  cons=IsolatedHazardConsumer(r,e.keys[7]);cons.setup()
  assert inspect(r,m.keys,e.keys,cons,intent_id='i1')['state']=='FREEZE'
  assert e.persist(intent_id='i1',generation='g1',client_order_id='cid1',
    event_id='risk1',epoch=e.begin(),result=dict(state='FREEZE_UNPROTECTED'),
    started_ms=1000,ended_ms=1100)['state']=='READY'
  a=inspect(r,m.keys,e.keys,cons,intent_id='i1')
  assert a['state']=='CRITICAL_FREEZE' and a['pending_local_receipts']==1
  (sid,row),=cons.poll()
  assert cons.acknowledge(sid,row)['state']=='READY'
  a=inspect(r,m.keys,e.keys,cons,intent_id='i1')
  assert a['state']=='CRITICAL_FREEZE' and a['pending_local_receipts']==0
  assert not a['can_restore_normal'] and not a['can_open']
  assert e.persist(intent_id='i1',generation='g1',client_order_id='cid1',
    event_id='positive2',epoch=e.begin(),result=dict(state='PROTECTED_EVIDENCE_ONLY'),
    started_ms=1200,ended_ms=1300)['state']=='READY'
  a=inspect(r,m.keys,e.keys,cons,intent_id='i1')
  assert a['state']=='CRITICAL_FREEZE' and a['risk_severity']==5
  stop(proc,r);proc=None
  proc,r=boot(folder)
  m=IsolatedAtomicReservePrepare(r);e=IsolatedEpochAudit(r,m.keys)
  cons=IsolatedHazardConsumer(r,e.keys[7])
  a=inspect(r,m.keys,e.keys,cons,intent_id='i1')
  assert a['state']=='CRITICAL_FREEZE' and a['risk_severity']==5
  assert r.hlen(m.keys[1])==1
  print(json.dumps(dict(result='PASS',critical_freeze_persists_with_receipts=True,
      critical_freeze_persists_with_later_positive=True,
      AOF_restart_remains_critical=True,
      live_gate_change_authorized=False,
      external_alert_delivery=False,production_redis_writes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
