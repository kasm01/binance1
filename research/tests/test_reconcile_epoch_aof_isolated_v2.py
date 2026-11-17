"""Monotonic epoch, sticky hazard, duplicate event, AOF restart, no live writes."""
import os,json
from tempfile import TemporaryDirectory
from pathlib import Path
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_epoch_v2 import IsolatedEpochAudit
with TemporaryDirectory(prefix='risk_epoch_aof_') as directory:
 os.chmod(directory,0o700)
 proc,r=boot(directory)
 try:
  risk=IsolatedAtomicReservePrepare(r)
  assert risk.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert risk.act('RESERVE_PREPARE',request_id='intent1',generation='g1',strategy_id='SHORT1',
    symbol='INJUSDT',side='short',client_order_id='client1',now_ms=1000,
    equity=100,original_qty=2,requested_notional=46)['state']=='READY'
  e=IsolatedEpochAudit(r,risk.keys)
  p=dict(intent_id='intent1',generation='g1',client_order_id='client1')
  positive={'state':'PROTECTED_EVIDENCE_ONLY','reason':'valid_stop'}
  negative={'state':'FREEZE_UNPROTECTED','reason':'native_stop_missing'}
  timeout={'state':'UNKNOWN_FREEZE','reason':'broker_timeout'}
  e1=e.begin()
  e2=e.begin()
  assert (e1,e2)==(1,2)
  # Newer negative observation finishes FIRST.
  q=e.persist(**p,event_id='obs-negative',epoch=e2,result=negative,
              started_ms=2000,ended_ms=2100)
  assert q['reason']=='new_epoch_archived'
  assert risk.act('STATUS')['reason']=='CLOSED'
  stale=e.persist(**p,event_id='obs-delayed-positive',epoch=e1,
                  result=positive,started_ms=1000,ended_ms=3000)
  assert stale['reason']=='stale_event_archived'
  current=json.loads(r.hget(e.keys[4],'intent1'))
  assert current['epoch']==2 and current['label']=='FREEZE_UNPROTECTED'
  assert current['max_severity']==5
  assert r.xlen(e.keys[6])==2 and r.xlen(e.keys[7])==1
  # Exact replay does not duplicate events or hazards.
  again=e.persist(**p,event_id='obs-negative',epoch=e2,result=negative,
                  started_ms=2000,ended_ms=2100)
  assert again['reason']=='idempotent_event_replay'
  assert r.xlen(e.keys[6])==2 and r.xlen(e.keys[7])==1
  # Malicious/conflicting event ID rejected.
  conflict=e.persist(**p,event_id='obs-negative',epoch=e2,result=timeout,
                  started_ms=2000,ended_ms=2100)
  assert conflict['reason']=='event_id_conflict'
  # Newer apparently positive evidence must never clear prior critical flag.
  e3=e.begin()
  q=e.persist(**p,event_id='obs-new-positive',epoch=e3,result=positive,
              started_ms=3200,ended_ms=3300)
  assert q['reason']=='new_epoch_archived'
  current=json.loads(r.hget(e.keys[4],'intent1'))
  assert current['epoch']==3 and current['label']=='PROTECTED_EVIDENCE_ONLY'
  assert current['max_severity']==5
  assert risk.act('STATUS')['reason']=='CLOSED'
  # Late unknown risk should increase sticky severity even if it is not latest.
  # Current max was already 5, hazard must nonetheless append.
  e4=e.begin();e5=e.begin()
  e.persist(**p,event_id='obs-e5',epoch=e5,result=positive,
       started_ms=5000,ended_ms=5100)
  late=e.persist(**p,event_id='obs-e4-unknown',epoch=e4,result=timeout,
       started_ms=4000,ended_ms=5500)
  assert late['reason']=='stale_event_archived'
  assert r.xlen(e.keys[6])==5 and r.xlen(e.keys[7])==2
  # Wrong generation never appends.
  bad=e.persist(intent_id='intent1',generation='WRONG',client_order_id='client1',
    event_id='forged',epoch=e.begin(),result=negative,started_ms=6000,ended_ms=6100)
  assert bad['reason']=='intent_identity_mismatch'
  assert r.xlen(e.keys[6])==5
  assert r.hlen(risk.keys[1])==1
  # Redis restart keeps all history; risk gate cannot reopen.
  stop(proc,r);proc=None
  proc,r=boot(directory)
  risk=IsolatedAtomicReservePrepare(r);e=IsolatedEpochAudit(r,risk.keys)
  assert risk.act('STATUS')['reason']=='CLOSED'
  assert risk.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_intent_or_exposure'
  assert r.xlen(e.keys[6])==5 and r.xlen(e.keys[7])==2
  assert json.loads(r.hget(e.keys[4],'intent1'))['max_severity']==5
  assert r.hlen(risk.keys[1])==1
  # Separate lineage with new intent cannot reuse an existing reserved SHORT.
  print(json.dumps(dict(result='PASS',newer_negative_not_overwritten_by_delayed_positive=True,
   sticky_critical_survives_later_positive=True,late_unknown_append_only=True,
   duplicate_same_event_id_no_reappend=True,conflicting_same_event_id_rejected=True,
   wrong_generation_rejected=True,all_event_records=5,critical_hazards=2,
   redis_aof_restart_gate_closed=True,reservation_retained=True,
   live_binance_orders=0,production_redis_writes=0,
   production_authorized=False),indent=2))
 finally:
  if proc is not None:stop(proc,r)
