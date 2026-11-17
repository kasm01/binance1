"""Isolated AOF Redis; consumer crash-before-ack, restart, idempotent receipts."""
from tempfile import TemporaryDirectory
import json,os
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_epoch_v2 import IsolatedEpochAudit
from orchestration.risk.isolated_hazard_outbox_consumer_v1 import IsolatedHazardConsumer
with TemporaryDirectory(prefix='isolated_hazard_consumption_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  m=IsolatedAtomicReservePrepare(r)
  assert m.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert m.act('RESERVE_PREPARE',request_id='i1',generation='g1',
    strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='cid1',now_ms=1000,equity=100,
    original_qty=2,requested_notional=46)['state']=='READY'
  audit=IsolatedEpochAudit(r,m.keys)
  identity=dict(intent_id='i1',generation='g1',client_order_id='cid1')
  for eid,label in [('h1','FREEZE_UNPROTECTED'),('h2','UNKNOWN_FREEZE')]:
   epoch=audit.begin()
   assert audit.persist(**identity,event_id=eid,epoch=epoch,
     result={'state':label},started_ms=1000+epoch*100,
     ended_ms=1100+epoch*100)['state']=='READY'
  assert r.xlen(audit.keys[7])==2
  consumer=IsolatedHazardConsumer(r,audit.keys[7]);consumer.setup()
  items=consumer.poll()
  assert len(items)==2
  sid1,f1=items[0];sid2,f2=items[1]
  # Crash before ACK: both events are still in group's pending list.
  assert r.xpending(audit.keys[7],consumer.group)['pending']==2
  stop(proc,r);proc=None
  proc,r=boot(folder)
  m=IsolatedAtomicReservePrepare(r)
  audit=IsolatedEpochAudit(r,m.keys)
  consumer=IsolatedHazardConsumer(r,audit.keys[7],consumer='new_worker_after_restart')
  consumer.setup()
  reclaimed=consumer.recover(min_idle_ms=0)
  assert len(reclaimed)==2
  receipts=[consumer.acknowledge(sid,fields) for sid,fields in reclaimed]
  assert all(x['state']=='READY' for x in receipts),receipts
  assert r.xpending(audit.keys[7],consumer.group)['pending']==0
  assert r.hlen(consumer.receipts)==2
  assert r.xlen(consumer.receipt_stream)==2
  # Replayed same stream id cannot insert another delivery receipt.
  again=consumer.acknowledge(sid1,f1)
  assert again['reason']=='idempotent_receipt'
  assert r.xlen(consumer.receipt_stream)==2
  # Tampered fields cannot create receipt or ACK.
  assert consumer.acknowledge(sid1,dict(f1,event_id='forged'))['reason']=='source_event_identity_mismatch'
  # New hazard after restart is consumed and acknowledged once.
  epoch=audit.begin()
  assert audit.persist(**identity,event_id='h3',epoch=epoch,
      result={'state':'UNKNOWN_FREEZE','reason':'new_outage'},
      started_ms=4000,ended_ms=4100)['state']=='READY'
  batch=consumer.poll()
  assert len(batch)==1
  assert consumer.acknowledge(*batch[0])['state']=='READY'
  assert r.hlen(consumer.receipts)==3
  assert r.xlen(consumer.receipt_stream)==3
  assert r.xpending(audit.keys[7],consumer.group)['pending']==0
  assert m.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==1
  # Wrong-type receipt stream must fail *before* any new ack/receipt.
  epoch=audit.begin()
  assert audit.persist(**identity,event_id='h4',epoch=epoch,
      result={'state':'FREEZE_UNPROTECTED','reason':'stop_outage'},
      started_ms=5000,ended_ms=5100)['state']=='READY'
  item=consumer.poll()
  assert len(item)==1
  r.rename(consumer.receipt_stream,consumer.receipt_stream+':backup')
  r.set(consumer.receipt_stream,'invalid_storage_type')
  wrong=consumer.acknowledge(*item[0])
  assert wrong['reason']=='wrong_storage_type'
  assert r.xpending(audit.keys[7],consumer.group)['pending']==1
  assert r.hlen(consumer.receipts)==3
  r.delete(consumer.receipt_stream)
  r.rename(consumer.receipt_stream+':backup',consumer.receipt_stream)
  assert consumer.acknowledge(*item[0])['reason']=='local_receipt_acked'
  assert r.xpending(audit.keys[7],consumer.group)['pending']==0
  assert r.hlen(consumer.receipts)==4
  print(json.dumps(dict(result='PASS',initial_hazards=2,
    crash_before_ack_recovered=2,pending_after_recovery=0,
    new_event_after_restart=1,
    local_receipts=4,local_receipt_events=4,
    wrong_type_write_failure_retains_pending=True,
    duplicate_replay_idempotent=True,tampered_event_rejected=True,
    central_gate='CLOSED',reservation_retained=True,
    external_notification_sent=False,exactly_once_external_delivery_proven=False,
    production_redis_writes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
