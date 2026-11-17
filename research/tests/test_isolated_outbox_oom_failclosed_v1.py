"""Real isolated Redis noeviction OOM fault injection for hazard receipts.

No production Redis configs, no production disk fill. Memory ceiling is set
below existing used memory in a temporary Redis process, then restored.
"""
import os,json,redis
from tempfile import TemporaryDirectory
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.isolated_reconcile_epoch_v2 import IsolatedEpochAudit
from orchestration.risk.isolated_hazard_outbox_consumer_v1 import IsolatedHazardConsumer
with TemporaryDirectory(prefix='redis_oom_hazard_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  master=IsolatedAtomicReservePrepare(r)
  assert master.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert master.act('RESERVE_PREPARE',request_id='i1',generation='g1',
    strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='cid1',now_ms=1000,equity=100,
    original_qty=2,requested_notional=46)['state']=='READY'
  audit=IsolatedEpochAudit(r,master.keys)
  assert audit.persist(intent_id='i1',generation='g1',client_order_id='cid1',
    event_id='hazard1',epoch=audit.begin(),
    result=dict(state='FREEZE_UNPROTECTED'),started_ms=1000,
    ended_ms=1100)['state']=='READY'
  c=IsolatedHazardConsumer(r,audit.keys[7]);c.setup()
  (sid,event),=c.poll()
  assert r.xpending(audit.keys[7],c.group)['pending']==1
  used=r.info('memory')['used_memory']
  r.config_set('maxmemory',max(1,used-1024))
  failed=False
  try:
   out=c.acknowledge(sid,event)
   failed=out['state']!='READY'
  except redis.exceptions.ResponseError as err:
   failed=True
  assert failed,'artificial memory pressure must reject write'
  assert r.hlen(c.receipts)==0
  assert r.xlen(c.receipt_stream)==0
  assert r.xpending(audit.keys[7],c.group)['pending']==1
  assert master.act('STATUS')['reason']=='CLOSED'
  assert r.hlen(master.keys[1])==1
  r.config_set('maxmemory',33554432)
  assert c.acknowledge(sid,event)['reason']=='local_receipt_acked'
  assert r.xpending(audit.keys[7],c.group)['pending']==0
  assert r.hlen(c.receipts)==1
  print(json.dumps(dict(result='PASS',oom_test_server_only=True,
    simulated_memory_ceiling_bytes=max(1,used-1024),
    oom_failed_before_receipt=True,pending_alert_preserved=True,
    risk_gate_remained_closed=True,reservation_held=True,
    successful_ack_after_restoring_test_limit=True,
    production_redis_config_changes=0,live_binance_orders=0),indent=2))
 finally:stop(proc,r)
