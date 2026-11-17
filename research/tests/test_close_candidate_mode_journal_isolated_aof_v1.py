"""No close candidate can credit Recovery state in isolated Redis AOF."""
from tempfile import TemporaryDirectory
import os,json
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.isolated_risk_mode_journal_v1 import IsolatedRiskModeJournal
from orchestration.risk.isolated_close_candidate_queue_v1 import IsolatedCloseCandidateQueue
with TemporaryDirectory(prefix='recovery_close_candidate_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  mode=IsolatedRiskModeJournal(r)
  assert mode.act('INIT_TEST',source_time_ms=1000,equity=100,pnl=110)['state']=='READY'
  assert mode.act('MTM',event_id='recover1',expected_epoch=mode.state()['epoch'],
       source_time_ms=1100,equity=103,pnl=0)['mode']=='REDUCED'
  assert mode.act('MTM',event_id='recover2',expected_epoch=mode.state()['epoch'],
       source_time_ms=1200,equity=104.5,pnl=0)['mode']=='RECOVERY'
  before=mode.state().copy()
  queue=IsolatedCloseCandidateQueue(r,mode.keys[0])
  good=dict(state='NET_PNL_CANDIDATE_ONLY',net_pnl_usdt='0.03',
     candidate_recovery_win=True,eligible_recovery_win=False,
     can_update_risk_mode=False)
  first=queue.enqueue('close-one','intent1',before['source_gen'],before['epoch'],good)
  assert first['reason']=='candidate_queued_no_risk_promotion'
  assert queue.enqueue('close-one','intent1',before['source_gen'],before['epoch'],good)['reason']=='idempotent_candidate'
  assert queue.enqueue('close-one','intent1',before['source_gen'],before['epoch'],
      dict(good,net_pnl_usdt='1'))['reason']=='candidate_event_conflict'
  bnb=dict(state='UNPRICED_FEES_HOLD',candidate_recovery_win=False,
           eligible_recovery_win=False,can_update_risk_mode=False)
  assert queue.enqueue('close-two','intent1',before['source_gen'],before['epoch'],bnb)['state']=='READY'
  assert r.hlen(queue.keys[1])==2 and r.xlen(queue.keys[2])==2
  assert mode.state()==before
  assert mode.state()['recovery_successes']==0
  assert r.get(queue.keys[4])=='CLOSED'
  stop(proc,r);proc=None
  proc,r=boot(folder)
  mode=IsolatedRiskModeJournal(r);queue=IsolatedCloseCandidateQueue(r,mode.keys[0])
  assert mode.state()==before
  assert r.hlen(queue.keys[1])==2 and r.xlen(queue.keys[2])==2
  assert queue.enqueue('close-three','intent1','wrong-generation',
    before['epoch'],good)['reason']=='source_generation_mismatch'
  assert queue.enqueue('close-four','intent1',before['source_gen'],
    before['epoch']+100,good)['reason']=='mode_epoch_changed'
  print(json.dumps(dict(result='PASS',
    candidates_persisted=2,idempotent_duplicate=True,
    conflicting_duplicate_rejected=True,
    recovery_successes_unchanged=0,
    recovery_mode_kept='RECOVERY',risk_gate='CLOSED',
    redis_aof_restart_preserves_pending=True,
    wrong_generation_rejected=True,wrong_epoch_rejected=True,
    auto_mode_upgrade=False,real_orders=0,production_redis_writes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
