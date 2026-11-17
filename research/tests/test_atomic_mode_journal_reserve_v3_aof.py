"""End-to-end isolated persistent mode journal + atomic RESERVE/PREPARE."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json,os
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.isolated_risk_mode_journal_v1 import IsolatedRiskModeJournal
from orchestration.risk.atomic_reserve_mode_journal_v3_research import IsolatedModeJournalReserveV3
with TemporaryDirectory(prefix='mode_journal_atomic_v3_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  j=IsolatedRiskModeJournal(r)
  v=IsolatedModeJournalReserveV3(r)
  assert v.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  def reserve(tag,eq=100):
   return v.op('RESERVE_PREPARE',request_id=tag,generation='gen-'+tag,
      strategy_id='LONG1',symbol='NEARUSDT',side='long',
      client_order_id='cid-'+tag,now_ms=1000000,equity=eq,
      original_qty=2,requested_notional=eq/2)
  assert reserve('missing')['reason']=='risk_snapshot_missing'
  assert j.act('INIT_TEST',source_time_ms=900000,equity=100,pnl=110)['state']=='READY'
  r.set(v.keys[8],json.dumps(j.state()))
  v.test_snapshot(mode_state=j.state(),eq=100)
  first=reserve('A')
  assert first['state']=='READY' and abs(first['allowed_notional']-23)<1e-7,first
  assert not first['order_submission_authorized']
  assert r.hlen(v.keys[1])==r.hlen(v.keys[4])==1
  assert v.op('STATUS')['reason']=='OPEN'
  stop(proc,r);proc=None
  proc,r=boot(folder)
  v=IsolatedModeJournalReserveV3(r);j=IsolatedRiskModeJournal(r)
  assert v.op('STATUS')['reason']=='CLOSED'
  assert r.hlen(v.keys[1])==1
  assert j.state()['mode']=='REDUCED'
  assert v.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_intent_or_exposure'
  stop(proc,r);proc=None
  d2=Path(folder)/'second';d2.mkdir(mode=0o700)
  proc,r=boot(d2);v=IsolatedModeJournalReserveV3(r);j=IsolatedRiskModeJournal(r)
  assert v.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert j.act('INIT_TEST',source_time_ms=900000,equity=100,pnl=110)['state']=='READY'
  r.set(v.keys[8],json.dumps(j.state()))
  v.test_snapshot(mode_state=j.state(),eq=100,mode='NORMAL')
  assert reserve('forged_normal')['reason']=='mode_snapshot_generation_or_state_mismatch'
  v.test_snapshot(mode_state=j.state(),eq=100,generation='different')
  assert reserve('forged_generation')['reason']=='mode_snapshot_generation_or_state_mismatch'
  v.test_snapshot(mode_state=j.state(),eq=100,mode_epoch=999)
  assert reserve('future_epoch')['reason']=='mode_snapshot_generation_or_state_mismatch'
  v.test_snapshot(mode_state=j.state(),eq=100)
  assert reserve('legit')['state']=='READY'
  assert r.hlen(v.keys[1])==1 and r.hlen(v.keys[4])==1
  # Even the one legitimate reserve remains not executable.
  assert r.get(v.keys[0]).startswith('OPEN:')
  print(json.dumps(dict(result='PASS',initial_mode='REDUCED',
    actual_budget_ratio=.23,forged_mode_normal_rejected=True,
    forged_source_generation_rejected=True,
    future_mode_epoch_rejected=True,
    reserve_and_intent_atomic=True,AOF_restart_preserves_mode_and_reservation=True,
    restart_gate_closed=True,
    live_order_authority=False,
    authenticated_broker_proof=False,
    production_redis_changes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
