"""Actual isolated Redis AOF mode lifecycle and epoch consistency."""
from tempfile import TemporaryDirectory
import os,json
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.isolated_risk_mode_journal_v1 import IsolatedRiskModeJournal
with TemporaryDirectory(prefix='mode_journal_aof_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  m=IsolatedRiskModeJournal(r)
  assert m.act('GET')['state']=='UNKNOWN'
  # Cannot initialize with historical peak lower than current.
  assert m.act('INIT_TEST',source_time_ms=1000,equity=62.668,pnl=60)['state']=='REJECT'
  assert m.act('INIT_TEST',source_time_ms=1000,equity=62.668,pnl=68.08915253)['state']=='READY'
  assert m.state()['mode']=='REDUCED'
  assert round(m.state()['drawdown_ratio'],3)==-.08
  assert m.act('INIT_TEST',source_time_ms=1001,equity=62.668,pnl=68.09)['reason']=='already_initialized'
  def tick(tag,ts,eq,close=False,pnl=0,opened='',stop_flag=False,expected=None,gen='test-generation'):
   n=m.state()['epoch'] if expected is None else expected
   return m.act('CLOSE_EVENT' if close else 'MTM',event_id=tag,expected_epoch=n,
       source_time_ms=ts,equity=eq,pnl=pnl,opened_risk_mode=opened,
       is_risk_stop=stop_flag,generation=gen)
  # One stale or conflicting source cannot change past DD regime.
  assert tick('source-bad',1100,67,gen='forged')['reason']=='source_generation_mismatch'
  assert tick('epoch-stale',1100,67,expected=0)['reason']=='invalid_input'
  assert tick('unchanged',1100,63)['mode']=='REDUCED'
  assert tick('stale',1099,67)['reason']=='stale_source_time'
  assert tick('same-id',1200,63.5)['mode']=='REDUCED'
  assert tick('same-id',1300,66)['reason']=='event_id_already_seen'
  # Exit REDUCED only after >=94% of historical peak, then RECOVERY.
  assert tick('recover-enter',1400,63.99)['mode']=='REDUCED'
  assert tick('recover-good',1500,64.2)['mode']=='RECOVERY'
  assert tick('mtm-not-a-close',1600,64.4,pnl=10)['mode']=='RECOVERY'
  assert m.state()['recovery_successes']==0
  # A profitable close from NORMAL-opened position does not count.
  assert tick('wrong-opened-mode',1700,65,close=True,pnl=1,opened='NORMAL')['mode']=='RECOVERY'
  assert m.state()['recovery_successes']==0
  assert tick('win1',1800,66,close=True,pnl=1,opened='RECOVERY')['mode']=='RECOVERY'
  assert m.state()['recovery_successes']==1
  assert tick('loss',1900,65.5,close=True,pnl=-1,opened='RECOVERY')['mode']=='RECOVERY'
  assert m.state()['recovery_successes']==1
  assert tick('win2',2000,66.7,close=True,pnl=1,opened='RECOVERY')['mode']=='NORMAL'
  assert m.state()['recovery_successes']==0
  # Drawdown >12% from historical peak should force REDUCED again.
  assert tick('dd-drop',2100,58)['mode']=='REDUCED'
  assert tick('recover-again',2200,65)['mode']=='RECOVERY'
  assert tick('stop',2300,65,close=True,pnl=-.5,opened='RECOVERY',stop_flag=True)['mode']=='REDUCED'
  before=m.state()
  assert tick('old-generation',2400,65,gen='wrong')['state']=='REJECT'
  assert m.state()==before
  # Restart persistence and CLOSED risk gate.
  event_count=r.xlen(m.keys[2]);assert event_count>8
  stop(proc,r);proc=None
  proc,r=boot(folder)
  m=IsolatedRiskModeJournal(r)
  assert m.state()==before
  assert r.get(m.keys[3])=='CLOSED'
  assert r.xlen(m.keys[2])==event_count
  assert tick('old-time-after-restart',2250,65)['reason']=='stale_source_time'
  assert tick('valid-next',2400,64)['mode']=='REDUCED'
  print(json.dumps(dict(result='PASS',mode='REDUCED',
   epochs=m.state()['epoch'],events=r.xlen(m.keys[2]),
   stale_epoch_rejected=True,stale_timestamp_rejected=True,
   mismatched_generation_rejected=True,
   duplicate_event_rejected=True,one_tick_cannot_escalate_mode=True,
   recovery_needs_two_recovery_close_wins=True,
   stop_relapse_to_reduced=True,
   AOF_restart_preserves_mode_peak_and_wins=True,
   gate_closed_after_restart=True,live_bot_changes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
