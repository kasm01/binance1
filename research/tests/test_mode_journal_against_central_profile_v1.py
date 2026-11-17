"""Crosscheck isolated durable journal against installed pure Central Master."""
import os,sys
from tempfile import TemporaryDirectory
from pathlib import Path
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.isolated_risk_mode_journal_v1 import IsolatedRiskModeJournal
import orchestration.risk as risk_package
risk_package.__path__.append('/home/kasm920/binance1-pro/orchestration/risk')
from orchestration.risk.central_risk_master import CentralRiskMaster,PortfolioRiskState
from orchestration.risk.central_risk_enforce_dry_profile import build_enforce_dry_config
with TemporaryDirectory(prefix='mode_profile_parity_') as directory:
 os.chmod(directory,0o700)
 p,r=boot(directory)
 try:
  j=IsolatedRiskModeJournal(r)
  m=CentralRiskMaster(build_enforce_dry_config())
  assert j.act('INIT_TEST',source_time_ms=1000,equity=62.66814695,pnl=68.08915253)['state']=='READY'
  state=PortfolioRiskState(68.08915253,'REDUCED',0)
  actions=[('MTM',63,0,'',False),('MTM',64.2,0,'',False),
           ('CLOSE_EVENT',65,1,'NORMAL',False),
           ('CLOSE_EVENT',65.4,1,'RECOVERY',False),
           ('CLOSE_EVENT',66,1,'RECOVERY',False),
           ('MTM',57.5,0,'',False),
           ('MTM',65,0,'',False),
           ('CLOSE_EVENT',65,-1,'RECOVERY',True)]
  for i,(typ,equity,pnl,opened,stop_flag) in enumerate(actions,1):
   if typ=='MTM':
    state,_=m.update_state_on_mtm(state=state,mtm_equity=equity)
   else:
    state,_=m.update_state_on_close(state=state,equity_after=equity,
             pnl_net=pnl,opened_risk_mode=opened,is_risk_stop=stop_flag)
   out=j.act(typ,event_id='evt-'+str(i),expected_epoch=j.state()['epoch'],
         source_time_ms=1000+i*100,equity=equity,pnl=pnl,
         opened_risk_mode=opened,is_risk_stop=stop_flag)
   assert out['state']=='READY',out
   assert j.state()['mode']==state.mode,(i,state.mode,j.state())
   assert j.state()['recovery_successes']==state.recovery_successes
  print('PASS 8 sequential risk-mode transitions match actual enforce-dry CentralRiskMaster profile')
 finally:stop(p,r)
