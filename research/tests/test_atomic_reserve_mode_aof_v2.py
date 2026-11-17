"""V2 risk mode snapshot and atomic budget AOF isolated Redis test."""
from tempfile import TemporaryDirectory
from decimal import Decimal
import os,json
from concurrent.futures import ThreadPoolExecutor
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_mode_v2_research import IsolatedModeRiskReserve
from orchestration.risk.atomic_reserve_prepare_mode_v2_research import LUA
def call(m,tag,strategy='LONG1',eq=100,now=1000000,asked=None):
 return m.op('RESERVE_PREPARE',request_id=tag,generation='gen-'+tag,
 strategy_id=strategy,symbol='NEARUSDT' if strategy=='LONG1' else 'INJUSDT',
 side='long' if strategy=='LONG1' else 'short',
 client_order_id='cid-'+tag,now_ms=now,equity=eq,
 original_qty=2,requested_notional=eq*.5 if asked is None else asked)
def start(m):
 assert m.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
with TemporaryDirectory(prefix='atomic_mode_reserve_aof_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  m=IsolatedModeRiskReserve(r)
  assert m.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  assert call(m,'missing')['reason']=='risk_snapshot_missing'
  m.test_snapshot(mode='REDUCED',equity=100,peak=110)
  one=call(m,'one')
  assert one['state']=='READY' and abs(one['allowed_notional']-23)<1e-7,one
  two=call(m,'two',strategy='SHORT1')
  assert two['state']=='READY' and abs(two['allowed_notional']-23)<1e-7,two
  assert r.hlen(m.keys[1])==r.hlen(m.keys[4])==r.hlen(m.keys[3])==2
  assert sum(json.loads(x)['notional'] for x in r.hvals(m.keys[1]))==46
  assert call(m,'one')==one
  assert call(m,'another')['reason']=='strategy_slot_occupied'
  assert m.op('STATUS')['reason']=='OPEN'
  stop(proc,r);proc=None
  proc,r=boot(folder)
  m=IsolatedModeRiskReserve(r)
  assert m.op('STATUS')['reason']=='CLOSED'
  assert r.hlen(m.keys[1])==2
  assert m.op('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_intent_or_exposure'
  stop(proc,r);proc=None
  new=__import__('pathlib').Path(folder)/'fresh';new.mkdir(mode=0o700)
  proc,r=boot(new);m=IsolatedModeRiskReserve(r);start(m)
  for i,(mode,expected) in enumerate([('NORMAL',46),('RECOVERY',32.2),('REDUCED',23)]):
   m.test_snapshot(mode=mode,equity=100,peak=100)
   result=call(m,'case'+str(i))
   assert result['state']=='READY' and abs(result['allowed_notional']-expected)<1e-7,result
   # reset research keys only, never on production Redis.
   r.delete(*m.keys[:7]);start(m)
  # 62.66814695 USDT REDUCED => 14.4136737985, exactly policy profile.
  eq=62.66814695
  m.test_snapshot(mode='REDUCED',equity=eq,peak=68.08915253)
  v=call(m,'live_equity_example',eq=eq)
  assert v['state']=='READY' and abs(v['allowed_notional']-14.4136737985)<1e-6,v
  assert not v['order_submission_authorized']
  r.delete(*m.keys[:7]);start(m)
  m.test_snapshot(mode='REDUCED',equity=100,peak=110,ts_ms=100000)
  assert call(m,'stale')['reason']=='risk_snapshot_invalid_or_stale'
  assert r.hlen(m.keys[1])==r.hlen(m.keys[4])==0
  m.test_snapshot(mode='NORMAL',equity=100,peak=115)
  assert call(m,'ddbad')['reason']=='normal_mode_inconsistent_with_drawdown'
  m.test_snapshot(mode='REDUCED',equity=100,peak=110,inflight_unknown=True)
  assert call(m,'unknown')['reason']=='risk_snapshot_untrusted_or_unhealthy'
  m.test_snapshot(mode='REDUCED',equity=100,peak=110)
  assert call(m,'equity-conflict',eq=101)['reason']=='caller_equity_disagrees_with_snapshot'
  r.set(m.keys[7],'{bad')
  assert call(m,'invalid-json')['reason']=='risk_snapshot_invalid_json'
  r.set(m.keys[7],json.dumps(dict(source='OTHER',mode='NORMAL')))
  assert call(m,'untrusted')['reason']=='risk_snapshot_untrusted_or_unhealthy'
  m.test_snapshot(mode='BAD',equity=100,peak=110)
  assert call(m,'invalid-mode')['reason']=='risk_mode_invalid'
  m.test_snapshot(mode='REDUCED',equity=100,peak=110)
  with ThreadPoolExecutor(max_workers=20) as pool:
   outcomes=list(pool.map(lambda i:call(m,'rush'+str(i),strategy='SHORT1'),range(60)))
  assert sum(x['state']=='READY' for x in outcomes)==1
  assert r.hlen(m.keys[1])==r.hlen(m.keys[4])==1
  print(json.dumps(dict(result='PASS',red_budget_ratio='.23',
      normal_budget_ratio='.46',recovery_budget_ratio='.322',
      sample_reduced_notional=14.4136737985,
      risk_snapshot_required=True,stale_snapshot_rejected=True,
      dd_conflict_rejected=True,untrusted_input_rejected=True,
      red_unknown_frozen=True,concurrent_requests=60,concurrent_admitted=1,
      restart_gate_closed_with_persistent_reservations=True,
      actual_broker_authorization=False,production_redis_writes=0),indent=2))
 finally:
  if proc is not None:stop(proc,r)
