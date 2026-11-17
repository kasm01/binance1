"""End-to-end research: Redis AOF RESERVE+PREPARE + broker-stop mock evidence.

Does NOT implement real fills, exchange auth, COMMIT or RELEASE.
"""
import json,os
from tempfile import TemporaryDirectory
from research.tests.test_atomic_reserve_prepare_aof_isolated_v1 import boot,stop
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare
from orchestration.risk.position_stop_lifecycle_evidence_v1 import assess
from orchestration.risk.authoritative_snapshot_evidence_v1 import verify
with TemporaryDirectory(prefix='full_risk_shadow_') as folder:
 os.chmod(folder,0o700)
 proc,r=boot(folder)
 try:
  master=IsolatedAtomicReservePrepare(r)
  assert master.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
  accepted=master.act('RESERVE_PREPARE',request_id='intent-e2e',
    generation='gen-e2e',strategy_id='SHORT1',symbol='INJUSDT',side='short',
    client_order_id='cid-e2e',now_ms=1000,equity=100,original_qty=2,
    requested_notional=46)
  assert accepted['state']=='READY'
  reservation=json.loads(r.hget(master.keys[1],'SHORT1'))
  ledger=json.loads(r.hget(master.keys[4],'intent-e2e'))
  # EXCHANGE NEVER ACTUALLY CALLED; simulate an independently owned
  # confirmed partial fill and a 3x leverage short position.
  ledger.update(stage='PARTIAL',filled_qty='1',client_algo_id='b1s1_stop_intent-e2e')
  position=[dict(symbol='INJUSDT',positionSide='SHORT',positionAmt='-1')]
  stop_orders=[dict(symbol='INJUSDT',clientAlgoId='b1s1_stop_intent-e2e',
    side='BUY',positionSide='SHORT',orderType='STOP_MARKET',
    closePosition=True,workingType='MARK_PRICE',algoStatus='NEW',algoId=12)]
  unprotected=assess(reservation=reservation,ledger=ledger,
    positions=position,orders=[],algos=[])
  protected=assess(reservation=reservation,ledger=ledger,
    positions=position,orders=[],algos=stop_orders)
  still_open=assess(reservation=reservation,ledger=ledger,
    positions=[],orders=[],algos=stop_orders)
  flat_ledger=dict(ledger,stage='FLAT_RECONCILED',verified_exit_order_id='exit2',
     verified_exit_qty='1')
  completed=assess(reservation=reservation,ledger=flat_ledger,
    positions=[],orders=[],algos=[])
  assert unprotected['state']=='FREEZE_UNPROTECTED'
  assert protected['state']=='PROTECTED_POSITION_OBSERVED'
  assert still_open['state']=='BLOCK_RELEASE'
  assert completed['state']=='FLAT_HISTORY_CONSISTENT'
  assert not any(z['can_release'] or z['can_commit'] or z['can_open_new']
    for z in (unprotected,protected,still_open,completed))
  # No test module touches Redis reservation after creation; no release
  # occurs even when simulated flat consistency is reported.
  assert r.hlen(master.keys[1])==1
  print(json.dumps(dict(result='PASS',isolated_redis=True,
      atomic_reserve_prepared=accepted['state'],
      partial_unprotected=unprotected['state'],
      partial_protected=protected['state'],
      flat_with_open_algo=still_open['state'],
      flat_with_exit_history=completed['state'],
      central_reservation_retained=bool(r.hget(master.keys[1],'SHORT1')),
      auto_release=False,production_authority=False),indent=2))
 finally:stop(proc,r)
