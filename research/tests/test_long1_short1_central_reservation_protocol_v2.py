"""Isolated future central Risk Master protocol acceptance tests, no Redis writes."""
from concurrent.futures import ThreadPoolExecutor
from orchestration.risk.central_reservation_protocol_v2_dry_reference import DryReservationReference
r=DryReservationReference(equity=100)
a=r.reserve(request_id='A',strategy_id='LONG1',symbol='NEARUSDT',side='long',now_ms=1000)
assert a['state']=='READY' and a['allowed_notional']==46.0
assert r.reserve(request_id='A',strategy_id='LONG1',symbol='NEARUSDT',side='long',now_ms=1100).get('replayed')
assert r.reserve(request_id='A',strategy_id='SHORT1',symbol='INJUSDT',side='short',now_ms=1100)['reason']=='request_id_payload_conflict'
assert r.reserve(request_id='B',strategy_id='LONG1',symbol='NEARUSDT',side='long',now_ms=1100)['reason']=='strategy_slot_busy'
b=r.reserve(request_id='C',strategy_id='SHORT1',symbol='INJUSDT',side='short',now_ms=1200)
assert b['state']=='READY' and b['allowed_notional']==46.0
assert sum(s.notional for s in r.active.values())==92.0<=r.cap
assert r.release(strategy_id='LONG1',request_id='bad')['state']=='REJECT'
assert r.release(strategy_id='LONG1',request_id='A')['state']=='READY'
a2=r.reserve(request_id='D',strategy_id='LONG1',symbol='NEARUSDT',side='long',now_ms=1300)
assert a2['state']=='READY'
assert r.release(strategy_id='LONG1',request_id='A')['state']=='REJECT'
assert 'LONG1' in r.active
assert r.commit(strategy_id='LONG1',request_id='D',position_id='pos-1',now_ms=1400)['state']=='READY'
assert r.commit(strategy_id='LONG1',request_id='D',position_id='pos-1',now_ms=1400)['reason']=='already_committed'
assert r.release(strategy_id='LONG1',request_id='D',position_id='pos-1',flat_verified=False)['reason']=='exchange_flat_not_verified'
assert r.release(strategy_id='LONG1',request_id='D',position_id='wrong',flat_verified=True)['reason']=='position_owner_mismatch'
assert r.release(strategy_id='LONG1',request_id='D',position_id='pos-1',flat_verified=True)['reason']=='released_confirmed_flat'
assert r.release(strategy_id='LONG1',request_id='D',position_id='pos-1',flat_verified=True)['reason']=='already_released'
assert r.expire(1_000_000_000)==['C']
assert 'SHORT1' in r.active
assert r.release(strategy_id='SHORT1',request_id='C')['state']=='READY'
# simultaneous contender rush: one atomic owner, same strategy slot.
r2=DryReservationReference(equity=100)
with ThreadPoolExecutor(max_workers=20) as pool:
    outcomes=list(pool.map(lambda i:r2.reserve(request_id='rush-'+str(i),strategy_id='SHORT1',symbol='INJUSDT',side='short',now_ms=100),range(50)))
assert sum(z['state']=='READY' for z in outcomes)==1
assert len(r2.active)==1
print('PASS 17 ownership/idempotency/atomic-in-process/dual-budget/commit/flat/fencing tests')
print('NOTE: reference only; real Redis multi-process atomicity not proven.')
