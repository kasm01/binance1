"""Real Redis 7 Lua atomicity, AOF restart fence and fault rejection tests.

Test server has no TCP listener (port 0), uses isolated Unix socket, AOF
fsync=always, and never connects to production Redis or Binance.
"""
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from pathlib import Path
import json,os,subprocess,time
import redis
from orchestration.risk.atomic_reserve_prepare_v1_research import IsolatedAtomicReservePrepare

def boot(folder):
    sock=str(Path(folder)/'atomic.sock')
    cmd=['redis-server','--port','0','--unixsocket',sock,
         '--unixsocketperm','600','--appendonly','yes','--appendfsync','always',
         '--save','','--daemonize','no','--dir',str(folder),
         '--maxmemory','32mb','--maxmemory-policy','noeviction','--loglevel','warning']
    proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    r=redis.Redis(unix_socket_path=sock,decode_responses=True,socket_timeout=3,socket_connect_timeout=.3)
    for _ in range(60):
        try:
            if r.ping():return proc,r
        except (OSError,redis.exceptions.RedisError):time.sleep(.05)
    proc.terminate();proc.wait(timeout=3)
    raise RuntimeError('isolated_redis_failed')
def stop(proc,r):
    try:r.shutdown(nosave=False)
    except (OSError,redis.exceptions.RedisError):pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate();proc.wait(timeout=3)
def request(l,tag,strategy='LONG1',equity=100,asked=46,client=None):
    return l.act('RESERVE_PREPARE',request_id=tag,generation='gen-'+tag,
       strategy_id=strategy,symbol='NEARUSDT' if strategy=='LONG1' else 'INJUSDT',
       side='long' if strategy=='LONG1' else 'short',
       client_order_id=client or 'cid-'+tag,now_ms=1000,equity=equity,
       original_qty=2,requested_notional=asked)
def assert_account_consistency(l,r):
    pending=r.hgetall(l.keys[1])
    intents=r.hgetall(l.keys[4])
    journal=r.hgetall(l.keys[3])
    cid=r.hgetall(l.keys[5])
    assert len(pending)==len(intents)==len(journal)==len(cid)
    for strat,raw in pending.items():
        item=json.loads(raw)
        req=item['request_id']
        assert req in intents and req in journal
        sub=json.loads(intents[req])
        assert sub['reservation_id']==item['reservation_id']
        assert sub['owner']==strat
        assert cid[sub['client_order_id']]==req
    return pending
def test():
 with TemporaryDirectory(prefix='atomic_risk_prep_aof_') as folder:
    os.chmod(folder,0o700)
    proc,r=boot(folder)
    try:
        l=IsolatedAtomicReservePrepare(r)
        assert l.act('STATUS')['reason']=='CLOSED'
        assert request(l,'closed')['reason']=='gate_closed_or_restart'
        assert r.hlen(l.keys[1])==0 and r.hlen(l.keys[4])==0
        assert l.act('OPEN_TEST',proof='fake')['reason']=='reconciliation_required'
        assert l.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
        a=request(l,'A')
        assert a['state']=='READY' and a['allowed_notional']==46
        assert_account_consistency(l,r)
        assert request(l,'A')==a
        assert l.act('RESERVE_PREPARE',request_id='A',generation='gen-A',
          strategy_id='LONG1',symbol='NEARUSDT',side='long',client_order_id='cid-A',
          original_qty=2,requested_notional=45)['reason']=='request_id_payload_conflict'
        assert request(l,'colliding',strategy='SHORT1',client='cid-A')['reason']=='client_order_id_occupied'
        assert_account_consistency(l,r)
        b=request(l,'B',strategy='SHORT1')
        assert b['state']=='READY' and b['allowed_notional']==46
        pending=assert_account_consistency(l,r)
        assert len(pending)==2
        assert sum(json.loads(row)['notional'] for row in pending.values())==92
        assert request(l,'C')['reason']=='strategy_slot_occupied'
        assert request(l,'D',strategy='SHORT1')['reason']=='strategy_slot_occupied'
        assert_account_consistency(l,r)
        # AOF restart while persisted OPEN. New Redis run ID must deny risk.
        assert l.act('STATUS')['reason']=='OPEN'
        stop(proc,r)
        proc=None
        proc,r=boot(folder)
        l=IsolatedAtomicReservePrepare(r)
        assert l.act('STATUS')['reason']=='CLOSED'
        assert r.hlen(l.keys[1])==2 and r.hlen(l.keys[4])==2
        assert request(l,'A')['reason']=='gate_closed_or_restart'
        assert request(l,'E')['reason']=='gate_closed_or_restart'
        assert l.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_intent_or_exposure'
        assert_account_consistency(l,r)
        # Separate fresh isolated instance for concurrency/budget/faults.
        stop(proc,r);proc=None
        fresh=Path(folder)/'fresh';fresh.mkdir(mode=0o700)
        proc,r=boot(fresh);l=IsolatedAtomicReservePrepare(r)
        assert l.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
        with ThreadPoolExecutor(max_workers=20) as pool:
            rows=list(pool.map(lambda i:request(l,'rush-'+str(i),'SHORT1'),range(60)))
        assert sum(row['state']=='READY' for row in rows)==1
        pending=assert_account_consistency(l,r)
        assert len(pending)==1
        # Use separate identity but same SHORT1 slot, must not create dangling intents.
        assert request(l,'late','SHORT1')['reason']=='strategy_slot_occupied'
        assert len(assert_account_consistency(l,r))==1
        c=request(l,'long1','LONG1',equity=100)
        assert c['state']=='READY' and c['allowed_notional']==46
        assert len(assert_account_consistency(l,r))==2
        # Two strategies share 92 notional, no budget overshoot.
        assert sum(json.loads(x)['notional'] for x in r.hvals(l.keys[1]))==92
        # Third side/invalid values fail without creating an intent.
        bad=l.act('RESERVE_PREPARE',request_id='bad',generation='g',
          strategy_id='LONG1',symbol='INJUSDT',side='long',
          client_order_id='cbad',original_qty=3,requested_notional=46)
        assert bad['reason']=='strategy_symbol_side_invalid'
        assert len(assert_account_consistency(l,r))==2
        # Missing equity is rejected, while no pending state is inserted.
        failed=l.act('RESERVE_PREPARE',request_id='zero',generation='gz',
          strategy_id='LONG1',symbol='NEARUSDT',side='long',
          client_order_id='czero',equity=0,original_qty=2)
        assert failed['reason']=='invalid_number'
        assert len(assert_account_consistency(l,r))==2
        # Lua type validation must deny a malformed audit stream BEFORE
        # writing pending/intent/journal on a new isolated Redis instance.
        stop(proc,r);proc=None
        damaged=Path(folder)/'damaged';damaged.mkdir(mode=0o700)
        proc,r=boot(damaged);l=IsolatedAtomicReservePrepare(r)
        assert l.act('OPEN_TEST',proof='ISOLATED_TEST_RECONCILED')['state']=='READY'
        r.set(l.keys[6],'corrupted-not-stream')
        bad=l.act('RESERVE_PREPARE',request_id='malformed',
          generation='bad',strategy_id='LONG1',symbol='NEARUSDT',
          side='long',client_order_id='cid-malformed',now_ms=1000,
          equity=100,original_qty=2,requested_notional=46)
        assert bad['reason']=='audit_event_stream_wrong_type'
        assert r.hlen(l.keys[1])==r.hlen(l.keys[3])==r.hlen(l.keys[4])==0
        # Simulate network loss: no falling through into accepting an order.
        stop(proc,r);proc=None
        outage=False
        try:request(l,'during-outage','SHORT1')
        except (OSError,redis.exceptions.RedisError):outage=True
        assert outage
        print(json.dumps(dict(
          result='PASS',isolated_redis_socket_only=True,aof_always=True,
          concurrent_requests=60,concurrent_accepts=1,
          two_bot_reservations=2,combined_notional=92,
          reserve_and_intent_same_lua_transaction=True,
          no_dangling_intents=True,prewrite_wrongtype_rejected_without_partial_state=True,
          idempotent_same_payload=True,
          conflicting_payload_rejected=True,
          duplicate_clientid_rejected=True,
          restart_with_persisted_open_fenced=True,
          restart_with_unresolved_reservations_stays_closed=True,
          network_failure_does_not_create_order_permission=True,
          real_binance_orders=0,production_redis_writes=0,
          production_service_changes=0),indent=2))
    finally:
        if proc is not None:stop(proc,r)
if __name__=='__main__':test()
