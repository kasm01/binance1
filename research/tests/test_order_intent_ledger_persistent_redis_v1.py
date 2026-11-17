"""Test durable journaling on private Redis 7 unix socket, AOF fsync always.
Production Redis never used; no Binance orders, no live service restarts.
"""
from tempfile import TemporaryDirectory
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,subprocess,time,json
import redis
from orchestration.risk.order_intent_ledger_v1_research import IsolatedOrderLedger

def start(folder):
    sock=str(Path(folder)/'ledger.sock')
    cmd=['redis-server','--port','0','--unixsocket',sock,
       '--unixsocketperm','600','--appendonly','yes','--appendfsync','always',
       '--save','','--maxmemory','40mb','--maxmemory-policy','noeviction',
       '--daemonize','no','--dir',folder,'--loglevel','warning']
    proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    cli=redis.Redis(unix_socket_path=sock,decode_responses=True,socket_connect_timeout=.3,socket_timeout=2)
    for _ in range(80):
        try:
            if cli.ping():return proc,cli
        except (redis.exceptions.RedisError,OSError):
            time.sleep(.05)
    proc.terminate();proc.wait(timeout=5)
    raise RuntimeError('isolated_persistent_redis_failed')
def stop(proc,cli):
    try:cli.shutdown(nosave=False)
    except (redis.exceptions.RedisError,OSError):pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.terminate();proc.wait(timeout=3)

def test():
 with TemporaryDirectory(prefix='ledger_AOF_isolated_') as d:
    os.chmod(d,0o700)
    proc,r=start(d)
    try:
        l=IsolatedOrderLedger(r)
        base=dict(intent='intent-1',owner='LONG1',reservation='gen-111',
                  symbol='NEARUSDT',side='BUY',client_id='cid-1',
                  time_ms=1000,orig_qty=3.0)
        assert l.op('STATUS')['reason']=='CLOSED'
        assert l.op('PREPARE',**base)['reason']=='gate_closed'
        assert l.op('ENABLE_TEST',proof='ISOLATED_OPERATOR_RECONCILED')['state']=='READY'
        assert l.op('PREPARE',**base)['reason']=='prepared'
        assert l.op('PREPARE',**base)['reason']=='already_prepared_or_progressed'
        assert l.op('PREPARE',**dict(base,client_id='cid-fake'))['reason']=='immutable_payload_conflict'
        assert l.op('PREPARE',**dict(base,intent='intent-2'))['reason']=='client_id_already_owned'
        assert l.op('MARK_SUBMITTING',**base)['stage']=='SUBMITTING'
        assert l.op('MARK_SUBMITTING',**base)['reason']=='never_resubmit_unconfirmed'
        assert l.op('SUBMIT_UNKNOWN',**base)['stage']=='UNKNOWN'
        assert l.op('MARK_SUBMITTING',**base)['reason']=='never_resubmit_unconfirmed'
        # Redis restart WHILE GATE IS STILL OPEN; persisted AOF OPEN must
        # be fenced by a new Redis server run_id automatically.
        assert l.op('STATUS')['reason']=='OPEN'
        stop(proc,r);proc=None
        proc,r=start(d);l=IsolatedOrderLedger(r)
        saved=l.op('GET',intent='intent-1')
        assert saved['stage']=='UNKNOWN'
        assert saved['client_order_id']=='cid-1'
        assert saved['reservation_id']=='gen-111'
        assert l.op('STATUS')['reason']=='CLOSED'
        assert l.op('ENABLE_TEST',proof='ISOLATED_OPERATOR_RECONCILED')['reason']=='pending_intents_need_reconciliation'
        assert l.op('PREPARE',**base)['reason']=='gate_closed'
        # No production override: during test we simulate read-only reconciliation
        # with CLOSED gate, but the current module forbids transitions when closed.
        # Hence the recovered UNKNOWN remains frozen; safety > availability.
        assert saved['filled_qty']==0
        # Separate new isolated journal to test ACK / partial fill / duplicate
        # without bypassing a recovered-unknown gate.
        stop(proc,r);proc=None
        d2=str(Path(d)/'fresh')
        Path(d2).mkdir()
        proc,r=start(d2);l=IsolatedOrderLedger(r)
        assert l.op('ENABLE_TEST',proof='ISOLATED_OPERATOR_RECONCILED')['state']=='READY'
        assert l.op('PREPARE',**base)['state']=='READY'
        assert l.op('MARK_SUBMITTING',**base)['state']=='READY'
        assert l.op('ACK',**dict(base,order_id='oid-1'))['stage']=='ACKED'
        assert l.op('ACK',**dict(base,order_id='different'))['reason']=='conflicting_exchange_order_id'
        assert l.op('FILL',**dict(base,order_id='oid-1',fill_id='f1',fill_qty=1))['stage']=='PARTIAL'
        assert l.op('FILL',**dict(base,order_id='oid-1',fill_id='f1',fill_qty=1))['reason']=='duplicate_fill_ignored'
        assert l.op('FILL',**dict(base,order_id='oid-1',fill_id='f1',fill_qty=2))['reason']=='conflicting_fill_duplicate'
        assert l.op('FILL',**dict(base,order_id='oid-1',fill_id='f2',fill_qty=3))['reason']=='overfill'
        assert l.op('FLAT_RECONCILED',**base)['reason']=='fills_or_terminal_unresolved'
        assert l.op('FILL',**dict(base,order_id='oid-1',fill_id='f2',fill_qty=2))['stage']=='FILLED'
        assert l.op('FLAT_RECONCILED',**base)['reason']=='broker_flat_proof_required'
        assert l.op('FLAT_RECONCILED',**dict(base,proof='BROKER_FLAT_TEST_ONLY'))['stage']=='FLAT_RECONCILED'
        assert l.op('MARK_SUBMITTING',**base)['reason']=='never_resubmit_unconfirmed'
        # 60 concurrent PREPARE for SAME identity -> exactly one original
        # PREPARE; all others idempotent responses.
        base2=dict(base,intent='intent-many',client_id='cid-many')
        with ThreadPoolExecutor(max_workers=20) as pool:
            replies=list(pool.map(lambda _:l.op('PREPARE',**base2),range(60)))
        assert sum(z['reason']=='prepared' for z in replies)==1
        assert sum(z['reason']=='already_prepared_or_progressed' for z in replies)==59
        assert r.hlen(l.keys[1])==2
        print(json.dumps(dict(
            result='PASS',persistent_aof=True,redis_restarts=2,
            unknown_intent_survives_restart=True,
            restart_gate_closed_even_when_AOF_old_gate_OPEN=True,
            recovered_unknown_cannot_be_resubmitted=True,
            immutable_client_identity=True,
            partial_fill_and_overfill_detected=True,
            duplicate_fills_idempotent=True,
            mismatched_exchange_order_rejected=True,
            automatic_release_authorized=False,
            external_broker_proof_authenticated=False,
            concurrent_prepares=60,one_new_intent=True,
            real_binance_orders=0,real_redis_touched=False
        ),indent=2))
    finally:
        if proc is not None:stop(proc,r)
if __name__=='__main__':test()
