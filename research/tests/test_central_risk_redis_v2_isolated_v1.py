"""Ephemeral isolated Redis socket, zero network listeners, no persistent data.
Never connect to port 6379. Redis server process started only for this test.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess,time,json,os
import redis
from orchestration.risk.central_reservation_redis_v2_research import IsolatedRedisRiskV2

def boot(folder):
 socket=str(Path(folder)/'test.sock')
 cmd=['redis-server','--port','0','--unixsocket',socket,
   '--unixsocketperm','600','--save','','--appendonly','no',
   '--daemonize','no','--maxmemory','24mb','--maxmemory-policy','noeviction',
   '--loglevel','warning','--dir',folder]
 proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 client=redis.Redis(unix_socket_path=socket,decode_responses=True,
     socket_connect_timeout=.3,socket_timeout=1)
 for _ in range(60):
  try:
   if client.ping():return proc,client
  except (redis.exceptions.RedisError,OSError):
   time.sleep(.05)
 proc.terminate();proc.wait(timeout=5)
 raise RuntimeError('isolated_redis_not_ready')
def stop(proc,client):
 try:client.shutdown(nosave=True)
 except (redis.exceptions.RedisError,OSError):pass
 try:proc.wait(timeout=3)
 except subprocess.TimeoutExpired:
  proc.terminate();proc.wait(timeout=3)
def run():
 with TemporaryDirectory(prefix='short_risk_v2_only_') as folder:
  os.chmod(folder,0o700)
  proc,r=boot(folder)
  try:
   x=IsolatedRedisRiskV2(r)
   assert x.act('STATUS')['reason']=='CLOSED'
   assert x.act('RESERVE','A','LONG1','NEARUSDT','long',reservation_id='a')['reason']=='gate_closed_or_restarted'
   assert x.act('OPEN_GATE',verified='no')['state']=='REJECT'
   assert x.act('OPEN_GATE',verified='ISOLATED_TEST_RECONCILED')['state']=='READY'
   a=x.act('RESERVE','A','LONG1','NEARUSDT','long',reservation_id='a')
   assert a['state']=='READY' and a['allowed_notional']==46
   b=x.act('RESERVE','B','SHORT1','INJUSDT','short',reservation_id='b')
   assert b['state']=='READY' and b['allowed_notional']==46
   replay=x.act('RESERVE','A','LONG1','NEARUSDT','long',reservation_id='a')
   assert replay==a
   assert x.act('RESERVE','A','LONG1','NEARUSDT','long',reservation_id='z')['reason']=='request_id_conflict'
   assert x.act('RESERVE','Z','LONG1','NEARUSDT','long',reservation_id='z')['reason']=='strategy_slot_busy'
   assert x.act('RELEASE','Z','LONG1','NEARUSDT','long',reservation_id='z',verified='EXCHANGE_FLAT_AND_NO_ORDERS')['reason']=='owner_mismatch'
   assert x.act('RELEASE','A','LONG1','NEARUSDT','long',reservation_id='a')['reason']=='exchange_reconcile_required'
   assert x.act('COMMIT','A','LONG1','NEARUSDT','long',reservation_id='a',position_id='p1')['reason']=='committed'
   assert x.act('COMMIT','A','LONG1','NEARUSDT','long',reservation_id='a',position_id='p1')['reason']=='already_committed'
   assert x.act('RELEASE','A','LONG1','NEARUSDT','long',reservation_id='a',position_id='p1',verified='EXCHANGE_FLAT_AND_NO_ORDERS')['reason']=='released_committed'
   new=x.act('RESERVE','C','LONG1','NEARUSDT','long',reservation_id='c')
   assert new['state']=='READY'
   assert x.act('RELEASE','A','LONG1','NEARUSDT','long',reservation_id='a',position_id='p1',verified='EXCHANGE_FLAT_AND_NO_ORDERS')['reason']=='owner_mismatch'
   assert x.act('RELEASE','C','LONG1','NEARUSDT','long',reservation_id='c',verified='EXCHANGE_FLAT_AND_NO_ORDERS')['reason']=='released_uncommitted'
   assert x.act('RELEASE','B','SHORT1','INJUSDT','short',reservation_id='b',verified='EXCHANGE_FLAT_AND_NO_ORDERS')['reason']=='released_uncommitted'
   # Atomically gate CLOSED before restart; unknown runtime must freeze.
   assert x.act('CLOSE_GATE')['reason']=='closed'
   assert x.act('RESERVE','D','LONG1','NEARUSDT','long',reservation_id='d')['reason']=='gate_closed_or_restarted'
   # Test separate pool of concurrent requests at one strategy slot.
   assert x.act('OPEN_GATE',verified='ISOLATED_TEST_RECONCILED')['state']=='READY'
   with ThreadPoolExecutor(max_workers=20) as pool:
    outcomes=list(pool.map(lambda i:x.act('RESERVE',f'rush{i}','SHORT1','INJUSDT','short',
      reservation_id=f'tok{i}'),range(60)))
   assert sum(o['state']=='READY' for o in outcomes)==1
   assert r.hlen(x.keys[1])==1
   # Old request ID exact replay, even when another request is blocked.
   i=next(j for j,o in enumerate(outcomes) if o['state']=='READY')
   assert x.act('RESERVE',f'rush{i}','SHORT1','INJUSDT','short',reservation_id=f'tok{i}')==outcomes[i]
   # Expired reservations must remain blocked until actual reconciliation.
   assert x.act('RESERVE','postexpiry','SHORT1','INJUSDT','short',
                reservation_id='postexpiry',now_ms=1000000000)['reason']=='strategy_slot_busy'
   # Gate cannot be reopened while unresolved risk is in the isolated hash.
   assert x.act('OPEN_GATE',verified='ISOLATED_TEST_RECONCILED')['reason']=='unresolved_risk_state'
   # Close gate, restart Redis without persistence. No reservation survives,
   # but a restarted gate is CLOSED and cannot admit new risk until reconciled.
   stop(proc,r)
   proc=None
   # On network/Redis loss there is NO fall-through to READY or BYPASS.
   failed=False
   try:
    x.act('RESERVE','offline','LONG1','NEARUSDT','long',reservation_id='offline')
   except (redis.exceptions.RedisError,OSError):
    failed=True
   assert failed
   proc,r=boot(folder);x=IsolatedRedisRiskV2(r)
   assert r.hlen(x.keys[1])==0
   assert x.act('STATUS')['reason']=='CLOSED'
   assert x.act('RESERVE','after','LONG1','NEARUSDT','long',reservation_id='after')['reason']=='gate_closed_or_restarted'
   print(json.dumps(dict(
     isolated_server='unix_socket_only_port_0_no_persistence',
     atomic_contenders=60,successful_reservations=1,
     initial_two_bot_notional=92,
     duplicate_request_same_response=True,
     stale_release_blocked=True,
     owner_required=True,
     commit_idempotent=True,
     release_requires_external_flat_proof_flag=True,
     redis_restart_gate_closed=True,
     real_exchange_reconciliation_executed=False,
     real_redis_touched=False,
     live_service_restarted=False,
     overall='PASS'
   ),indent=2))
  finally:
   if proc is not None:stop(proc,r)
if __name__=='__main__':run()
