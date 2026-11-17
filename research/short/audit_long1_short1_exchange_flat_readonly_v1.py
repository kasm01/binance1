#!/usr/bin/env python3
"""Read-only signed Binance position/open order/algo plus local Redis consistency.

No Redis writes, no Binance order placement or deletion, no risk release.
"""
from pathlib import Path
import json,time
from dotenv import load_dotenv
load_dotenv('.env')
load_dotenv('.runtime_secrets.env',override=False)
import redis
from core.binance_futures_v3 import position_risk,signed_get
from orchestration.risk.exchange_flat_readonly_reconcile_v1 import read_only_snapshot

def normal(symbol):
 return signed_get('/fapi/v1/openOrders',{'symbol':symbol})
def algo(symbol):
 return signed_get('/fapi/v1/openAlgoOrders',{'symbol':symbol})
snapshots=[]
for _ in range(2):
 snapshots.append(read_only_snapshot(position_risk,normal,algo))
 if len(snapshots)==1:time.sleep(1)
r=redis.Redis(decode_responses=True,socket_timeout=2)
local={}
for symbol in ('NEARUSDT','INJUSDT'):
 key='bot:positions:'+symbol
 try:
  raw=r.get(key)
  obj=json.loads(raw) if raw else None
  local[symbol]=dict(redis_position_present=obj is not None,
                     symbol=symbol,
                     local_qty_present=(obj.get('qty') is not None if isinstance(obj,dict) else None))
 except Exception as exc:
  local[symbol]=dict(redis_status='UNKNOWN',error=type(exc).__name__)
try:
 raw=r.hgetall('central:risk:dry:reservations:v1')
 reservations=dict(state='OK',count=len(raw),strategies=sorted(raw))
except Exception as exc:
 reservations=dict(state='UNKNOWN',error=type(exc).__name__)
try:
 raw=r.get('telemetry:central_risk:shadow:current')
 obj=json.loads(raw) if raw else {}
 age=time.time()-float(obj.get('ts') or 0)
 risk_snapshot=dict(state='OK' if raw and 0<=age<=420 else 'UNKNOWN',
                    age_seconds=round(age,2),
                    health=(obj.get('exposure_health') or {}).get('status'),
                    position_count=len(obj.get('positions') or []))
except Exception as exc:
 risk_snapshot=dict(state='UNKNOWN',error=type(exc).__name__)
is_two_flat=all(x['overall_state']=='FLAT_CANDIDATE' for x in snapshots)
is_local_clear=all(not z.get('redis_position_present',True) for z in local.values())
is_reservation_clear=reservations.get('state')=='OK' and reservations.get('count')==0
result=dict(version='read_only_broker_local_reconciliation.v1',
    broker_observations=snapshots,
    broker_twice_flat_candidate=is_two_flat,
    local_positions=local,
    central_dry_reservations=reservations,
    central_snapshot=risk_snapshot,
    all_readonly_sources_agree_flat_candidate=bool(is_two_flat and is_local_clear and is_reservation_clear and risk_snapshot.get('state')=='OK' and risk_snapshot.get('health')=='OK' and risk_snapshot.get('position_count')==0),
    safe_to_auto_release=False,
    risk_master_ready=False,
    authorization='NONE',
    warning='No full order history, fills, reconciliation generation or broker-atomic snapshot. Pending unknown outcome must never auto-release on this observation.')
out=Path('research/results/LONG1_SHORT1_BINANCE_REDIS_READONLY_RECONCILE_20261114.json')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(
  broker_observations=[{sym:z['state'] for sym,z in row['symbols'].items()} for row in snapshots],
  broker_twice_flat_candidate=result['broker_twice_flat_candidate'],
  redis_position_present={sym:z.get('redis_position_present') for sym,z in local.items()},
  reservations=reservations,risk_snapshot=risk_snapshot,
  all_readonly_sources_agree_flat_candidate=result['all_readonly_sources_agree_flat_candidate'],
  safe_to_auto_release=False,report=str(out)
),indent=2))
