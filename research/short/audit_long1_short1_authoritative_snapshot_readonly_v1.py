#!/usr/bin/env python3
"""Bounded signed Binance account/position reads vs central Redis snapshot.

Never produces executable policy or writes Redis/Binance. No secrets persisted.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,time,redis
from dotenv import load_dotenv
load_dotenv('.env');load_dotenv('.runtime_secrets.env',override=False)
from core.binance_futures_v3 import account,position_risk
from orchestration.risk.authoritative_snapshot_evidence_v1 import verify
r=redis.Redis(decode_responses=True,socket_timeout=3)
start=time.monotonic()
try:
    a=account()
    p=position_risk()
    raw=r.get('telemetry:central_risk:shadow:current')
    local=json.loads(raw) if raw else {}
    result=verify(a,p,local,now_sec=time.time(),elapsed_sec=time.monotonic()-start,
                  max_snapshot_age_sec=120,max_capture_age_sec=15)
    if result['state']=='CONSISTENT_READONLY':
        try:
            reservation_hash=r.hgetall('central:risk:dry:reservations:v1')
            result['central_dry_pending_reservation_count']=len(reservation_hash)
            if reservation_hash:
                result['state']='UNKNOWN'
                result['reason']='central_risk_reservation_not_empty'
        except Exception:
            result['state']='UNKNOWN';result['reason']='reservation_source_unavailable'
except Exception as err:
    result=dict(state='UNKNOWN',reason='signed_source_failure:'+type(err).__name__,
                can_authorize_new_reservation=False,can_auto_commit=False,can_auto_release=False)
# Diagnostic only. Do not persist raw account data, API credentials, order IDs or symbols beyond own scope.
result.update(audited_at_utc=datetime.now(timezone.utc).isoformat(),source='READONLY_SIGNED_BINANCE_AND_CENTRAL_SNAPSHOT',
              risk_master_authorized=False,no_exchange_order_sent=True)
out=Path('research/results/LONG1_SHORT1_CENTRAL_RISK_ACCOUNT_SNAPSHOT_CROSSCHECK_20261114.json')
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
