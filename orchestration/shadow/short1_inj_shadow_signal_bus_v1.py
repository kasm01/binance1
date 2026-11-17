#!/usr/bin/env python3
"""SHORT1 INJUSDT shadow-only model -> PENDING -> READY observation bus.

Reads existing V5/V6 economic model telemetry, writes ONLY shadow:short1:inj:*
state and streams. No Binance calls, no production signals, no executor import,
no orders. V6 is the primary score; V5 is a separate advisory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import redis
from dotenv import load_dotenv
from core.short_pending_freshness_contract import check_independent_model_revalidation

load_dotenv(ROOT/'.env')
V5='telemetry:short1:inj:model_shadow_v5'
V6='telemetry:short1:inj:model_shadow_v6'
BUS='shadow:short1:inj:signal_bus:v1'
TRANSITIONS='shadow:short1:inj:pending_fsm:v1'
STATE='shadow:short1:inj:pending_fsm_state:v1'
CHECKPOINT='shadow:short1:inj:signal_bus_last_close_ms:v1'
MODEL_PRIMARY='short_arm_quality_logistic_v6'
BAR_MS=180000
PENDING_TTL_MS=15*60*1000
LIVE_FRESH_MS=8*60*1000

def check_shadow_only() -> None:
    if str(os.getenv('DRY_RUN','1')).lower() not in ('1','true','yes','on'):
        raise RuntimeError('SHORT1 shadow bus refuses DRY_RUN=0')
    if str(os.getenv('ARMED','0')).lower() not in ('0','false','no','off'):
        raise RuntimeError('SHORT1 shadow bus refuses ARMED=1')
    if str(os.getenv('SYMBOL','')).upper()!='INJUSDT':
        raise RuntimeError('SHORT1 shadow bus requires INJUSDT')
    if str(os.getenv('STRATEGY_ID','')).upper()!='SHORT1':
        raise RuntimeError('SHORT1 shadow bus requires SHORT1')

def score_valid(e:dict[str,Any]) -> bool:
    try:
        score=float(e['economic_probability'])
        threshold=float(e['reference_threshold'])
        return (math.isfinite(score) and math.isfinite(threshold)
                and 0 <=score<=1 and 0<threshold<=1)
    except (KeyError,TypeError,ValueError):
        return False

def input_valid(e:dict[str,Any], *, version:str) -> bool:
    if not isinstance(e,dict):return False
    if e.get('routing')!='OFF' or e.get('symbol')!='INJUSDT' or e.get('strategy_id')!='SHORT1':
        return False
    if e.get('interval')!='3m' or not score_valid(e):
        return False
    if not e.get('model_feature_hash') or not e.get('event_id') or not e.get('model_provider'):
        return False
    if version=='v6' and e.get('model_provider')!=MODEL_PRIMARY:
        return False
    try:
        stamp=int(e['completed_close_time'])
        return stamp>0 and (stamp+1)%BAR_MS==0
    except (TypeError,ValueError):
        return False

def construct_model_candidate(e:dict[str,Any])->dict[str,Any]:
    event_id=str(e['event_id'])
    # Stable shadow-only intent key, never a real executor intent
    intent_id='sh1_'+hashlib.sha256(('INJUSDT:v6:'+event_id).encode()).hexdigest()[:24]
    return dict(
        intent_id=intent_id,
        source_signal_event_id=event_id,
        completed_close_ms=int(e['completed_close_time']),
        model_provider=str(e['model_provider']),
        model_score=float(e['economic_probability']),
        model_score_gate_threshold=float(e['reference_threshold']),
        model_score_gate_passed=float(e['economic_probability'])>=float(e['reference_threshold']),
    )

def process(e6:dict, e5:dict, state:dict, *, now_ms:int)->tuple[dict,dict]:
    """Pure finite-state logic. 'SHADOW_READY' is NEVER executable READY."""
    if not input_valid(e6,version='v6') or not input_valid(e5,version='v5'):
        raise ValueError('shadow_input_contract_invalid')
    if int(e6['completed_close_time'])!=int(e5['completed_close_time']):
        raise ValueError('closed_bar_mismatch')
    source_ms=int(e6['completed_close_time'])
    if not 0<=now_ms-source_ms<=LIVE_FRESH_MS:
        raise ValueError('event_not_fresh_or_from_future')
    cand=construct_model_candidate(e6)
    v6_pass=cand['model_score_gate_passed']
    v5_pass=float(e5['economic_probability'])>=float(e5['reference_threshold'])
    confirmed=e6.get('confirmation_state')=='CONFIRM'
    progress=float(e6.get('sniper_progress') or -1)
    in_sniper=float(e6.get('sniper_min') or 0)<=progress<=float(e6.get('sniper_max') or 0)
    confirmed=confirmed and in_sniper
    pending=state.get('pending') if isinstance(state,dict) else None
    pending=pending if isinstance(pending,dict) else None
    decision='SHADOW_REJECT';reason='v6_score_below_reference'
    next_state=dict(state or {})
    if pending and (source_ms-int(pending.get('completed_close_ms') or 0)>PENDING_TTL_MS):
        pending=None;next_state['pending']=None
    if v6_pass:
        if confirmed:
            if pending:
                origin=pending
                proof,why=check_independent_model_revalidation(origin,cand,now_ms=now_ms)
                if proof:
                    decision='SHADOW_READY_FRESH_MODEL';reason=why
                    next_state['pending']=None
                else:
                    decision='SHADOW_PENDING';reason='await_independent_model:'+why
            else:
                decision='SHADOW_READY_DIRECT';reason='v6_score_and_confirmation'
        else:
            if not pending:
                next_state['pending']=dict(cand)
                decision='SHADOW_PENDING';reason='v6_score_pass_confirmation_pending'
            else:
                decision='SHADOW_PENDING';reason='v6_score_pass_still_waiting'
    elif pending:
        decision='SHADOW_WAIT';reason='pending_wait_next_v6_score'
    obj=dict(
        strategy_id='SHORT1',symbol='INJUSDT',interval='3m',
        observed_at_ms=int(now_ms),completed_close_ms=source_ms,
        intent_id=cand['intent_id'],source_signal_event_id=cand['source_signal_event_id'],
        model_provider=cand['model_provider'],model_feature_hash=e6['model_feature_hash'],
        v6_score=round(cand['model_score'],7),v6_threshold=round(cand['model_score_gate_threshold'],7),
        v6_score_pass=bool(v6_pass),
        v5_provider=e5['model_provider'],v5_score=round(float(e5['economic_probability']),7),
        v5_threshold=round(float(e5['reference_threshold']),7),v5_pass=bool(v5_pass),
        confirmation_state=e6.get('confirmation_state'),sniper_progress=round(progress,5),
        decision=decision,reason=reason,source_routing='OFF',target_routing='SHADOW_ONLY',
        pending_origin= (pending or {}).get('intent_id'),
    )
    return obj,next_state

def recent_by_bar(r:redis.Redis,stream:str,count:int)->dict[int,dict]:
    output={}
    for sid,fields in r.xrevrange(stream,count=count):
        try:
            obj=json.loads(fields.get('json') or '{}')
            bar=int(obj.get('completed_close_time') or 0)
        except (TypeError,ValueError,json.JSONDecodeError):
            continue
        if bar and bar not in output:
            output[bar]=obj
    return output

def replay(r:redis.Redis, *, count:int=480)->dict:
    v5=recent_by_bar(r,V5,count+10)
    v6=recent_by_bar(r,V6,count+10)
    matched=sorted(set(v5)&set(v6))[-count:]
    state={}
    results=[]
    for bar in matched:
        try:
            out,state=process(v6[bar],v5[bar],state,now_ms=bar+5000)
            results.append(out)
        except ValueError:
            pass
    from collections import Counter
    counts=dict(Counter(x['decision'] for x in results))
    transitions=[x for x in results if x['decision']=='SHADOW_READY_FRESH_MODEL']
    return dict(matched=len(matched),valid=len(results),decisions=counts,
                fresh_ready=len(transitions),unique_fresh_intents=len(set(x['intent_id'] for x in transitions)),
                first_bar=matched[0] if matched else None,last_bar=matched[-1] if matched else None)

def run_once(r:redis.Redis, *, last_only_initial:bool=True)->dict:
    check_shadow_only()
    v5=recent_by_bar(r,V5,24)
    v6=recent_by_bar(r,V6,24)
    common=sorted(set(v5)&set(v6))
    if not common:return {'status':'WAIT_MATCHING_3M_BAR'}
    now_ms=int(time.time()*1000)
    # First startup must never backfill old "live" decisions from many hours
    last=int(r.get(CHECKPOINT) or 0)
    available=[b for b in common if b>last and 0<=now_ms-b<=LIVE_FRESH_MS]
    if not available:return {'status':'WAIT_NEW_3M_BAR','last':last}
    if last==0 and last_only_initial:
        available=available[-1:]
    written=[]
    for bar in available:
        e6,e5=v6[bar],v5[bar]
        # Full atomic state + checkpoint + two shadow stream writes, only
        # one worker/service unit is permitted. WATCH guards duplicate writers.
        for _ in range(3):
            pipe=r.pipeline()
            try:
                pipe.watch(CHECKPOINT,STATE)
                previous=int(pipe.get(CHECKPOINT) or 0)
                if previous>=bar:
                    pipe.unwatch();break
                old=json.loads(pipe.get(STATE) or '{}')
                out,new=process(e6,e5,old,now_ms=now_ms)
                pipe.multi()
                pipe.xadd(BUS,{'json':json.dumps(out,separators=(',',':'))},maxlen=2000,approximate=True)
                if out['decision'] in ('SHADOW_PENDING','SHADOW_READY_FRESH_MODEL','SHADOW_READY_DIRECT'):
                    pipe.xadd(TRANSITIONS,{'json':json.dumps(out,separators=(',',':'))},maxlen=1000,approximate=True)
                pipe.set(STATE,json.dumps(new,separators=(',',':')))
                pipe.set(CHECKPOINT,str(bar))
                pipe.execute()
                written.append(out)
                break
            except redis.WatchError:
                continue
            finally:
                pipe.reset()
    return dict(status='SHADOW_WRITTEN' if written else 'ALREADY_PROCESSED',
                count=len(written),
                bars=[(x['completed_close_ms'],x['decision']) for x in written])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--once',action='store_true',help='Process recent new closed bars; shadow keys only')
    parser.add_argument('--replay',action='store_true',help='Pure read-only replay of existing telemetry')
    parser.add_argument('--count',type=int,default=480)
    parser.add_argument('--poll-sec',type=int,default=20)
    args=parser.parse_args()
    check_shadow_only()
    r=redis.Redis.from_url(os.getenv('REDIS_URL','redis://localhost:6379/0'),decode_responses=True,
                            socket_connect_timeout=3,socket_timeout=3)
    r.ping()
    if args.replay:
        print(json.dumps(replay(r,count=min(max(args.count,1),1200))),flush=True);return
    if args.once:
        print(json.dumps(run_once(r)),flush=True);return
    if not 10<=args.poll_sec<=120:raise SystemExit('invalid_poll')
    print('SHORT1_INJ_SHADOW_BUS_START routing=SHADOW_ONLY v6_primary=true v5_advisory=true',flush=True)
    prev=''
    while True:
        try:
            answer=run_once(r)
            s=json.dumps(answer,sort_keys=True)
            if answer['status']!='WAIT_NEW_3M_BAR' or s!=prev:
                print(s,flush=True)
            prev=s
        except Exception as e:
            print('SHORT1_SHADOW_BUS_ERROR',type(e).__name__,str(e)[:220],flush=True)
        time.sleep(args.poll_sec)

if __name__=='__main__':main()
