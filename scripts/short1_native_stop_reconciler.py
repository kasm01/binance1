#!/usr/bin/env python3
"""SHORT1 INJUSDT native Algo stop reconciler, staged, read-only by default.

IMPORTANT: not installed as an always-on service until production permission.
"""
from __future__ import annotations
import argparse,os,sys,time
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT/'.env')
load_dotenv(ROOT/'.runtime_secrets.env',override=False)
from core.binance_futures_v3 import position_risk,_base_url
from core.short_native_algo_stop import get_open,is_short_close_stop,cancel_owned,place,PREFIX
import requests

SYMBOL='INJUSDT'
def expected_native_stop(entry:str, leverage:str, tick:str, roi:str):
    p=Decimal(str(entry));lev=Decimal(str(leverage))
    step=Decimal(str(tick));r=Decimal(str(roi))
    if p<=0 or lev<=0 or step<=0 or not -1<r<0:raise ValueError('invalid_short_stop_inputs')
    raw=p*(Decimal(1)+abs(r)/lev)
    return (raw/step).to_integral_value(rounding=ROUND_DOWN)*step

def state(*,apply=False):
    if apply and os.getenv('DRY_RUN','1').lower() not in ('0','false','off'):
        raise RuntimeError('DRY_RUN_ACTIVE_REAL_ORDER_MUTATIONS_FORBIDDEN')
    positions=position_risk(SYMBOL)
    longs=[p for p in positions if p.get('symbol')==SYMBOL and
           p.get('positionSide')=='SHORT' and float(p.get('positionAmt') or 0)<0]
    if len(longs)>1:raise RuntimeError('ambiguous_short_positions')
    orders=get_open(SYMBOL)
    tagged=[o for o in orders if str(o.get('clientAlgoId') or '').startswith(PREFIX)]
    if not longs:
        if not tagged:return {'status':'FLAT_CLEAN'}
        if not apply:return {'status':'FLAT_STALE_TAGGED','count':len(tagged)}
        for x in tagged:
            result=cancel_owned(SYMBOL,x.get('algoId'))
            if result.get('status')!='canceled':raise RuntimeError('tagged_stale_cancel_unverified')
        return {'status':'FLAT_CLEANED','count':len(tagged)}
    pos=longs[0]
    info=requests.get(_base_url()+'/fapi/v1/exchangeInfo',timeout=8).json()
    filters=next(s.get('filters') for s in info.get('symbols',[]) if s.get('symbol')==SYMBOL)
    tick=next(f['tickSize'] for f in filters if f.get('filterType')=='PRICE_FILTER')
    trigger=expected_native_stop(pos['entryPrice'],pos['leverage'],tick,os.getenv('ROI_HARD_STOP_PCT','-.12'))
    mark=Decimal(str(pos['markPrice']))
    valid=[o for o in orders if is_short_close_stop(o,SYMBOL) and
           str(o.get('clientAlgoId') or '').startswith(PREFIX) and
           Decimal(str(o.get('triggerPrice') or 0))==trigger]
    if len(valid)==1 and len(tagged)==1:
        return {'status':'PROTECTED','stop':str(trigger),'algoId':valid[0].get('algoId')}
    if orders:
        # Safe default: no dynamic replacement while another Algo order exists.
        return {'status':'MANUAL_REVIEW_ALGO_EXISTS','count':len(orders),'expected_stop':str(trigger)}
    if mark>=trigger:
        return {'status':'PRICE_ALREADY_PAST_STOP','expected_stop':str(trigger)}
    if not apply:
        return {'status':'MISSING_STOP','expected_stop':str(trigger)}
    out=place(SYMBOL,float(trigger))
    if out.get('status')!='placed':raise RuntimeError('short_stop_creation_unverified')
    return {'status':'PROTECTED_CREATED','stop':str(trigger),'algoId':out['order'].get('algoId')}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--apply',action='store_true')
    p.add_argument('--loop',action='store_true')
    p.add_argument('--interval',type=int,default=45)
    a=p.parse_args()
    if a.loop and not 30<=a.interval<=300:raise SystemExit('bad_interval')
    while True:
        try:print(state(apply=a.apply),flush=True)
        except Exception as e:
            print('RECONCILER_ERROR',type(e).__name__,str(e)[:220],flush=True)
            if not a.loop:return 2
        if not a.loop:return 0
        time.sleep(a.interval)
if __name__=='__main__':raise SystemExit(main())
