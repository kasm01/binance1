#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
import redis

SRC=os.getenv('SHORT1_HAZARD_SOURCE_STREAM','telemetry:short1:inj:model_shadow_v5')
OUT=os.getenv('SHORT1_HAZARD_OUT_STREAM','telemetry:short1:inj:entry_hazard_v1_outcomes')
DONE=os.getenv('SHORT1_HAZARD_DONE_SET','telemetry:short1:inj:entry_hazard_v1:finalized')
SYMBOL='INJUSDT'; FAPI='https://fapi.binance.com'
COLS=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']

def fetch_1m(start_ms:int,end_ms:int)->pd.DataFrame:
    sess=requests.Session(); rows=[]; cur=start_ms
    while cur<=end_ms:
        x=sess.get(f'{FAPI}/fapi/v1/klines',params={'symbol':SYMBOL,'interval':'1m','startTime':cur,'endTime':end_ms,'limit':1500},timeout=8)
        x.raise_for_status(); b=x.json()
        if not b: break
        rows.extend(b); cur=int(b[-1][0])+60000
        if len(b)<1500: break
    return pd.DataFrame(rows,columns=COLS).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True) if rows else pd.DataFrame(columns=COLS)

def label_window(d:pd.DataFrame, entry_minute_ms:int, entry_px:float, n:int)->Dict[str,Any]:
    z=d[(pd.to_numeric(d.open_time)>=entry_minute_ms)&(pd.to_numeric(d.open_time)<entry_minute_ms+n*60000)]
    if z.empty: return {'available':False}
    hi=pd.to_numeric(z.high).to_numpy(float); lo=pd.to_numeric(z.low).to_numpy(float)
    adverse=hi/entry_px-1.0; favorable=1.0-lo/entry_px
    adv40=np.flatnonzero(adverse>=0.004); fav40=np.flatnonzero(favorable>=0.004)
    ia=int(adv40[0]) if len(adv40) else None; iff=int(fav40[0]) if len(fav40) else None
    if ia is None and iff is None: touch='NEITHER'
    elif ia is None: touch='FAVORABLE_FIRST'
    elif iff is None: touch='ADVERSE_FIRST'
    elif ia<iff: touch='ADVERSE_FIRST'
    elif iff<ia: touch='FAVORABLE_FIRST'
    else: touch='BOTH_SAME_BAR'
    return {'available':True,'mae_pct':float(np.nanmax(adverse)),'mfe_pct':float(np.nanmax(favorable)),'touch_40bp':touch}

def main():
    r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
    cutoff=time.time()-6*60
    pending=[]
    for sid,f in r.xrevrange(SRC,count=5000):
        try:
            o=json.loads(f.get('json','{}')); eid=str(o.get('event_id') or '')
            if not eid or r.sismember(DONE,eid): continue
            if o.get('confirmation_state')!='CONFIRM': continue
            hs=o.get('entry_hazard_snapshot') if isinstance(o.get('entry_hazard_snapshot'),dict) else {}
            if not hs.get('available'): continue
            if float(o.get('ts') or 0)>cutoff: continue
            entry=float((o.get('confirmation') or {}).get('curr_close') or 0)
            if entry<=0: continue
            pending.append((sid,o,entry))
        except Exception: continue
    if not pending:
        print('[ENTRY-HAZARD-FINALIZER] pending=0'); return
    tms=[int(float(o.get('ts'))*1000)//60000*60000 for _,o,_ in pending]
    d=fetch_1m(min(tms),max(tms)+6*60000)
    wrote=0
    for sid,o,entry in pending:
        minute=int(float(o.get('ts'))*1000)//60000*60000
        labels={f'{n}m':label_window(d,minute,entry,n) for n in (1,3,5)}
        out={'schema_version':'entry_hazard_outcome.v1','event_id':o['event_id'],'source_stream_id':sid,'source_ts':float(o['ts']),'finalized_ts':time.time(),'symbol':o.get('symbol'),'economic_probability':float(o.get('economic_probability') or 0),'confirmation_state':'CONFIRM','entry_price':entry,'entry_hazard_snapshot':o.get('entry_hazard_snapshot'),'labels':labels}
        rid=r.xadd(OUT,{'json':json.dumps(out,separators=(',',':'))},maxlen=20000,approximate=True)
        if rid: r.sadd(DONE,o['event_id']); wrote+=1
    print(f'[ENTRY-HAZARD-FINALIZER] pending={len(pending)} wrote={wrote} source={SRC} out={OUT}')
if __name__=='__main__': main()
