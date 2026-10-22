#!/usr/bin/env python3
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, requests

URL='https://fapi.binance.com/fapi/v1/klines'
COLS=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
SYMS=['LINKUSDT','AVAXUSDT','INJUSDT']
INTERVAL='1m'; LIMIT=1500
EVENTS=Path('research/short/results/SHORT_REAL_CONFIRM_EVENTS.csv')
OUTDIR=Path('research/short/data/real_confirm_window_1m')

def fetch(sym,start_ms,end_ms):
    ses=requests.Session(); rows=[]; cur=start_ms; req=0
    while cur<=end_ms:
        r=ses.get(URL,params={'symbol':sym,'interval':INTERVAL,'startTime':cur,'endTime':end_ms,'limit':LIMIT},timeout=15)
        r.raise_for_status(); batch=r.json(); req+=1
        if not batch: break
        rows.extend(batch); last=int(batch[-1][0]);
        if last<cur: raise RuntimeError('non-advancing cursor')
        cur=last+1
        if req%15==0: print(sym,'REQ',req,'rows',len(rows),flush=True)
        if len(batch)<LIMIT: break
        time.sleep(.06)
    d=pd.DataFrame(rows,columns=COLS).drop_duplicates('open_time').sort_values('open_time')
    d=d[(pd.to_numeric(d.open_time)>=start_ms)&(pd.to_numeric(d.open_time)<=end_ms)]
    return d,req

def main():
    e=pd.read_csv(EVENTS); e['ts']=pd.to_datetime(e.ts,utc=True)
    # 2 minutes before first telemetry and 3 hours after last for lifecycle outcomes.
    start_ms=int((e.ts.min()-pd.Timedelta(minutes=2)).timestamp()*1000)//60000*60000
    end_ms=int((e.ts.max()+pd.Timedelta(hours=3)).timestamp()*1000)//60000*60000
    expected=(end_ms-start_ms)//60000+1
    print('WINDOW',pd.to_datetime(start_ms,unit='ms',utc=True),pd.to_datetime(end_ms,unit='ms',utc=True),'EXPECTED',expected,flush=True)
    OUTDIR.mkdir(parents=True,exist_ok=True)
    for sym in SYMS:
        d,req=fetch(sym,start_ms,end_ms)
        print(sym,'ROWS',len(d),'REQ',req,'FIRST',int(d.open_time.iloc[0]),'LAST',int(d.open_time.iloc[-1]),flush=True)
        if len(d)!=expected: raise RuntimeError(f'{sym} row mismatch {len(d)} != {expected}')
        p=OUTDIR/f'{sym}_1m.csv'; d.to_csv(p,index=False); print('WROTE',p,flush=True)
if __name__=='__main__': main()
