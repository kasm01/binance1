#!/usr/bin/env python3
import time
from pathlib import Path
import pandas as pd, requests
SYMBOL='INJUSDT'; INTERVAL='3m'; LIMIT=1500; STEP=180000
URL='https://fapi.binance.com/fapi/v1/klines'
OUT=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
COLS=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
now_ms=int(time.time()*1000); end_ms=(now_ms//STEP)*STEP-1; start_ms=end_ms-460*24*60*60*1000+1
s=requests.Session(); rows=[]; cur=start_ms; req=0
while cur<=end_ms:
    r=s.get(URL,params={'symbol':SYMBOL,'interval':INTERVAL,'startTime':cur,'endTime':end_ms,'limit':LIMIT},timeout=15); r.raise_for_status(); b=r.json(); req+=1
    if not b: break
    rows.extend(b); last=int(b[-1][0]); cur=last+STEP
    if req%25==0: print('REQ',req,'ROWS',len(rows),flush=True)
    if len(b)<LIMIT: break
    time.sleep(.05)
df=pd.DataFrame(rows,columns=COLS).drop_duplicates('open_time').sort_values('open_time')
df=df[(pd.to_numeric(df.open_time)>=start_ms)&(pd.to_numeric(df.open_time)<=end_ms)]
OUT.parent.mkdir(parents=True,exist_ok=True); df.to_csv(OUT,index=False)
print('WROTE',OUT,'ROWS',len(df),'REQ',req)
print('FIRST',pd.to_datetime(int(df.open_time.iloc[0]),unit='ms',utc=True),'LAST',pd.to_datetime(int(df.close_time.iloc[-1]),unit='ms',utc=True))
# continuity audit
ot=pd.to_numeric(df.open_time).to_numpy('int64'); gaps=(ot[1:]-ot[:-1])!=STEP; print('GAPS',int(gaps.sum()))
