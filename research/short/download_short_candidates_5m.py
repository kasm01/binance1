#!/usr/bin/env python3
from pathlib import Path
import time, requests, pandas as pd
SYMS=['INJUSDT','SEIUSDT','ARBUSDT','OPUSDT','APTUSDT','SUIUSDT']
URL='https://fapi.binance.com/fapi/v1/klines'; INTERVAL='5m'; LIMIT=1500
END=int(time.time()*1000); START=END-360*24*60*60*1000
OUT=Path('research/short/data/candidate_5m_360d'); OUT.mkdir(parents=True,exist_ok=True)
cols=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
for sym in SYMS:
    rows=[]; cur=START; req=0
    while cur<END:
        r=requests.get(URL,params={'symbol':sym,'interval':INTERVAL,'startTime':cur,'endTime':END,'limit':LIMIT},timeout=15)
        if r.status_code!=200:
            print(sym,'HTTP',r.status_code,r.text[:180],flush=True); break
        z=r.json(); req+=1
        if not z: break
        rows.extend(z); nxt=int(z[-1][0])+300_000
        if nxt<=cur: break
        cur=nxt; time.sleep(.03)
    if rows:
        df=pd.DataFrame(rows,columns=cols).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        fp=OUT/f'{sym}_5m_360d.csv'; df.to_csv(fp,index=False)
        print(sym,'rows',len(df),'req',req,'start',pd.to_datetime(df.open_time.iloc[0],unit='ms',utc=True),'end',pd.to_datetime(df.close_time.iloc[-1],unit='ms',utc=True),'file',fp,flush=True)
