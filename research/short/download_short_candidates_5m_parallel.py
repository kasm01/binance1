#!/usr/bin/env python3
from pathlib import Path
import os, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
SYMS=[x.strip().upper() for x in os.getenv('SHORT_DOWNLOAD_SYMBOLS','INJUSDT,SEIUSDT,ARBUSDT,OPUSDT,APTUSDT,SUIUSDT').split(',') if x.strip()]
URL='https://fapi.binance.com/fapi/v1/klines'; INTERVAL='5m'; LIMIT=1500
END=int(time.time()*1000); START=END-360*24*60*60*1000
OUT=Path('research/short/data/candidate_5m_360d'); OUT.mkdir(parents=True,exist_ok=True)
cols=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
def one(sym):
 rows=[];cur=START;req=0;s=requests.Session()
 while cur<END:
  r=s.get(URL,params={'symbol':sym,'interval':INTERVAL,'startTime':cur,'endTime':END,'limit':LIMIT},timeout=15)
  if r.status_code==429:
   time.sleep(2); continue
  r.raise_for_status();z=r.json();req+=1
  if not z:break
  rows.extend(z);nxt=int(z[-1][0])+300_000
  if nxt<=cur:break
  cur=nxt;time.sleep(.08)
 df=pd.DataFrame(rows,columns=cols).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
 fp=OUT/f'{sym}_5m_360d.csv';df.to_csv(fp,index=False)
 return sym,len(df),req,pd.to_datetime(df.open_time.iloc[0],unit='ms',utc=True),pd.to_datetime(df.close_time.iloc[-1],unit='ms',utc=True),str(fp)
with ThreadPoolExecutor(max_workers=3) as ex:
 fut={ex.submit(one,s):s for s in SYMS}
 for f in as_completed(fut):
  try: print('DONE',*f.result(),flush=True)
  except Exception as e: print('ERROR',fut[f],repr(e),flush=True)
