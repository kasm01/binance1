#!/usr/bin/env python3
from pathlib import Path
import time, requests, pandas as pd
from datetime import datetime, timezone
URL='https://fapi.binance.com/fapi/v1/klines'; SYMBOL='LINKUSDT'; INTERVAL='1m'; LIMIT=1500
START=int(pd.Timestamp('2026-09-14 20:22:00',tz='UTC').timestamp()*1000)
END=int(time.time()*1000)
out=[]; cur=START
while cur<END:
 r=requests.get(URL,params={'symbol':SYMBOL,'interval':INTERVAL,'startTime':cur,'endTime':END,'limit':LIMIT},timeout=15); r.raise_for_status(); rows=r.json()
 if not rows: break
 out.extend(rows)
 nxt=int(rows[-1][0])+60_000
 if nxt<=cur: break
 cur=nxt
 time.sleep(.05)
cols=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume','ignore']
df=pd.DataFrame(out,columns=cols).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
path=Path('research/short/data');path.mkdir(parents=True,exist_ok=True);fp=path/'LINKUSDT_1m_FORWARD_20260914.csv';df.to_csv(fp,index=False)
print('ROWS',len(df),'START',pd.to_datetime(df.open_time.iloc[0],unit='ms',utc=True),'END',pd.to_datetime(df.close_time.iloc[-1],unit='ms',utc=True),'FILE',fp)
