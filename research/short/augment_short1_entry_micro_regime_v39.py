#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path('/home/kasm920/binance1'); SRC=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'; ONE=ROOT/'research/short/data/inj_1m'; OUT=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V39.csv'
parts=[pd.read_csv(p) for p in sorted(ONE.glob('*.csv'))]; m=pd.concat(parts,ignore_index=True)
m['ts']=pd.to_datetime(pd.to_numeric(m.open_time),unit='ms',utc=True)
for c in ['open','high','low','close','quote_volume','count']:m[c]=pd.to_numeric(m[c],errors='coerce')
m=m.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
m['range1']=(m.high-m.low)/m.open.replace(0,np.nan);m['retcc']=m.close.pct_change();m['avg_trade_quote']=m.quote_volume/m['count'].replace(0,np.nan)
for w in [20,60]:
 m[f'count_mean{w}_prev']=m['count'].rolling(w,min_periods=max(5,w//4)).mean().shift(1)
 m[f'range_med{w}_prev']=m.range1.rolling(w,min_periods=max(5,w//4)).median().shift(1)
 m[f'rv{w}_prev']=m.retcc.rolling(w,min_periods=max(5,w//4)).std(ddof=0).shift(1)
 m[f'avg_trade_quote_med{w}_prev']=m.avg_trade_quote.rolling(w,min_periods=max(5,w//4)).median().shift(1)
base=m[['ts']+[c for c in m.columns if c.endswith('_prev')]].copy()
d=pd.read_csv(SRC);d['confirm_ts']=pd.to_datetime(d.confirm_ts,utc=True);d=d.merge(base,left_on='confirm_ts',right_on='ts',how='left').drop(columns=['ts'])
for sec in [5,10,15,20,30]:
 d[f'trade_pace20_{sec}s']=d[f'trade_count_{sec}s']*(60/sec)/d.count_mean20_prev
 d[f'trade_pace60_{sec}s']=d[f'trade_count_{sec}s']*(60/sec)/d.count_mean60_prev
 d[f'range_rel20_{sec}s']=d[f'range_{sec}s']/d.range_med20_prev
 d[f'range_rel60_{sec}s']=d[f'range_{sec}s']/d.range_med60_prev
 d[f'ret_vol20_{sec}s']=d[f'ret_{sec}s']/d.rv20_prev
 d[f'ret_vol60_{sec}s']=d[f'ret_{sec}s']/d.rv60_prev
 d[f'trade_size_rel20_{sec}s']=d[f'avg_trade_quote_{sec}s']/d.avg_trade_quote_med20_prev
 d[f'trade_size_rel60_{sec}s']=d[f'avg_trade_quote_{sec}s']/d.avg_trade_quote_med60_prev
for a,b in [(5,10),(10,20),(5,20),(15,30),(5,30),(20,30)]:
 d[f'ret_delta_vol20_{a}_{b}']=d[f'ret_delta_{a}_{b}']/d.rv20_prev
 d[f'ret_delta_vol60_{a}_{b}']=d[f'ret_delta_{a}_{b}']/d.rv60_prev
d.to_csv(OUT,index=False);print('saved',OUT,'rows',len(d),'cols',len(d.columns))
