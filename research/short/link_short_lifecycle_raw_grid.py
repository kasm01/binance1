#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache/LINKUSDT_1m_360d.csv')
OUT=Path('/home/kasm920/binance1/research/short/results/LINKUSDT_SHORT_LIFECYCLE_RAW_GRID.csv')
FEE=.0008
d=pd.read_csv(SRC,usecols=['open_time','open','high','low','close'])
ot=d.open_time.to_numpy(np.int64); o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float); cl=d.close.to_numpy(float)
dt=pd.to_datetime(ot,unit='ms',utc=True)
anchors=np.flatnonzero((dt.minute.to_numpy()%5)==0)
rows=[]
for tp in [.004,.006,.008,.012]:
 for sl in [.006,.010,.015,.020]:
  for H in [30,60,120]:
   a=anchors[anchors+H<len(d)]
   vals=[]; states=[]; durs=[]; months=[]
   for j in a:
    e=o[j]; gross=None; state='TIMEOUT'; dur=H
    # conservative ambiguity: adverse SL checked first
    for k in range(H):
     ix=j+k
     if hi[ix]>=e*(1+sl): gross=-sl; state='SL'; dur=k+1; break
     if lo[ix]<=e*(1-tp): gross=tp; state='TP'; dur=k+1; break
    if gross is None: gross=1-cl[j+H-1]/e
    vals.append(gross-FEE); states.append(state); durs.append(dur); months.append(dt[j].strftime('%Y-%m'))
   v=np.asarray(vals,float); st=np.asarray(states); m=pd.Series(v).groupby(pd.Series(months)).mean()
   pos=v[v>0].sum(); neg=-v[v<0].sum();
   rows.append(dict(tp=tp,sl=sl,horizon_min=H,n=len(v),net_bps=v.mean()*1e4,gross_bps=(v.mean()+FEE)*1e4,pf=pos/max(neg,1e-12),win_rate=(v>0).mean(),tp_rate=(st=='TP').mean(),sl_rate=(st=='SL').mean(),timeout_rate=(st=='TIMEOUT').mean(),median_dur=float(np.median(durs)),positive_months=int((m>0).sum()),months=len(m),median_month_bps=float(m.median()*1e4),worst_month_bps=float(m.min()*1e4)))
r=pd.DataFrame(rows).sort_values('net_bps',ascending=False); r.to_csv(OUT,index=False)
print(r.head(20).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('POSITIVE_CONFIGS',int((r.net_bps>0).sum()),'/',len(r),'OUT',OUT)
