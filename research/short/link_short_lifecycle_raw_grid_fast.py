#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache/LINKUSDT_1m_360d.csv')
OUT=Path('/home/kasm920/binance1/research/short/results/LINKUSDT_SHORT_LIFECYCLE_RAW_GRID.csv')
FEE=.0008
TPS=[.004,.006,.008,.012]; SLS=[.006,.010,.015,.020]; HS=[30,60,120]
d=pd.read_csv(SRC,usecols=['open_time','open','high','low','close'])
ot=d.open_time.to_numpy(np.int64); o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float); cl=d.close.to_numpy(float)
dt=pd.to_datetime(ot,unit='ms',utc=True); mins=dt.minute.to_numpy(); anchors0=np.flatnonzero((mins%5)==0)
acc={}
for H in HS:
    a=anchors0[anchors0+H<len(d)]
    months=dt[a].strftime('%Y-%m').to_numpy()
    for tp in TPS:
        for sl in SLS:
            acc[(tp,sl,H)]={'v':[],'state':[],'dur':[],'month':[]}
    CH=4000
    for start in range(0,len(a),CH):
        aa=a[start:start+CH]
        e=o[aa]
        idx=aa[:,None]+np.arange(H)[None,:]
        adv=hi[idx]/e[:,None]-1.0
        fav=1.0-lo[idx]/e[:,None]
        timeout=1.0-cl[aa+H-1]/e
        mm=months[start:start+len(aa)]
        for tp in TPS:
          tp_hit=fav>=tp; tp_any=tp_hit.any(axis=1); tp_i=np.where(tp_any,tp_hit.argmax(axis=1),H+1)
          for sl in SLS:
            sl_hit=adv>=sl; sl_any=sl_hit.any(axis=1); sl_i=np.where(sl_any,sl_hit.argmax(axis=1),H+1)
            is_sl=(sl_i<=tp_i)&(sl_i<=H-1)
            is_tp=(tp_i<sl_i)&(tp_i<=H-1)
            gross=np.where(is_sl,-sl,np.where(is_tp,tp,timeout))
            v=gross-FEE
            st=np.where(is_sl,2,np.where(is_tp,1,0)).astype(np.int8) # 2 SL,1 TP,0 timeout
            dur=np.where(is_sl,sl_i+1,np.where(is_tp,tp_i+1,H)).astype(np.int16)
            z=acc[(tp,sl,H)]; z['v'].append(v); z['state'].append(st); z['dur'].append(dur); z['month'].append(mm)
rows=[]
for (tp,sl,H),z in acc.items():
    v=np.concatenate(z['v']); st=np.concatenate(z['state']); dur=np.concatenate(z['dur']); months=np.concatenate(z['month'])
    pos=v[v>0].sum(); neg=-v[v<0].sum(); ser=pd.Series(v).groupby(pd.Series(months)).mean()
    rows.append(dict(tp=tp,sl=sl,horizon_min=H,n=len(v),net_bps=v.mean()*1e4,gross_bps=(v.mean()+FEE)*1e4,pf=pos/max(neg,1e-12),win_rate=(v>0).mean(),tp_rate=(st==1).mean(),sl_rate=(st==2).mean(),timeout_rate=(st==0).mean(),median_dur=float(np.median(dur)),positive_months=int((ser>0).sum()),months=len(ser),median_month_bps=float(ser.median()*1e4),worst_month_bps=float(ser.min()*1e4)))
r=pd.DataFrame(rows).sort_values('net_bps',ascending=False); r.to_csv(OUT,index=False)
print(r.head(24).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('POSITIVE_CONFIGS',int((r.net_bps>0).sum()),'/',len(r),'OUT',OUT)
