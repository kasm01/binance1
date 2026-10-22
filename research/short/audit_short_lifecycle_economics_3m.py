#!/usr/bin/env python3
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd

SYMS=[x.strip().upper() for x in os.getenv('SHORT_LC_SYMBOLS','LINKUSDT,AVAXUSDT').split(',') if x.strip()]
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache'); OUT=Path('research/short/results'); OUT.mkdir(parents=True,exist_ok=True)
FEE=.0008
TPS=[.00275,.004,.006,.008,.010,.012]
SLS=[.003,.005,.008,.010,.015,.020]
HORIZONS=[6,12,20,40]  # 18m,36m,60m,120m

def run(d,tp,sl,H):
    from numpy.lib.stride_tricks import sliding_window_view
    o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float); cl=d.close.to_numpy(float)
    start_r=20
    nrows=len(d)-H-2-start_r
    if nrows <= 0:
        return dict(n=0,net_bps=np.nan,win=np.nan,pf=np.nan,tp_rate=np.nan,sl_rate=np.nan,timeout_rate=np.nan)
    ent_start=start_r+1
    e=o[ent_start:ent_start+nrows]
    hw=sliding_window_view(hi[ent_start:],H)[:nrows]
    lw=sliding_window_view(lo[ent_start:],H)[:nrows]
    sh=(hw/e[:,None]-1)>=sl
    th=(1-lw/e[:,None])>=tp
    sany=sh.any(1); tany=th.any(1)
    si=np.where(sany,sh.argmax(1),H+1)
    ti=np.where(tany,th.argmax(1),H+1)
    is_sl=(si<=ti)&(si<H)
    is_tp=(ti<si)&(ti<H)
    timeout=(~is_sl)&(~is_tp)
    timeout_exit=cl[start_r+H:start_r+H+nrows]
    tout=1-timeout_exit/e-FEE
    v=np.where(is_sl,-sl-FEE,np.where(is_tp,tp-FEE,tout))
    pos=v[v>0].sum(); neg=-v[v<0].sum()
    return dict(n=len(v),net_bps=float(v.mean()*1e4),win=float((v>0).mean()),pf=float(pos/max(neg,1e-12)),tp_rate=float(is_tp.mean()),sl_rate=float(is_sl.mean()),timeout_rate=float(timeout.mean()))

rows=[]
for sym in SYMS:
 d=pd.read_csv(SRC/f'{sym}_3m_360d.csv')
 n=len(d); hs=int(n*5/6)
 # final 60d untouched-like segment and earlier dev separately
 parts={'dev':d.iloc[:hs].reset_index(drop=True),'hold':d.iloc[hs:].reset_index(drop=True)}
 for part,dd in parts.items():
  for H in HORIZONS:
   for tp in TPS:
    for sl in SLS:
     m=run(dd,tp,sl,H); rows.append(dict(symbol=sym,part=part,H=H,tp=tp,sl=sl,fee=FEE,**m))
R=pd.DataFrame(rows);R.to_csv(OUT/'SHORT_LIFECYCLE_ECON_3M_GRID.csv',index=False)
# configs robust across both coins + dev/hold: aggregate mean and worst net
A=[]
for (H,tp,sl),g in R.groupby(['H','tp','sl']):
 A.append(dict(H=H,tp=tp,sl=sl,mean_bps=float(g.net_bps.mean()),worst_bps=float(g.net_bps.min()),positive_cells=int((g.net_bps>0).sum()),mean_pf=float(g.pf.mean()),mean_win=float(g.win.mean()),mean_tp=float(g.tp_rate.mean()),mean_sl=float(g.sl_rate.mean())))
A=pd.DataFrame(A).sort_values(['positive_cells','mean_bps','worst_bps'],ascending=[False,False,False]);A.to_csv(OUT/'SHORT_LIFECYCLE_ECON_3M_GRID_SUMMARY.csv',index=False)
print(A.head(30).to_string(index=False))
