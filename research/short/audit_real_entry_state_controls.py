#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
EV=Path('research/short/results/SHORT_REAL_ENTRY_STATE_EVENTS_15S.csv'); DD=Path('research/short/data/real_confirm_window_1m'); OUT=Path('research/short/results')
FEE=.0008; CFGS=[('current18',.00275,.003,18),('wide60',.006,.010,60),('wide120',.006,.010,120)]
def one(d,ev,tp,sl,H):
 ts=pd.Timestamp(ev.ts); ts=ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC'); start=int((ts.floor('min')+pd.Timedelta(minutes=1)).timestamp()*1000)
 pos=d.index[d.open_time==start]
 if len(pos)!=1:return None
 w=d.iloc[int(pos[0]):int(pos[0])+H]
 if len(w)<H:return None
 e=float(ev.entry_price); hi=w.high.to_numpy(float);lo=w.low.to_numpy(float);cl=w.close.to_numpy(float)
 sh=hi/e-1>=sl; th=1-lo/e>=tp; si=int(np.argmax(sh)) if sh.any() else H+1;ti=int(np.argmax(th)) if th.any() else H+1
 if si<=ti and si<H:out='sl';gross=-sl;hb=si+1
 elif ti<si and ti<H:out='tp';gross=tp;hb=ti+1
 else:out='timeout';gross=1-cl[-1]/e;hb=H
 return dict(outcome=out,net_return_pct=gross-FEE,mfe_pct=float(np.max(1-lo/e)),mae_pct=float(np.max(hi/e-1)),hit_bar=hb)
def met(g):
 r=g.net_return_pct.to_numpy(float);p=r[r>0].sum();q=-r[r<0].sum();return dict(n=len(g),bps=float(r.mean()*1e4),win=float((r>0).mean()),pf=float(p/max(q,1e-12)),tp=float((g.outcome=='tp').mean()),sl=float((g.outcome=='sl').mean()),timeout=float((g.outcome=='timeout').mean()),mfe=float(g.mfe_pct.mean()*1e4),mae=float(g.mae_pct.mean()*1e4))
def main():
 E=pd.read_csv(EV);E['ts']=pd.to_datetime(E.ts,utc=True);ds={}
 for s in E.symbol.unique():
  d=pd.read_csv(DD/f'{s}_1m.csv');
  for c in ['open_time','open','high','low','close']:d[c]=pd.to_numeric(d[c])
  ds[s]=d.reset_index(drop=True)
 rows=[]
 for _,e in E.iterrows():
  for name,tp,sl,H in CFGS:
   z=one(ds[e.symbol],e,tp,sl,H)
   if z:rows.append(dict(ts=e.ts,symbol=e.symbol,state=e.state,reversal_pct=e.reversal_pct,config=name,**z))
 R=pd.DataFrame(rows);R.to_csv(OUT/'SHORT_REAL_ENTRY_STATE_OUTCOMES.csv',index=False)
 S=[]
 for (cfg,state),g in R.groupby(['config','state']):S.append(dict(config=cfg,state=state,symbol='ALL',**met(g)))
 for (cfg,state,sym),g in R.groupby(['config','state','symbol']):S.append(dict(config=cfg,state=state,symbol=sym,**met(g)))
 S=pd.DataFrame(S).sort_values(['config','state','symbol']);S.to_csv(OUT/'SHORT_REAL_ENTRY_STATE_SUMMARY.csv',index=False);print(S.to_string(index=False))
 # confirmation lift versus all non-confirm controls
 print('\nLIFT CONFIRM vs NONCONF')
 for cfg,g in R.groupby('config'):
  a=met(g[g.state=='CONFIRM']);b=met(g[g.state!='CONFIRM']);print(cfg,'CONF',a,'NONCONF',b,'bps_lift',a['bps']-b['bps'])
if __name__=='__main__':main()
