#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
DATA=Path('research/short/data/candidate_5m_360d'); OUT=Path('research/short/results/SHORT_CANDIDATE_5M_SCREEN.csv')
SYMS=[x.strip().upper() for x in os.getenv('SHORT_SCREEN_SYMBOLS','INJUSDT,SEIUSDT,ARBUSDT,OPUSDT,APTUSDT,SUIUSDT').split(',') if x.strip()]
TP=.012;SL=.020;H=24;FEE=.0008;FRACS=[.005,.01,.02]

def build(d):
 o=d.open.astype(float);hi=d.high.astype(float);lo=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r1=c.pct_change();f=pd.DataFrame({'ts':d.close_time.astype('int64')})
 f['ret1']=r1;f['ret2']=c.pct_change(2);f['ret3']=c.pct_change(3);f['ret6']=c.pct_change(6);f['ret12']=c.pct_change(12);f['body']=(c-o)/o;f['range']=(hi-lo)/c;rng=(hi-lo).replace(0,np.nan);f['close_loc']=(c-lo)/rng;f['upper_wick']=(hi-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-lo)/c
 for span in [5,10,20,50]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1
  if span in [10,20,50]:f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([hi-lo,(hi-pc).abs(),(lo-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r1.rolling(w).std()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean();f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r1.rolling(w).mean()
 f['ret_accel']=r1-r1.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std();return f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
def utility(d, rows):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);e=o[rows];idx=rows[:,None]+np.arange(H)[None,:];adv=hi[idx]/e[:,None]-1.;fav=1.-lo[idx]/e[:,None];sh=adv>=SL;th=fav>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1.-cl[rows+H-1]/e;gross=np.where(isl,-SL,np.where(ist,TP,timeout));return gross-FEE
rows=[]
for sym in SYMS:
 p=DATA/f'{sym}_5m_360d.csv'
 if not p.exists(): print('MISSING',sym,flush=True);continue
 d=pd.read_csv(p);f=build(d);ot=d.open_time.to_numpy(np.int64);ei=np.searchsorted(ot,f.ts.to_numpy(np.int64),side='right');ok=ei+H<len(d);f=f.loc[ok].reset_index(drop=True);ei=ei[ok];feat=[c for c in f if c!='ts'];X=f[feat].to_numpy('float32');y=utility(d,ei);n=len(y);edges=np.linspace(0,n,7,dtype=int)
 print('SYMBOL',sym,'n',n,flush=True)
 fold_cache=[]
 for testseg in [3,4,5]:
  tr=np.arange(0,edges[testseg-1]);va=np.arange(edges[testseg-1],edges[testseg]);te=np.arange(edges[testseg],edges[testseg+1])
  rg=HistGradientBoostingRegressor(max_iter=90,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=5000+testseg).fit(X[tr],y[tr])
  ptr=rg.predict(X[tr]);pva=rg.predict(X[va]);pte=rg.predict(X[te]);mu=float(ptr.mean());sd=float(ptr.std()) or 1.;sv=(pva-mu)/sd;st=(pte-mu)/sd
  fold_cache.append((testseg,sv,st,va,te))
 for frac in FRACS:
  fold=[]
  for testseg,sv,st,va,te in fold_cache:
   thr=float(np.quantile(sv,1-frac));mv=sv>=thr;mt=st>=thr;vv=y[va][mv];tt=y[te][mt];active=len(vv)>=50 and vv.mean()>0
   fold.append(dict(fold=testseg,val_n=len(vv),val_bps=float(vv.mean()*1e4),active=active,test_n=int(mt.sum()),test_bps=float(tt.mean()*1e4) if len(tt) else np.nan,test_win=float((tt>0).mean()) if len(tt) else np.nan))
  g=pd.DataFrame(fold);a=g[g.active & g.test_n.gt(0)];selected=int(a.test_n.sum());net=float(np.average(a.test_bps,weights=a.test_n)) if selected else np.nan;win=float(np.average(a.test_win,weights=a.test_n)) if selected else np.nan;worst=float(a.test_bps.min()) if selected else np.nan
  rows.append(dict(symbol=sym,frac=frac,folds=3,active_folds=len(a),selected=selected,net_bps=net,win=win,worst_active_bps=worst,val_positive_folds=int((g.val_bps>0).sum()),test_positive_folds=int((g.test_bps>0).sum()),val_bps='|'.join(f'{x:.2f}' for x in g.val_bps),test_bps='|'.join(f'{x:.2f}' for x in g.test_bps)))
 # checkpoint after each symbol so interrupted runs keep completed work
 pd.DataFrame(rows).sort_values(['active_folds','net_bps'],ascending=[False,False]).to_csv(OUT,index=False)
R=pd.DataFrame(rows).sort_values(['active_folds','net_bps'],ascending=[False,False]);R.to_csv(OUT,index=False);print('\nRESULTS');print(R.to_string(index=False));print('OUT',OUT)
