#!/usr/bin/env python3
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SYMS=['LINKUSDT','AVAXUSDT']; SRC=Path('/home/kasm920/binance1-pro/data/offline_cache'); OUT=Path('research/short/results');OUT.mkdir(parents=True,exist_ok=True)
FEE=.0008
CONFIGS=[('current',.00275,.003,6),('wide',.006,.010,40),('runner',.012,.010,40)]
FRACS=[.0025,.005,.01,.02,.05,.10]

def feats(d):
 o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r=c.pct_change()
 f=pd.DataFrame(index=d.index)
 for n in [1,2,3,6,12,20,60,240]: f[f'ret{n}']=c.pct_change(n)
 f['body']=(c-o)/o; f['range']=(h-l)/c; rng=(h-l).replace(0,np.nan); f['close_loc']=(c-l)/rng
 f['upper_wick']=(h-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-l)/c
 for span in [5,10,20,50,100]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1;f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60,240]:
  f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r.rolling(w).std();f[f'down_frac{w}']=(r<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r.rolling(w).mean()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12,30]: f[f'imb_mean{w}']=f.imbalance.rolling(w).mean()
 f['ret_accel']=r-r.shift(1); f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std()
 f['hour_sin']=np.sin(2*np.pi*(pd.to_datetime(d.open_time,unit='ms',utc=True).dt.hour)/24);f['hour_cos']=np.cos(2*np.pi*(pd.to_datetime(d.open_time,unit='ms',utc=True).dt.hour)/24)
 return f.replace([np.inf,-np.inf],np.nan)

def barrier(d, rows,tp,sl,H):
 from numpy.lib.stride_tricks import sliding_window_view
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float)
 # vectorized row-specific windows via index matrix; rows chosen safely
 e=o[rows+1]; idx=(rows+1)[:,None]+np.arange(H)[None,:]
 hw=hi[idx];lw=lo[idx]; sh=(hw/e[:,None]-1)>=sl; th=(1-lw/e[:,None])>=tp
 si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);itp=(ti<si)&(ti<H)
 tout=1-cl[rows+H]/e-FEE
 return np.where(isl,-sl-FEE,np.where(itp,tp-FEE,tout))

def mets(r):
 r=np.asarray(r); n=len(r)
 if not n:return dict(n=0,bps=np.nan,win=np.nan,pf=np.nan,sumret=0.)
 pos=r[r>0].sum();neg=-r[r<0].sum();return dict(n=n,bps=float(r.mean()*1e4),win=float((r>0).mean()),pf=float(pos/max(neg,1e-12)),sumret=float(r.sum()))

foldout=[]; finals=[]
for sym in SYMS:
 d=pd.read_csv(SRC/f'{sym}_3m_360d.csv'); F=feats(d); cols=list(F.columns); X=F.to_numpy(float)
 valid=np.isfinite(X).all(1); valid[-45:]=False; rows=np.flatnonzero(valid); X=X[rows]
 n=len(rows); hs=int(n*5/6); dev=np.arange(hs); hold=np.arange(hs,n); edges=np.linspace(0,hs,6,dtype=int)
 print('DATA',sym,n,'features',len(cols),flush=True)
 for name,tp,sl,H in CONFIGS:
  r=barrier(d,rows,tp,sl,H)
  folds=[]
  for seg in [2,3,4]:
   tr=np.arange(edges[0],edges[seg-1]); va=np.arange(edges[seg-1],edges[seg]); te=np.arange(edges[seg],edges[seg+1])
   m=HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=15,l2_regularization=4.,learning_rate=.05,random_state=7000+seg).fit(X[tr],r[tr])
   pv=m.predict(X[va]);pt=m.predict(X[te])
   for fq in FRACS:
    th=float(np.quantile(pv,1-fq));z=pt>=th; mm=mets(r[te][z]); folds.append(dict(symbol=sym,config=name,fold=seg,frac=fq,threshold=th,**mm))
  FF=pd.DataFrame(folds);foldout.extend(folds)
  choices=[]
  for fq,g in FF.groupby('frac'):
   gg=g[g.n>0];N=int(gg.n.sum());pos=int((gg.bps>0).sum());net=float(np.average(gg.bps,weights=gg.n)) if N else np.nan;worst=float(gg.bps.min()) if len(gg) else np.nan
   choices.append(dict(frac=fq,n=N,posfold=pos,net=net,worst=worst))
  C=pd.DataFrame(choices).sort_values(['posfold','net','n'],ascending=[False,False,False]);E=C[(C.posfold>=2)&(C.n>=30)];ch=E.iloc[0] if len(E) else C.iloc[0];fq=float(ch.frac)
  m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=4.,learning_rate=.05,random_state=9999).fit(X[dev],r[dev]);pdv=m.predict(X[dev]);ph=m.predict(X[hold]);th=float(np.quantile(pdv,1-fq));z=ph>=th;mm=mets(r[hold][z])
  res=dict(symbol=sym,config=name,tp=tp,sl=sl,H=H,features=len(cols),frac=fq,wf_posfold=int(ch.posfold),wf_n=int(ch.n),wf_bps=float(ch.net),wf_worst=float(ch.worst),threshold=th,**{f'hold_{k}':v for k,v in mm.items()})
  finals.append(res); print('RESULT',json.dumps(res,sort_keys=True),flush=True)
pd.DataFrame(foldout).to_csv(OUT/'SHORT_RICH_MODEL_3M_WF.csv',index=False);pd.DataFrame(finals).to_csv(OUT/'SHORT_RICH_MODEL_3M_HOLDOUT.csv',index=False);(OUT/'SHORT_RICH_MODEL_3M_HOLDOUT.json').write_text(json.dumps(finals,indent=2))
print('\nFINAL');print(pd.DataFrame(finals).to_string(index=False))
