#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, os, sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from features.pipeline import make_matrix, FEATURE_SCHEMA_20

SYMS=[x.strip().upper() for x in os.getenv('SHORT_ECON_SYMBOLS','LINKUSDT,AVAXUSDT').split(',') if x.strip()]
SRC=Path(os.getenv('SHORT_ECON_DATA_DIR','/home/kasm920/binance1-pro/data/offline_cache'))
OUT=Path('research/short/results'); OUT.mkdir(parents=True,exist_ok=True)
FEE=.0008; LEV=2.; TP=.0055/LEV; SL=.0060/LEV; HB=6
FRACS=[.0025,.005,.01,.02,.05,.10]
MODELS=['sgd','hgb']

def barrier(d, rows):
 o=d.open.to_numpy(float); h=d.high.to_numpy(float); l=d.low.to_numpy(float); c=d.close.to_numpy(float)
 ret=[]; tag=[]
 for r in rows:
  ei=r+1; e=o[ei]; val=None; t='timeout'
  for k in range(ei,min(len(d),ei+HB)):
   # conservative tie -> stop
   if h[k]/e-1>=SL: val=-SL-FEE; t='sl'; break
   if 1-l[k]/e>=TP: val=TP-FEE; t='tp'; break
  if val is None:
   ex=c[min(len(d)-1,ei+HB-1)]; val=1-ex/e-FEE
  ret.append(val); tag.append(t)
 return np.asarray(ret),np.asarray(tag,object)

def fit(kind,X,y,seed):
 if kind=='sgd':
  cnt=np.maximum(np.bincount(y,minlength=2),1); sw=np.where(y==1,.5/cnt[1],.5/cnt[0])
  m=Pipeline([('scaler',StandardScaler()),('clf',SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,max_iter=100,tol=1e-3,shuffle=False,random_state=seed,n_jobs=-1))])
  m.fit(X,y,clf__sample_weight=sw); return m
 return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=3.,learning_rate=.05,random_state=seed).fit(X,y)

def mets(r):
 r=np.asarray(r,float); n=len(r)
 if not n:return dict(n=0,net_bps=np.nan,win=np.nan,pf=np.nan,sum_ret=0.)
 po=r[r>0].sum(); ne=-r[r<0].sum()
 return dict(n=n,net_bps=float(r.mean()*1e4),win=float((r>0).mean()),pf=float(po/max(ne,1e-12)),sum_ret=float(r.sum()))

allfold=[]; finals=[]
for sym in SYMS:
 d=pd.read_csv(SRC/f'{sym}_3m_360d.csv')
 X=make_matrix(d,FEATURE_SCHEMA_20); valid=np.isfinite(X).all(1); valid[-(HB+2):]=False
 rows=np.flatnonzero(valid); X=X[rows]
 r,t=barrier(d,rows); ok=np.isfinite(r); X=X[ok]; rows=rows[ok]; r=r[ok]; t=t[ok]
 # Direct economic target: net profitable under current fee/lifecycle proxy.
 y=(r>0).astype(int)
 n=len(y); hs=int(n*5/6); dev=np.arange(hs); hold=np.arange(hs,n); edges=np.linspace(0,hs,6,dtype=int)
 print('DATA',sym,n,'positive_rate',float(y.mean()),flush=True)
 for kind in MODELS:
  folds=[]
  for testseg in [2,3,4]:
   tr=np.arange(0,edges[testseg-1]); va=np.arange(edges[testseg-1],edges[testseg]); te=np.arange(edges[testseg],edges[testseg+1])
   m=fit(kind,X[tr],y[tr],5000+testseg); pv=m.predict_proba(X[va])[:,1]; pt=m.predict_proba(X[te])[:,1]
   auc=float(roc_auc_score(y[te],pt)) if len(np.unique(y[te]))>1 else .5
   for fq in FRACS:
    th=float(np.quantile(pv,1-fq)); z=pt>=th; mm=mets(r[te][z]); folds.append(dict(symbol=sym,model=kind,fold=testseg,frac=fq,threshold=th,auc=auc,**mm))
  F=pd.DataFrame(folds); allfold.extend(folds)
  choices=[]
  for fq,g in F.groupby('frac'):
   gg=g[g.n>0]; N=int(gg.n.sum()); pos=int((gg.net_bps>0).sum()); net=float(np.average(gg.net_bps,weights=gg.n)) if N else np.nan; worst=float(gg.net_bps.min()) if len(gg) else np.nan
   choices.append(dict(frac=fq,n=N,posfold=pos,net=net,worst=worst))
  C=pd.DataFrame(choices).sort_values(['posfold','net','n'],ascending=[False,False,False]); E=C[(C.posfold>=2)&(C.n>=30)]; ch=E.iloc[0] if len(E) else C.iloc[0]; fq=float(ch.frac)
  m=fit(kind,X[dev],y[dev],9900); pdv=m.predict_proba(X[dev])[:,1]; ph=m.predict_proba(X[hold])[:,1]; th=float(np.quantile(pdv,1-fq)); z=ph>=th; mm=mets(r[hold][z]); auc=float(roc_auc_score(y[hold],ph))
  tags=t[hold][z]
  res=dict(symbol=sym,model=kind,frac=fq,wf_posfold=int(ch.posfold),wf_n=int(ch.n),wf_net_bps=float(ch.net),wf_worst_bps=float(ch.worst),hold_auc=auc,threshold=th,hold_tp=int((tags=='tp').sum()),hold_sl=int((tags=='sl').sum()),hold_timeout=int((tags=='timeout').sum()),**{f'hold_{k}':v for k,v in mm.items()})
  finals.append(res); print('RESULT',json.dumps(res,sort_keys=True),flush=True)
pd.DataFrame(allfold).to_csv(OUT/'SHORT_DIRECT_ECON_3M_WF.csv',index=False)
pd.DataFrame(finals).to_csv(OUT/'SHORT_DIRECT_ECON_3M_HOLDOUT.csv',index=False)
(OUT/'SHORT_DIRECT_ECON_3M_HOLDOUT.json').write_text(json.dumps(finals,indent=2))
print('\nFINAL'); print(pd.DataFrame(finals).to_string(index=False))
