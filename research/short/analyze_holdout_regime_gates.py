#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
BASE=Path('research/short/data/candidate_5m_360d'); RES=Path('research/short/results')
SYMS=['IMXUSDT','ENAUSDT','ATOMUSDT']; FEE=.0008

def build(d):
 o=d.open.astype(float);h=d.high.astype(float);l=d.low.astype(float);c=d.close.astype(float);v=d.volume.astype(float);tb=d.taker_buy_base_volume.astype(float);r1=c.pct_change();f=pd.DataFrame({'ts':d.close_time.astype('int64')})
 f['ret1']=r1;f['ret2']=c.pct_change(2);f['ret3']=c.pct_change(3);f['ret6']=c.pct_change(6);f['ret12']=c.pct_change(12);f['body']=(c-o)/o;f['range']=(h-l)/c;rng=(h-l).replace(0,np.nan);f['close_loc']=(c-l)/rng;f['upper_wick']=(h-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-l)/c
 for span in [5,10,20,50]:
  e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1
  if span in [10,20,50]:f[f'e{span}_slope3']=e/e.shift(3)-1
 pc=c.shift(1);tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
 for w in [10,20,60]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r1.rolling(w).std()
 buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
 for w in [3,6,12]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean();f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r1.rolling(w).mean()
 f['ret_accel']=r1-r1.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std()
 return f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
def utility(d,rows,tp,sl,H):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);out=[]
 for st in range(0,len(rows),4000):
  ids=rows[st:st+4000];e=o[ids];idx=ids[:,None]+np.arange(H)[None,:];adv=hi[idx]/e[:,None]-1;fav=1-lo[idx]/e[:,None];sh=adv>=sl;th=fav>=tp;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1-cl[ids+H-1]/e;gross=np.where(isl,-sl,np.where(ist,tp,timeout));out.append(gross-FEE)
 return np.concatenate(out)
def z(v,ref):
 sd=float(np.std(ref));return (v-float(np.mean(ref)))/(sd if sd>1e-9 else 1.)
def stats(y,m):
 yy=y[m]
 if len(yy)==0:return (0,np.nan,np.nan,np.nan)
 pos=yy[yy>0].sum();neg=-yy[yy<0].sum();return len(yy),float(yy.mean()*1e4),float((yy>0).mean()),float(pos/max(neg,1e-12))
rows=[]
for sym in SYMS:
 meta=json.loads((RES/f'{sym}_SHORT_TRUE_HOLDOUT_5M.json').read_text());tp=float(meta['config']['tp']);sl=float(meta['config']['sl']);H=int(meta['config']['H']);b=float(meta['selector']['blend']);fq=float(meta['selector']['frac'])
 d=pd.read_csv(BASE/f'{sym}_5m_360d.csv');f=build(d);ot=d.open_time.to_numpy(np.int64);ei=np.searchsorted(ot,f.ts.to_numpy(np.int64),side='right');ok=ei+H<len(d);f=f.loc[ok].reset_index(drop=True);ei=ei[ok];feat=[c for c in f if c!='ts'];X=f[feat].to_numpy('float32');y=utility(d,ei,tp,sl,H);n=len(y);hs=int(n*5/6);dev=np.arange(hs);hold=np.arange(hs,n)
 kw=dict(max_iter=110,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=8111);rg=HistGradientBoostingRegressor(**kw).fit(X[dev],y[dev]);cl=HistGradientBoostingClassifier(**kw).fit(X[dev],(y[dev]>0).astype(int));prd=rg.predict(X[dev]);ppd=cl.predict_proba(X[dev])[:,1];prh=rg.predict(X[hold]);pph=cl.predict_proba(X[hold])[:,1];sd=b*z(prd,prd)+(1-b)*z(ppd,ppd);sh=b*z(prh,prd)+(1-b)*z(pph,ppd);thr=float(np.quantile(sd,1-fq));selected=sh>=thr
 # causal regime fields on raw bar indexed by entry index-1/completed signal bar
 c=d.close.astype(float); e50=c.ewm(span=50,adjust=False).mean();e144=c.ewm(span=144,adjust=False).mean();e288=c.ewm(span=288,adjust=False).mean();e864=c.ewm(span=864,adjust=False).mean();sigidx=np.maximum(ei[hold]-1,0)
 gates={
 'all':np.ones(len(hold),bool),
 'c<ema288':(c.to_numpy()[sigidx] < e288.to_numpy()[sigidx]),
 'ema50<ema288':(e50.to_numpy()[sigidx] < e288.to_numpy()[sigidx]),
 'ema144<ema288':(e144.to_numpy()[sigidx] < e288.to_numpy()[sigidx]),
 'mom12h<0':(c.pct_change(144).to_numpy()[sigidx] < 0),
 'mom24h<0':(c.pct_change(288).to_numpy()[sigidx] < 0),
 'mom3d<0':(c.pct_change(864).to_numpy()[sigidx] < 0),
 'bear24':(c.to_numpy()[sigidx] < e288.to_numpy()[sigidx]) & (e50.to_numpy()[sigidx] < e288.to_numpy()[sigidx]) & (c.pct_change(288).to_numpy()[sigidx] < 0),
 'bear3d':(c.to_numpy()[sigidx] < e864.to_numpy()[sigidx]) & (e144.to_numpy()[sigidx] < e864.to_numpy()[sigidx]) & (c.pct_change(864).to_numpy()[sigidx] < 0),
 }
 yh=y[hold]
 for name,g in gates.items():
  m=selected & np.nan_to_num(g,nan=False); N,bps,win,pf=stats(yh,m);rows.append(dict(symbol=sym,gate=name,selected=N,net_bps=bps,win=win,pf=pf,coverage=N/max(int(selected.sum()),1)))
R=pd.DataFrame(rows);R.to_csv(RES/'SHORT_HOLDOUT_REGIME_GATES.csv',index=False);print(R.to_string(index=False))
