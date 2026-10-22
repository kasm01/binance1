#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

SRC=Path('research/short/data/candidate_5m_360d/INJUSDT_5m_360d.csv')
OUTDIR=Path('research/short/results'); OUTDIR.mkdir(parents=True,exist_ok=True)
FEE=.0008
CONFIGS=[(.006,.010,24),(.006,.015,24),(.006,.020,24),(.008,.015,24),(.008,.020,24),(.012,.020,24)]
BLENDS=[0.,.25,.5,.75,1.]
FRACS=[.005,.01,.02,.05]

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

def utility(d, rows, tp, sl, H):
 o=d.open.to_numpy(float);hi=d.high.to_numpy(float);lo=d.low.to_numpy(float);cl=d.close.to_numpy(float);out=[];CH=4000
 for st in range(0,len(rows),CH):
  ids=rows[st:st+CH];e=o[ids];idx=ids[:,None]+np.arange(H)[None,:];adv=hi[idx]/e[:,None]-1;fav=1-lo[idx]/e[:,None];sh=adv>=sl;th=fav>=tp;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1-cl[ids+H-1]/e;gross=np.where(isl,-sl,np.where(ist,tp,timeout));out.append(gross-FEE)
 return np.concatenate(out)

def z(v,ref):
 sd=float(np.std(ref));return (v-float(np.mean(ref)))/(sd if sd>1e-9 else 1.)

d=pd.read_csv(SRC);f=build(d);ot=d.open_time.to_numpy(np.int64);ei=np.searchsorted(ot,f.ts.to_numpy(np.int64),side='right');maxH=max(x[2] for x in CONFIGS);ok=ei+maxH<len(d);f=f.loc[ok].reset_index(drop=True);ei=ei[ok];feat=[c for c in f if c!='ts'];X=f[feat].to_numpy('float32');n=len(f);utils={cfg:utility(d,ei,*cfg) for cfg in CONFIGS}
print('DATA',n,'features',len(feat),flush=True)

# Nested WF on first 300d-ish (first 5/6 of data), reserve last 1/6 as untouched holdout.
hold_start=int(n*5/6); dev=np.arange(0,hold_start); hold=np.arange(hold_start,n); edges=np.linspace(0,len(dev),6,dtype=int)
rows=[]
for cfg in CONFIGS:
 y=utils[cfg]
 for testseg in [2,3,4]:
  tr=np.arange(0,edges[testseg-1]);va=np.arange(edges[testseg-1],edges[testseg]);te=np.arange(edges[testseg],edges[testseg+1]);kw=dict(max_iter=90,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=7100+testseg)
  rg=HistGradientBoostingRegressor(**kw).fit(X[tr],y[tr]);cl=HistGradientBoostingClassifier(**kw).fit(X[tr],(y[tr]>0).astype(int));prt=rg.predict(X[tr]);ppt=cl.predict_proba(X[tr])[:,1];prv=rg.predict(X[va]);ppv=cl.predict_proba(X[va])[:,1];pre=rg.predict(X[te]);ppe=cl.predict_proba(X[te])[:,1]
  best=None
  for b in BLENDS:
   sv=b*z(prv,prt)+(1-b)*z(ppv,ppt)
   for fq in FRACS:
    thr=float(np.quantile(sv,1-fq));m=sv>=thr;vv=y[va][m]
    if len(vv)<40:continue
    score=float(vv.mean()*np.sqrt(len(vv)))
    if best is None or score>best[0]:best=(score,b,fq,thr,len(vv),float(vv.mean()))
  if best is None:continue
  _,b,fq,thr,vn,vmean=best;st=b*z(pre,prt)+(1-b)*z(ppe,ppt);m=st>=thr;tt=y[te][m];pos=tt[tt>0].sum();neg=-tt[tt<0].sum()
  rows.append(dict(tp=cfg[0],sl=cfg[1],H=cfg[2],fold=testseg,blend=b,frac=fq,val_n=vn,val_bps=vmean*1e4,test_n=len(tt),test_bps=(tt.mean()*1e4 if len(tt) else np.nan),test_win=((tt>0).mean() if len(tt) else np.nan),test_pf=(pos/max(neg,1e-12) if len(tt) else np.nan)))
R=pd.DataFrame(rows);R.to_csv(OUTDIR/'INJUSDT_SHORT_NESTED_5M.csv',index=False)
A=[]
for cfg,g in R.groupby(['tp','sl','H']):
 gg=g[g.test_n>0];sel=int(gg.test_n.sum());A.append(dict(tp=cfg[0],sl=cfg[1],H=cfg[2],folds=len(g),selected=sel,net_bps=(np.average(gg.test_bps,weights=gg.test_n) if sel else np.nan),positive_folds=int((g.test_bps>0).sum()),worst_bps=float(g.test_bps.min()),win=(np.average(gg.test_win,weights=gg.test_n) if sel else np.nan),pf=float(g.test_pf.mean())))
A=pd.DataFrame(A).sort_values(['positive_folds','net_bps'],ascending=[False,False]);A.to_csv(OUTDIR/'INJUSDT_SHORT_NESTED_5M_SUMMARY.csv',index=False);print('\nNESTED');print(A.to_string(index=False),flush=True)

# Choose lifecycle by nested robustness: most positive folds, then net, require >=2 positive folds.
elig=A[A.positive_folds>=2]
if len(elig)==0:
 print('NO_ROBUST_CONFIG'); raise SystemExit(2)
bestcfg=elig.iloc[0];cfg=(float(bestcfg.tp),float(bestcfg.sl),int(bestcfg.H));y=utils[cfg]
# Within dev: train first 80%, validation last 20% to choose selector, then refit on full dev and apply locked selector settings to holdout.
cut=int(len(dev)*.8);tr=dev[:cut];va=dev[cut:];kw=dict(max_iter=110,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=8111)
rg=HistGradientBoostingRegressor(**kw).fit(X[tr],y[tr]);cl=HistGradientBoostingClassifier(**kw).fit(X[tr],(y[tr]>0).astype(int));prt=rg.predict(X[tr]);ppt=cl.predict_proba(X[tr])[:,1];prv=rg.predict(X[va]);ppv=cl.predict_proba(X[va])[:,1]
best=None
for b in BLENDS:
 sv=b*z(prv,prt)+(1-b)*z(ppv,ppt)
 for fq in FRACS:
  thr=float(np.quantile(sv,1-fq));m=sv>=thr;vv=y[va][m]
  if len(vv)<50:continue
  score=float(vv.mean()*np.sqrt(len(vv))); cand=(score,b,fq,float(vv.mean()),len(vv))
  if best is None or cand[0]>best[0]:best=cand
if best is None: print('NO_SELECTOR'); raise SystemExit(3)
_,b,fq,vmean,vn=best
# refit dev; threshold uses dev score quantile corresponding to locked frac
rg=HistGradientBoostingRegressor(**kw).fit(X[dev],y[dev]);cl=HistGradientBoostingClassifier(**kw).fit(X[dev],(y[dev]>0).astype(int));prd=rg.predict(X[dev]);ppd=cl.predict_proba(X[dev])[:,1];prh=rg.predict(X[hold]);pph=cl.predict_proba(X[hold])[:,1];sd=b*z(prd,prd)+(1-b)*z(ppd,ppd);sh=b*z(prh,prd)+(1-b)*z(pph,ppd);thr=float(np.quantile(sd,1-fq));m=sh>=thr;hh=y[hold][m];pos=hh[hh>0].sum();neg=-hh[hh<0].sum()
summary=dict(config={'tp':cfg[0],'sl':cfg[1],'H':cfg[2]},selector={'blend':b,'frac':fq,'validation_n':vn,'validation_bps':vmean*1e4,'threshold':thr},holdout={'rows':len(hold),'selected':int(len(hh)),'net_bps':float(hh.mean()*1e4) if len(hh) else None,'win':float((hh>0).mean()) if len(hh) else None,'pf':float(pos/max(neg,1e-12)) if len(hh) else None,'sum_return_unscaled':float(hh.sum()) if len(hh) else None},feature_count=len(feat))
(OUTDIR/'INJUSDT_SHORT_TRUE_HOLDOUT_5M.json').write_text(json.dumps(summary,indent=2))
print('\nHOLDOUT');print(json.dumps(summary,indent=2),flush=True)
