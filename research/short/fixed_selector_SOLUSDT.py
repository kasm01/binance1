#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache/SOLUSDT_1m_360d.csv'); OUT=Path('research/short/results/SOLUSDT_SHORT_FIXED_SELECTOR_WF.csv'); SUM=Path('research/short/results/SOLUSDT_SHORT_FIXED_SELECTOR_WF_SUMMARY.csv')
TP=.012;SL=.020;H=120;FEE=.0008; BLENDS=[0,.25,.5,.75,1.]; FRACS=[.005,.01,.02,.05,.10]
d=pd.read_csv(SRC); x=d.copy();x['_g']=x.open_time.astype('int64')//300000
a=x.groupby('_g',sort=True).agg(close_time=('close_time','last'),open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'),taker_buy_base_volume=('taker_buy_base_volume','sum')).reset_index(drop=True)
o=a.open.astype(float);hi5=a.high.astype(float);lo5=a.low.astype(float);c=a.close.astype(float);v=a.volume.astype(float);tb=a.taker_buy_base_volume.astype(float);r1=c.pct_change();f=pd.DataFrame({'ts':a.close_time.astype('int64')})
f['ret1']=r1;f['ret2']=c.pct_change(2);f['ret3']=c.pct_change(3);f['ret6']=c.pct_change(6);f['ret12']=c.pct_change(12);f['body']=(c-o)/o;f['range']=(hi5-lo5)/c;rng=(hi5-lo5).replace(0,np.nan);f['close_loc']=(c-lo5)/rng;f['upper_wick']=(hi5-np.maximum(o,c))/c;f['lower_wick']=(np.minimum(o,c)-lo5)/c
for span in [5,10,20,50]:
 e=c.ewm(span=span,adjust=False).mean();f[f'e{span}_rel']=c/e-1
 if span in [10,20,50]:f[f'e{span}_slope3']=e/e.shift(3)-1
pc=c.shift(1);tr=pd.concat([hi5-lo5,(hi5-pc).abs(),(lo5-pc).abs()],axis=1).max(axis=1);f['atr14']=tr.rolling(14).mean()/c
for w in [10,20,60]:f[f'vol_z{w}']=(v-v.rolling(w).mean())/v.rolling(w).std();f[f'rv{w}']=r1.rolling(w).std()
buy=tb/v.replace(0,np.nan);f['buy_ratio']=buy;f['imbalance']=2*buy-1
for w in [3,6,12]:f[f'imb_mean{w}']=f.imbalance.rolling(w).mean();f[f'down_frac{w}']=(r1<0).astype(float).rolling(w).mean();f[f'ret_mean{w}']=r1.rolling(w).mean()
f['ret_accel']=r1-r1.shift(1);f['range_z20']=(f['range']-f['range'].rolling(20).mean())/f['range'].rolling(20).std();f=f.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
ot=d.open_time.to_numpy(np.int64);oo=d.open.to_numpy(float);hh=d.high.to_numpy(float);ll=d.low.to_numpy(float);cc=d.close.to_numpy(float);ei=np.searchsorted(ot,f.ts.to_numpy(np.int64),side='right');ok=ei+H<len(d);f=f.loc[ok].reset_index(drop=True);ei=ei[ok];feat=[z for z in f if z!='ts'];X=f[feat].to_numpy('float32');n=len(f)
# target utility
vals=[];CH=3000
for st in range(0,n,CH):
 ids=ei[st:st+CH];e=oo[ids];idx=ids[:,None]+np.arange(H)[None,:];adv=hh[idx]/e[:,None]-1;fav=1-ll[idx]/e[:,None];sh=adv>=SL;th=fav>=TP;si=np.where(sh.any(1),sh.argmax(1),H+1);ti=np.where(th.any(1),th.argmax(1),H+1);isl=(si<=ti)&(si<H);ist=(ti<si)&(ti<H);timeout=1-cc[ids+H-1]/e;gross=np.where(isl,-SL,np.where(ist,TP,timeout));vals.append(gross-FEE)
y=np.concatenate(vals);edges=np.linspace(0,n,7,dtype=int);rows=[]
def z(v,ref):
 sd=float(np.std(ref));return (v-float(np.mean(ref)))/(sd if sd>1e-9 else 1)
for fold in [3,4,5]:
 tr=np.arange(edges[0],edges[fold-1]);va=np.arange(edges[fold-1],edges[fold]);te=np.arange(edges[fold],edges[fold+1]);kw=dict(max_iter=90,max_leaf_nodes=15,l2_regularization=2.,learning_rate=.05,random_state=3000+fold)
 rg=HistGradientBoostingRegressor(**kw).fit(X[tr],y[tr]);cl=HistGradientBoostingClassifier(**kw).fit(X[tr],(y[tr]>0).astype(int));prt=rg.predict(X[tr]);ppt=cl.predict_proba(X[tr])[:,1];prv=rg.predict(X[va]);ppv=cl.predict_proba(X[va])[:,1];prtst=rg.predict(X[te]);pptst=cl.predict_proba(X[te])[:,1]
 for blend in BLENDS:
  sv=blend*z(prv,prt)+(1-blend)*z(ppv,ppt);st=blend*z(prtst,prt)+(1-blend)*z(pptst,ppt)
  for frac in FRACS:
   thr=float(np.quantile(sv,1-frac));mv=sv>=thr;mt=st>=thr;vv=y[va][mv];tt=y[te][mt]
   rows.append(dict(fold=fold,blend=blend,frac=frac,score_thr=thr,val_n=len(vv),val_bps=vv.mean()*1e4,val_win=(vv>0).mean(),test_n=len(tt),test_bps=(tt.mean()*1e4 if len(tt) else np.nan),test_win=((tt>0).mean() if len(tt) else np.nan)))
R=pd.DataFrame(rows);R.to_csv(OUT,index=False)
agg=[]
for (b,fq),g in R.groupby(['blend','frac']):
 # policy: activate next test only if validation selected edge >0 and at least 50 samples
 active=g[(g.val_bps>0)&(g.val_n>=50)&(g.test_n>0)]
 if len(active):
  w=active.test_n; net=np.average(active.test_bps,weights=w); win=np.average(active.test_win,weights=w); sel=int(w.sum())
 else:net=np.nan;win=np.nan;sel=0
 agg.append(dict(blend=b,frac=fq,active_folds=len(active),selected=sel,net_bps=net,win=win,worst_active_bps=(active.test_bps.min() if len(active) else np.nan),all_val_positive=int((g.val_bps>0).sum()),all_test_positive=int((g.test_bps>0).sum()),fold_test_bps=','.join(f'{x:.2f}' for x in g.test_bps),fold_val_bps=','.join(f'{x:.2f}' for x in g.val_bps)))
A=pd.DataFrame(agg).sort_values(['active_folds','net_bps'],ascending=[False,False]);A.to_csv(SUM,index=False);print(A.to_string(index=False));print('\nDETAIL TOP');print(R.sort_values('val_bps',ascending=False).head(30).to_string(index=False))
