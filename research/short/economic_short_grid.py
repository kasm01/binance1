#!/usr/bin/env python3
from pathlib import Path
import os, sys, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
LONG_ROOT=Path('/home/kasm920/binance1-pro')
if str(LONG_ROOT) not in sys.path: sys.path.insert(0,str(LONG_ROOT))
from features.pipeline import make_matrix, FEATURE_SCHEMA_22
SYMBOL='LINKUSDT'; DATA=LONG_ROOT/'data/offline_cache'; OUT=ROOT/'research/short/results'
SCHEMA=[c for c in FEATURE_SCHEMA_22 if c not in ('open_time','close_time')]
FEE=0.0008
CONFIGS={
 '1m': {'horizons':[10,20,30,60], 'thresholds':[0.001,0.002,0.003,0.005]},
 '3m': {'horizons':[3,5,10,20], 'thresholds':[0.001,0.002,0.003,0.005]},
}
rows=[]
for itv,cfg in CONFIGS.items():
    df=pd.read_csv(DATA/f'{SYMBOL}_{itv}_6m.csv').reset_index(drop=True)
    Xfull=make_matrix(df,schema=SCHEMA)
    close=pd.to_numeric(df['close'],errors='coerce').to_numpy(float)
    for h in cfg['horizons']:
      for labthr in cfg['thresholds']:
        fut=np.roll(close,-h); ret=fut/np.maximum(close,1e-12)-1
        y=(ret<=-labthr).astype(int)
        X=Xfull[:-h]; yy=y[:-h]; rr=ret[:-h]
        finite=np.isfinite(X).all(axis=1)&np.isfinite(rr)
        X=X[finite]; yy=yy[finite]; rr=rr[finite]
        n=len(yy); i60=int(n*.6); i80=int(n*.8)
        Xtr,Xv,Xt=X[:i60],X[i60:i80],X[i80:]
        ytr,yv,yt=yy[:i60],yy[i60:i80],yy[i80:]
        rv,rt=rr[i60:i80],rr[i80:]
        cnt=np.bincount(ytr,minlength=2); sw=np.where(ytr==1,.5/max(cnt[1],1),.5/max(cnt[0],1))
        clf=Pipeline([('scaler',StandardScaler()),('sgd',SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,max_iter=80,tol=1e-3,shuffle=False,random_state=42,n_jobs=-1))])
        clf.fit(Xtr,ytr,sgd__sample_weight=sw)
        sv=clf.predict_proba(Xv)[:,1]; st=clf.predict_proba(Xt)[:,1]
        aucv=roc_auc_score(yv,sv) if len(np.unique(yv))>1 else .5
        auct=roc_auc_score(yt,st) if len(np.unique(yt))>1 else .5
        apv=average_precision_score(yv,sv); apt=average_precision_score(yt,st)
        for q in [.80,.90,.95,.975,.99]:
            score_thr=float(np.quantile(sv,q))
            mv=sv>=score_thr; mt=st>=score_thr
            def metrics(mask,yv0,r0):
                if mask.sum()==0: return dict(count=0,short_rate=np.nan,mean_ret=np.nan,median_ret=np.nan,down_rate=np.nan,edge=np.nan)
                a=r0[mask]
                return dict(count=int(mask.sum()),short_rate=float(yv0[mask].mean()),mean_ret=float(a.mean()),median_ret=float(np.median(a)),down_rate=float((a<0).mean()),edge=float(-a.mean()-FEE))
            vm=metrics(mv,yv,rv); tm=metrics(mt,yt,rt)
            rows.append({
              'interval':itv,'horizon_bars':h,'horizon_min':h*(1 if itv=='1m' else 3),'label_thr':labthr,
              'val_auc':aucv,'test_auc':auct,'val_ap':apv,'test_ap':apt,'score_quantile':q,'score_thr':score_thr,
              **{f'val_{k}':v for k,v in vm.items()}, **{f'test_{k}':v for k,v in tm.items()},
              'train_pos_rate':float(ytr.mean()),'val_pos_rate':float(yv.mean()),'test_pos_rate':float(yt.mean())
            })
R=pd.DataFrame(rows)
R.to_csv(OUT/'LINKUSDT_ECONOMIC_SHORT_GRID.csv',index=False)
# rank only by validation economics, never by test
cand=R[(R.val_count>=100)&R.val_edge.notna()].copy()
cand['val_objective']=cand['val_edge'] * np.sqrt(cand['val_count'])
rank=cand.sort_values(['val_objective','val_auc'],ascending=False)
rank.head(20).to_csv(OUT/'LINKUSDT_ECONOMIC_SHORT_TOP20.csv',index=False)
print('TOP_VALIDATION_ONLY')
cols=['interval','horizon_min','label_thr','score_quantile','score_thr','val_count','val_mean_ret','val_edge','val_short_rate','val_auc','test_count','test_mean_ret','test_edge','test_short_rate','test_auc']
print(rank[cols].head(20).to_string(index=False))
