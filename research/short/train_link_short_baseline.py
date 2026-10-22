#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
# Reuse the proven feature contract from long research repo without copying market data.
LONG_ROOT=Path(os.getenv('SHORT_RESEARCH_LONG_ROOT','/home/kasm920/binance1-pro'))
if str(LONG_ROOT) not in sys.path: sys.path.insert(0,str(LONG_ROOT))
from features.pipeline import make_matrix, FEATURE_SCHEMA_22

SYMBOL=os.getenv('SHORT_RESEARCH_SYMBOL','LINKUSDT').upper()
INTERVALS=[x.strip() for x in os.getenv('SHORT_RESEARCH_INTERVALS','1m,3m').split(',') if x.strip()]
DATA_DIR=Path(os.getenv('SHORT_RESEARCH_DATA_DIR',str(LONG_ROOT/'data/offline_cache')))
OUT_DIR=ROOT/'research/short/results'; OUT_DIR.mkdir(parents=True,exist_ok=True)
MODELS_DIR=ROOT/'research/short/models'; MODELS_DIR.mkdir(parents=True,exist_ok=True)
H=int(os.getenv('SHORT_LABEL_HORIZON','3'))
THR=float(os.getenv('SHORT_LABEL_THR','0.0005'))
SCHEMA=[c for c in FEATURE_SCHEMA_22 if c not in ('open_time','close_time')]

def run(interval:str):
    path=DATA_DIR/f'{SYMBOL}_{interval}_6m.csv'
    df=pd.read_csv(path).reset_index(drop=True)
    X=make_matrix(df,schema=SCHEMA)
    close=pd.to_numeric(df['close'],errors='coerce').to_numpy(float)
    fut=np.roll(close,-H); ret=fut/np.maximum(close,1e-12)-1.0
    # Direct short target: meaningful down move.
    y=(ret < -THR).astype(int)
    if H>0:
        X=X[:-H]; y=y[:-H]; ret=ret[:-H]
    finite=np.isfinite(X).all(axis=1)&np.isfinite(ret)
    X=X[finite]; y=y[finite]; ret=ret[finite]
    n=len(y); split=int(n*.8)
    Xtr,Xv=X[:split],X[split:]; ytr,yv=y[:split],y[split:]; rv=ret[split:]
    cnt=np.bincount(ytr,minlength=2); w0=.5/max(cnt[0],1); w1=.5/max(cnt[1],1)
    sw=np.where(ytr==1,w1,w0)
    clf=Pipeline([('scaler',StandardScaler()),('sgd',SGDClassifier(loss='log_loss',penalty='l2',alpha=1e-4,max_iter=50,tol=1e-3,shuffle=False,random_state=42,n_jobs=-1))])
    clf.fit(Xtr,ytr,sgd__sample_weight=sw)
    score=clf.predict_proba(Xv)[:,1]
    auc=float(roc_auc_score(yv,score)) if len(np.unique(yv))>1 else .5
    ap=float(average_precision_score(yv,score))
    rows=[]
    for q in [.80,.90,.95,.975,.99]:
        t=float(np.quantile(score,q)); m=score>=t
        rows.append({'interval':interval,'quantile':q,'threshold':t,'count':int(m.sum()),'share':float(m.mean()),'short_rate':float(yv[m].mean()),'mean_ret':float(rv[m].mean()),'median_ret':float(np.median(rv[m])),'down_rate':float((rv[m]<0).mean()),'down_thr_rate':float((rv[m] <= -THR).mean())})
    model_path=MODELS_DIR/f'online_model_{SYMBOL}_{interval}_short_best.joblib'; joblib.dump(clf,model_path)
    meta={'symbol':SYMBOL,'interval':interval,'target':'future_return_below_negative_threshold','label_horizon':H,'label_thr':THR,'n_samples':int(n),'n_features':int(X.shape[1]),'feature_schema':SCHEMA,'val_auc':auc,'val_ap':ap,'train_positive_rate':float(ytr.mean()),'val_positive_rate':float(yv.mean()),'model_type':'SGDClassifier/log_loss direct p_short'}
    (MODELS_DIR/f'model_meta_{SYMBOL}_{interval}_short.json').write_text(json.dumps(meta,indent=2))
    print('SUMMARY',interval,'n',n,'auc',round(auc,6),'ap',round(ap,6),'val_pos',round(float(yv.mean()),6),'score_minmax',round(float(score.min()),6),round(float(score.max()),6),flush=True)
    for r in rows: print('TAIL',r,flush=True)
    return meta,rows

allrows=[]; metas=[]
for itv in INTERVALS:
    m,r=run(itv); metas.append(m); allrows.extend(r)
pd.DataFrame(allrows).to_csv(OUT_DIR/f'{SYMBOL}_SHORT_BASELINE_TAILS.csv',index=False)
(Path(OUT_DIR/f'{SYMBOL}_SHORT_BASELINE_SUMMARY.json')).write_text(json.dumps(metas,indent=2))
