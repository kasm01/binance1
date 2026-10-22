from pathlib import Path
import json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features

DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
H=20; FAV=.006; ADV=.0075; TEST_DAYS=30; INIT_TRAIN_DAYS=90
FRACS=(.05,.10,.15,.20); BAD_CTX_UPLIFT=.03
FEE=.0008; LEV=3.0; TP_ARM=.04; GIVEBACK=.0024; SL_PRICE=.0299; CATA_PRICE=.035
MAX_LAYERS=3; COOLDOWN_BARS=1; DD_REDUCE=.08; DD_RECOVER=.04
NORMAL=(1.0,.50,2.0); REDUCED=(.50,.10,.70); START_EQ=1000.0

def make_model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])

d=pd.read_csv(DATA)
for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True); f=build_short_economic_features(d)
o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float)
rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; fut=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]
mfe=np.max(1-lo[fut]/ent[:,None],axis=1); mae=np.max(hi[fut]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
X=f.iloc[rows].copy(); good=np.isfinite(X[SHORT_ECON_MODEL_FEATURES_V1].to_numpy(float)).all(axis=1)
rows=rows[good]; X=X.loc[good].reset_index(drop=True); y=y[good]; ts=d.ts.iloc[rows].reset_index(drop=True)
score=np.full(len(d),np.nan); base_thr=np.full(len(d),np.nan); atr90=np.full(len(d),np.nan); fold_id=np.full(len(d),-1,dtype=int); wf_frac=np.full(len(d),np.nan)
start_time=ts.iloc[0]+pd.Timedelta(days=INIT_TRAIN_DAYS); end_time=ts.iloc[-1]; fold=0; fold_rows=[]
while start_time < end_time:
    test_end=min(start_time+pd.Timedelta(days=TEST_DAYS),end_time+pd.Timedelta(minutes=1))
    test_mask=(ts>=start_time)&(ts<test_end); test_idx=np.flatnonzero(test_mask.to_numpy())
    if not len(test_idx): start_time=test_end; continue
    safe_train_end=start_time-pd.Timedelta(minutes=66); train_idx=np.flatnonzero((ts<safe_train_end).to_numpy())
    if len(train_idx)<10000: start_time=test_end; continue
    split=max(1000,int(len(train_idx)*.80)); inner=train_idx[:split]; val=train_idx[split:]
    m0=make_model().fit(X.iloc[inner][SHORT_ECON_MODEL_FEATURES_V1],y[inner]); ptr=m0.predict_proba(X.iloc[inner][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pval=m0.predict_proba(X.iloc[val][SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    choices=[]
    for frac in FRACS:
        th=float(np.quantile(ptr,1-frac)); sel=pval>=th; n=int(sel.sum()); hit=float(y[val][sel].mean()) if n else -1.0; choices.append((hit,n,frac))
    eligible=[z for z in choices if z[1]>=500]; _,_,frac=max(eligible,key=lambda z:(z[0],z[1])) if eligible else max(choices,key=lambda z:z[1])
    m=make_model().fit(X.iloc[train_idx][SHORT_ECON_MODEL_FEATURES_V1],y[train_idx]); ptrain=m.predict_proba(X.iloc[train_idx][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; th=float(np.quantile(ptrain,1-frac)); ptest=m.predict_proba(X.iloc[test_idx][SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    orig=rows[test_idx]; score[orig]=ptest; base_thr[orig]=th; fold_id[orig]=fold; wf_frac[orig]=frac; a90=float(X.iloc[train_idx].atr14_pct.quantile(.90)); atr90[orig]=a90
    fold_rows.append({'fold':fold,'test_start':str(start_time),'test_end':str(test_end),'train_n':len(train_idx),'test_n':len(test_idx),'frac':frac,'threshold':th,'atr90':a90,'test_base_rate':float(y[test_idx].mean()),'test_selected_n':int((ptest>=th).sum()),'test_selected_hit':float(y[test_idx][ptest>=th].mean()) if (ptest>=th).any() else None})
    fold+=1; start_time=test_end

d['wf_score']=score; d['wf_threshold']=base_thr; d['wf_atr90']=atr90; d['wf_fold']=fold_id; d['wf_frac']=wf_frac
for c in ['ret_240','atr14_pct']: d[c]=f[c]
first_oos=d.loc[np.isfinite(d.wf_score),'ts'].min(); last_oos=d.loc[np.isfinite(d.wf_score),'ts'].max()


out=Path('research/short/results/SHORT1_INJ_WF_SCORES_V4.csv')
cols=['ts','open','high','low','close','wf_score','wf_threshold','wf_atr90','wf_fold','wf_frac','ret_240','atr14_pct']
d.loc[np.isfinite(d.wf_score),cols].to_csv(out,index=False)
print('WROTE',out,'ROWS',int(np.isfinite(d.wf_score).sum()),'OOS',first_oos,last_oos,'FOLDS',len(fold_rows))
print('FOLDS_JSON',json.dumps(fold_rows,default=float))
