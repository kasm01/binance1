#!/usr/bin/env python3
from pathlib import Path
import sys,json,hashlib
ROOT=Path('/home/kasm920/binance1'); PRO=Path('/home/kasm920/binance1-pro'); sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1,build_short_economic_features

SYMS=['AVAXUSDT','LINKUSDT','DOGEUSDT','SOLUSDT']; H=20; FAV=.006; ADV=.0075; FRACS=[.02,.05,.10,.15,.20]
OUT=ROOT/'research/short/results/SHORT2_V5_CANDIDATE_SCREEN_20261022'; OUT.mkdir(parents=True,exist_ok=True)

def model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])

def run(sym):
    d=pd.read_csv(PRO/f'data/offline_cache/{sym}_3m_360d.csv')
    for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d['ts']=pd.to_datetime(d.close_time,unit='ms',utc=True); f=build_short_economic_features(d); o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float)
    rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; idx=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]
    mfe=np.max(1-lo[idx]/ent[:,None],axis=1); mae=np.max(hi[idx]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
    X=f.iloc[rows][SHORT_ECON_MODEL_FEATURES_V1].copy(); good=np.isfinite(X.to_numpy(float)).all(axis=1); rows=rows[good]; X=X.loc[good].reset_index(drop=True); y=y[good]; ts=d.ts.iloc[rows].reset_index(drop=True)
    N=len(X); a=int(N*.70); b=int(N*.85); tr=slice(0,a); va=slice(a,b); te=slice(b,N)
    m=model().fit(X.iloc[tr],y[tr]); ptr=m.predict_proba(X.iloc[tr])[:,1]; pva=m.predict_proba(X.iloc[va])[:,1]; pte=m.predict_proba(X.iloc[te])[:,1]
    choices=[]
    for frac in FRACS:
        th=float(np.quantile(ptr,1-frac)); sv=pva>=th; st=pte>=th
        choices.append({'fraction':frac,'threshold':th,'val_selected':int(sv.sum()),'val_hit':float(y[va][sv].mean()) if sv.any() else np.nan,'val_lift':float(y[va][sv].mean()/max(y[va].mean(),1e-12)) if sv.any() else np.nan,'test_selected':int(st.sum()),'test_hit':float(y[te][st].mean()) if st.any() else np.nan,'test_lift':float(y[te][st].mean()/max(y[te].mean(),1e-12)) if st.any() else np.nan})
    C=pd.DataFrame(choices); elig=C[(C.val_selected>=300)&(C.val_hit.notna())]; best=elig.sort_values(['val_hit','val_selected'],ascending=False).iloc[0] if len(elig) else C.sort_values('val_selected',ascending=False).iloc[0]
    return {'symbol':sym,'rows':N,'positive_rate':float(y.mean()),'val_auc':float(roc_auc_score(y[va],pva)),'test_auc':float(roc_auc_score(y[te],pte)),'val_base_hit':float(y[va].mean()),'test_base_hit':float(y[te].mean()),'best_fraction':float(best.fraction),'best_threshold':float(best.threshold),'val_selected':int(best.val_selected),'val_hit':float(best.val_hit),'val_lift':float(best.val_lift),'test_selected':int(best.test_selected),'test_hit':float(best.test_hit),'test_lift':float(best.test_lift),'test_start':str(ts.iloc[b]),'test_end':str(ts.iloc[-1])},C

def main():
    rows=[]
    for sym in SYMS:
        s,c=run(sym); rows.append(s); c.to_csv(OUT/f'{sym}_fractions.csv',index=False); print('RESULT',s,flush=True)
    R=pd.DataFrame(rows).sort_values(['test_lift','test_hit','test_selected'],ascending=False); R.to_csv(OUT/'summary.csv',index=False); print('\nSUMMARY\n'+R.to_string(index=False),flush=True)
if __name__=='__main__':main()
