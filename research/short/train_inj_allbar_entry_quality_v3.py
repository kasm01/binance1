from pathlib import Path
import json, hashlib, numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, SHORT_ECON_MODEL_FEATURE_HASH_V1, build_short_economic_features

ROOT=Path('/home/kasm920/binance1'); DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv')
MODEL_DIR=ROOT/'research/short/models'; RES=ROOT/'research/short/results'
H=20; FAV=.006; ADV=.0075

def model():
    return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])

d=pd.read_csv(DATA)
for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['ts']=pd.to_datetime(d.close_time,unit='ms',utc=True)
f=build_short_economic_features(d)
o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float)
rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; idx=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]
mfe=np.max(1-lo[idx]/ent[:,None],axis=1); mae=np.max(hi[idx]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
X=f.iloc[rows][SHORT_ECON_MODEL_FEATURES_V1].copy(); good=np.isfinite(X.to_numpy(float)).all(axis=1); rows=rows[good]; X=X.loc[good].reset_index(drop=True); y=y[good]; ts=d.ts.iloc[rows].reset_index(drop=True)
N=len(X); a=int(N*.70); b=int(N*.85)
tr=slice(0,a); va=slice(a,b); te=slice(b,N)
m=model().fit(X.iloc[tr],y[tr]); pva=m.predict_proba(X.iloc[va])[:,1]; pte=m.predict_proba(X.iloc[te])[:,1]
fracs=[.02,.05,.10,.15,.20]; choices=[]
ptr=m.predict_proba(X.iloc[tr])[:,1]
for frac in fracs:
    thr=float(np.quantile(ptr,1-frac)); sel=pva>=thr; hit=float(y[va][sel].mean()) if sel.any() else None; choices.append((frac,thr,int(sel.sum()),hit))
elig=[x for x in choices if x[2]>=500 and x[3] is not None]; best=max(elig,key=lambda x:(x[3],x[2])) if elig else max(choices,key=lambda x:x[2])
frac,thr,_,_=best; sel=pte>=thr
report={'rows':N,'positive_rate':float(y.mean()),'train_rows':a,'val_rows':b-a,'test_rows':N-b,'val_auc':float(roc_auc_score(y[va],pva)),'test_auc':float(roc_auc_score(y[te],pte)),'fraction':frac,'threshold':thr,'val_choices':[{'frac':x[0],'thr':x[1],'n':x[2],'hit':x[3]} for x in choices],'test_selected':int(sel.sum()),'test_selected_rate':float(sel.mean()),'test_hit_rate':float(y[te][sel].mean()) if sel.any() else None,'test_base_hit_rate':float(y[te].mean()),'test_start':str(ts.iloc[b]),'test_end':str(ts.iloc[-1])}
final=model().fit(X,y); ps=final.predict_proba(X)[:,1]; ref=float(np.quantile(ps,1-frac))
meta={'bot_id':'SHORT1','symbol':'INJUSDT','side':'short','interval':'3m','provider':'short_entry_quality_allbar_logistic_v3','model_type':'StandardScaler+LogisticRegression','score_semantics':'short_economic_probability','target':'short_entry_quality_allbar_60m','observation_min':60,'target_favorable_mfe_pct':FAV,'target_max_adverse_mae_pct':ADV,'promoted':False,'status':'research_only','features':SHORT_ECON_MODEL_FEATURES_V1,'feature_count':len(SHORT_ECON_MODEL_FEATURES_V1),'feature_hash':SHORT_ECON_MODEL_FEATURE_HASH_V1,'train_rows':N,'positive_rate':float(y.mean()),'locked_top_fraction':frac,'reference_probability_threshold_full_train':ref,'validation_report':report}
MODEL_DIR.mkdir(parents=True,exist_ok=True); RES.mkdir(parents=True,exist_ok=True)
joblib.dump(final,MODEL_DIR/'short_model_INJUSDT_3m_allbar_v3.joblib'); (MODEL_DIR/'short_model_meta_INJUSDT_3m_allbar_v3.json').write_text(json.dumps(meta,indent=2,allow_nan=False)); (RES/'SHORT1_INJ_ALLBAR_V3_REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(report,indent=2)); print('FULL_REF_THRESHOLD',ref)
