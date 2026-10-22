from pathlib import Path
import itertools, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, build_short_economic_features
DATA=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv'); H=20; FAV=.006; ADV=.0075

def mdl(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])
d=pd.read_csv(DATA)
for c in ['open','high','low','close','open_time','close_time']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['ts']=pd.to_datetime(d.close_time,unit='ms',utc=True); f=build_short_economic_features(d)
o=d.open.to_numpy(float); hi=d.high.to_numpy(float); lo=d.low.to_numpy(float)
rows=np.arange(0,len(d)-H-1,dtype=int); ei=rows+1; idx=ei[:,None]+np.arange(H)[None,:]; ent=o[ei]
mfe=np.max(1-lo[idx]/ent[:,None],axis=1); mae=np.max(hi[idx]/ent[:,None]-1,axis=1); y=((mfe>=FAV)&(mae<=ADV)).astype(int)
X=f.iloc[rows].copy(); good=np.isfinite(X[SHORT_ECON_MODEL_FEATURES_V1].to_numpy(float)).all(axis=1); rows=rows[good]; X=X.loc[good].reset_index(drop=True); y=y[good]; ts=d.ts.iloc[rows].reset_index(drop=True)
N=len(X); a=int(N*.70); b=int(N*.85); tr=np.arange(a); va=np.arange(a,b); te=np.arange(b,N)
m=mdl().fit(X.iloc[tr][SHORT_ECON_MODEL_FEATURES_V1],y[tr]); ptr=m.predict_proba(X.iloc[tr][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pva=m.predict_proba(X.iloc[va][SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pte=m.predict_proba(X.iloc[te][SHORT_ECON_MODEL_FEATURES_V1])[:,1]
# locked model threshold from validation selection
fracs=[.02,.05,.10,.15,.20]; choices=[]
for frac in fracs:
    th=float(np.quantile(ptr,1-frac)); mask=pva>=th; choices.append((float(y[va][mask].mean()),int(mask.sum()),frac,th))
_,_,frac,thr=max([z for z in choices if z[1]>=500],key=lambda z:(z[0],z[1]))
base_va=pva>=thr; base_te=pte>=thr
# predefined interpretable SHORT context conditions; no test-derived thresholds
conds={
 'ema100_below': lambda z: z.ema100_rel<=0,
 'ema100_below_0.5pct': lambda z: z.ema100_rel<=-0.005,
 'ema100_slope_down': lambda z: z.ema100_slope3<=0,
 'ema50_slope_down': lambda z: z.ema50_slope3<=0,
 'ema10_slope_down': lambda z: z.ema10_slope3<=0,
 'ret60_down': lambda z: z.ret_60<=0,
 'ret240_down': lambda z: z.ret_240<=0,
 'downfrac20_ge_50': lambda z: z.down_frac20>=0.50,
 'downfrac20_ge_55': lambda z: z.down_frac20>=0.55,
 'imb6_negative': lambda z: z.imb_mean6<=0,
 'atr_not_extreme': lambda z: z.atr14_pct<=float(X.iloc[tr].atr14_pct.quantile(.90)),
 'rv60_not_extreme': lambda z: z.rv60<=float(X.iloc[tr].rv60.quantile(.90)),
}

def metrics(ix,p,base,rule):
    z=X.iloc[ix]; mask=base & np.asarray(rule(z),dtype=bool); n=int(mask.sum()); hit=float(y[ix][mask].mean()) if n else None; return n,hit
base_vn=int(base_va.sum()); base_vh=float(y[va][base_va].mean()); base_tn=int(base_te.sum()); base_th=float(y[te][base_te].mean())
res=[]
items=list(conds.items())
for k in [1,2,3]:
  for comb in itertools.combinations(items,k):
    name=' & '.join(x[0] for x in comb)
    def rule(z,cc=comb):
        out=np.ones(len(z),dtype=bool)
        for _,fn in cc: out &= np.asarray(fn(z),dtype=bool)
        return out
    vn,vh=metrics(va,pva,base_va,rule); tn,th=metrics(te,pte,base_te,rule)
    if vn < max(250,int(base_vn*.15)): continue
    res.append({'rule':name,'val_n':vn,'val_hit':vh,'val_lift':vh-base_vh,'val_keep':vn/base_vn,'test_n':tn,'test_hit':th,'test_lift':(th-base_th if th is not None else None),'test_keep':tn/base_tn if base_tn else 0})
res.sort(key=lambda r:(r['val_lift'],r['val_hit'],r['val_n']),reverse=True)
print('THRESHOLD',thr,'FRACTION',frac)
print('BASE_VAL',base_vn,round(base_vh,4),'BASE_TEST',base_tn,round(base_th,4))
for r in res[:20]: print(json.dumps(r,sort_keys=True))
# save complete
Path('research/short/results/SHORT1_INJ_MARKET_CONTEXT_V3.json').write_text(json.dumps({'threshold':thr,'fraction':frac,'base_val_n':base_vn,'base_val_hit':base_vh,'base_test_n':base_tn,'base_test_hit':base_th,'rules':res},indent=2,allow_nan=False))
