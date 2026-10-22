#!/usr/bin/env python3
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from features.short_economic import build_short_economic_features, SHORT_ECON_MODEL_FEATURES_V1
from research.short.economic_labels import ShortLifecycleSpec, build_short_economic_labels
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv')
OUT=ROOT/'research/short/results'; OUT.mkdir(parents=True,exist_ok=True)
FRACS=(.0025,.005,.01,.02,.05,.10)
SPEC=ShortLifecycleSpec(tp_price_pct=.0055/2,sl_price_pct=.0060/2,horizon_bars=6,round_trip_fee_pct=.0008,conservative_tie_stop=True)
def fit(X,y):
 return Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(C=.35,max_iter=3000,class_weight='balanced',random_state=42))]).fit(X,y)
def met(r):
 r=np.asarray(r,float); n=len(r)
 if not n:return {'n':0,'bps':np.nan,'win':np.nan,'pf':np.nan}
 pos=r[r>0].sum(); neg=-r[r<0].sum(); return {'n':n,'bps':float(r.mean()*1e4),'win':float((r>0).mean()),'pf':float(pos/max(neg,1e-12))}
d=pd.read_csv(SRC); f=build_short_economic_features(d); valid=np.isfinite(f[SHORT_ECON_MODEL_FEATURES_V1].to_numpy(float)).all(1); valid[-10:]=False
rows=np.flatnonzero(valid); lab=build_short_economic_labels(d,rows=rows,spec=SPEC); rr=lab.row.to_numpy(int); X=f.iloc[rr][SHORT_ECON_MODEL_FEATURES_V1].to_numpy(float); net=lab.net_return_pct.to_numpy(float); y=(net>0).astype(int)
ts=pd.to_datetime(pd.to_numeric(d.iloc[rr].open_time),unit='ms',utc=True)
n=len(y); hold0=int(n*.85); dev=np.arange(hold0); hold=np.arange(hold0,n); edges=np.linspace(0,hold0,6,dtype=int)
folds=[]
for seg in (2,3,4):
 tr=np.arange(0,edges[seg-1]); va=np.arange(edges[seg-1],edges[seg]); te=np.arange(edges[seg],edges[seg+1])
 m=fit(X[tr],y[tr]); pv=m.predict_proba(X[va])[:,1]; pt=m.predict_proba(X[te])[:,1]
 for frac in FRACS:
  th=float(np.quantile(pv,1-frac)); z=pt>=th; q=met(net[te][z]); folds.append({'fold':seg,'frac':frac,'threshold':th,**q})
F=pd.DataFrame(folds); choices=[]
for frac,g in F.groupby('frac'):
 gg=g[g.n>0]; N=int(gg.n.sum()); choices.append({'frac':frac,'wf_n':N,'posfold':int((gg.bps>0).sum()),'wf_bps':float(np.average(gg.bps,weights=gg.n)) if N else np.nan,'worst':float(gg.bps.min()) if len(gg) else np.nan})
C=pd.DataFrame(choices).sort_values(['posfold','wf_bps','worst','wf_n'],ascending=[False,False,False,False]); elig=C[(C.posfold>=2)&(C.wf_n>=30)]; ch=elig.iloc[0] if len(elig) else C.iloc[0]; frac=float(ch.frac)
# threshold locked from last dev slice only, not holdout
tr=np.arange(0,edges[4]); va=np.arange(edges[4],hold0); m=fit(X[tr],y[tr]); pv=m.predict_proba(X[va])[:,1]; th=float(np.quantile(pv,1-frac)); mh=fit(X[dev],y[dev]); ph=mh.predict_proba(X[hold])[:,1]; z=ph>=th; hm=met(net[hold][z]); base=met(net[hold])
res={'rows':n,'first_ts':str(ts.min()),'last_ts':str(ts.max()),'positive_rate':float(y.mean()),'selected_fraction':frac,'wf':ch.to_dict(),'locked_threshold':th,'hold_base':base,'hold_selected':hm,'hold_start_ts':str(ts.iloc[hold0])}
print(json.dumps(res,indent=2,default=float)); F.to_csv(OUT/'SHORT1_INJ_360D_ECON_V1_FOLDS.csv',index=False); C.to_csv(OUT/'SHORT1_INJ_360D_ECON_V1_CHOICES.csv',index=False); (OUT/'SHORT1_INJ_360D_ECON_V1.json').write_text(json.dumps(res,indent=2,default=float))
