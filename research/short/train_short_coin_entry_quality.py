#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json, sys, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, SHORT_ECON_MODEL_FEATURE_HASH_V1
DATA=ROOT/'research/short/results/SHORT_ENTRY_QUALITY_FEATURES.csv'; MODEL_DIR=ROOT/'research/short/models'; RESULT_DIR=ROOT/'research/short/results'
FRACS=(.10,.20,.30,.40,.50); DEFAULT=.30

def model(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(C=.35,penalty='l2',solver='liblinear',max_iter=2000,class_weight='balanced',random_state=42))])
def rate(x):
 a=pd.to_numeric(x,errors='coerce').dropna().to_numpy(float); return {'n':len(a),'hit_rate':float(a.mean()) if len(a) else None}
def choose(tr,va):
 if tr.empty or va.empty or tr.entry_quality_target.nunique()<2:return DEFAULT,{}
 m=model().fit(tr[SHORT_ECON_MODEL_FEATURES_V1],tr.entry_quality_target); pt=m.predict_proba(tr[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pv=m.predict_proba(va[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
 out=[]
 for f in FRACS:
  th=float(np.quantile(pt,1-f)); sel=va[(pv>=th)&va.state.eq('CONFIRM')]; base=va[va.state.eq('CONFIRM')]
  out.append({'frac':f,'threshold':th,'n':len(sel),'hit_rate':float(sel.entry_quality_target.mean()) if len(sel) else None,'base_hit_rate':float(base.entry_quality_target.mean()) if len(base) else None})
 elig=[x for x in out if x['n']>=3 and x['hit_rate'] is not None]
 if not elig:return DEFAULT,{'choices':out}
 best=max(elig,key=lambda x:(x['hit_rate']-(x['base_hit_rate'] or 0),x['hit_rate'],x['n'])); return best['frac'],{'selected':best,'choices':out}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--symbol',required=True); ap.add_argument('--bot-id',default='SHORT1'); ap.add_argument('--interval',default='3m'); a=ap.parse_args(); sym=a.symbol.upper()
 d=pd.read_csv(DATA); d=d[d.symbol.astype(str).str.upper().eq(sym)].copy(); d.ts=pd.to_datetime(d.ts,utc=True); d['date']=d.ts.dt.date
 x=d[SHORT_ECON_MODEL_FEATURES_V1].apply(pd.to_numeric,errors='coerce'); d=d[np.isfinite(x.to_numpy(float)).all(axis=1)].copy().sort_values('ts').reset_index(drop=True)
 dates=sorted(d.date.unique()); rows=[]
 for i in range(3,len(dates)):
  tr=d[d.date.isin(dates[:i-1])]; va=d[d.date.eq(dates[i-1])]; te=d[d.date.eq(dates[i])]; frac,dec=choose(tr,va); dev=d[d.date.isin(dates[:i])]
  if dev.entry_quality_target.nunique()<2: continue
  m=model().fit(dev[SHORT_ECON_MODEL_FEATURES_V1],dev.entry_quality_target); pdv=m.predict_proba(dev[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; pte=m.predict_proba(te[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; th=float(np.quantile(pdv,1-frac)); sel=te[(pte>=th)&te.state.eq('CONFIRM')]; base=te[te.state.eq('CONFIRM')]
  rows.append({'test_date':str(dates[i]),'fraction':frac,'threshold':th,'selected_n':len(sel),'selected_hit_rate':float(sel.entry_quality_target.mean()) if len(sel) else None,'base_n':len(base),'base_hit_rate':float(base.entry_quality_target.mean()) if len(base) else None})
 rdf=pd.DataFrame(rows); tr=d[d.date.isin(dates[:-2])]; va=d[d.date.eq(dates[-2])]; hold=d[d.date.eq(dates[-1])]; frac,dec=choose(tr,va); dev=d[d.date.isin(dates[:-1])]
 m=model().fit(dev[SHORT_ECON_MODEL_FEATURES_V1],dev.entry_quality_target); pdv=m.predict_proba(dev[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; ph=m.predict_proba(hold[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; th=float(np.quantile(pdv,1-frac)); sel=hold[(ph>=th)&hold.state.eq('CONFIRM')]; base=hold[hold.state.eq('CONFIRM')]
 final=model().fit(d[SHORT_ECON_MODEL_FEATURES_V1],d.entry_quality_target); ps=final.predict_proba(d[SHORT_ECON_MODEL_FEATURES_V1])[:,1]; ref=float(np.quantile(ps,1-frac))
 MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True); mp=MODEL_DIR/f'short_model_{sym}_{a.interval}_best.joblib'; jp=MODEL_DIR/f'short_model_meta_{sym}_{a.interval}.json'; rp=RESULT_DIR/f'{a.bot_id}_{sym}_ENTRY_QUALITY_WF.csv'
 joblib.dump(final,mp); rdf.to_csv(rp,index=False)
 meta={'bot_id':a.bot_id,'symbol':sym,'side':'short','interval':a.interval,'provider':'short_entry_quality_logistic_coin_v2','model_type':'StandardScaler+LogisticRegression','score_semantics':'short_economic_probability','target':'short_entry_quality_60m','observation_min':60,'target_favorable_mfe_pct':.0060,'target_max_adverse_mae_pct':.0075,'promoted':False,'status':'research_only','features':SHORT_ECON_MODEL_FEATURES_V1,'feature_count':len(SHORT_ECON_MODEL_FEATURES_V1),'feature_hash':SHORT_ECON_MODEL_FEATURE_HASH_V1,'train_rows':len(d),'train_days':len(dates),'positive_rate':float(d.entry_quality_target.mean()),'confirm_rows':int(d.state.eq('CONFIRM').sum()),'leverage':3.0,'tp_arm_roi':.012,'giveback_roi':.0024,'hard_stop_roi':-.09,'dynamic_stop_enable':True,'dynamic_stop_confirm_sec':60.0,'dynamic_stop_catastrophic_pct':.045,'time_exit_enable':False,'locked_top_fraction':frac,'reference_probability_threshold_full_train':ref,'strict_holdout':{'date':str(dates[-1]),'selected_n':len(sel),'selected_hit_rate':float(sel.entry_quality_target.mean()) if len(sel) else None,'base_n':len(base),'base_hit_rate':float(base.entry_quality_target.mean()) if len(base) else None},'day_forward':rdf.to_dict(orient='records')}
 jp.write_text(json.dumps(meta,indent=2,allow_nan=False)); print('ROWS',len(d),'DAYS',len(dates),'POS',round(meta['positive_rate'],4),'CONFIRM',meta['confirm_rows']); print('HOLD',meta['strict_holdout']); print('REF',ref,'FRAC',frac); print('PROMOTED',False)
if __name__=='__main__': main()
