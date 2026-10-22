#!/usr/bin/env python3
from pathlib import Path
import sys
import json,hashlib,joblib,pandas as pd,numpy as np
ROOT=Path('/home/kasm920/binance1')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1
SRC=ROOT/'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'
OUT=ROOT/'research/short/models'; OUT.mkdir(parents=True,exist_ok=True)
MODEL=OUT/'short_economic_logistic_v1.joblib'; META=OUT/'short_economic_logistic_v1.json'

def main():
 d=pd.read_csv(SRC).dropna(subset=SHORT_ECON_MODEL_FEATURES_V1+['net_return_pct']).sort_values('ts').reset_index(drop=True)
 X=d[SHORT_ECON_MODEL_FEATURES_V1]; y=(d.net_return_pct.to_numpy(float)>0).astype(int)
 pipe=Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(C=.25,max_iter=1000,class_weight='balanced',random_state=1))]).fit(X,y)
 prob=pipe.predict_proba(X)[:,1]; ref=float(np.quantile(prob,.80))
 joblib.dump(pipe,MODEL)
 feature_hash=hashlib.sha256('\n'.join(SHORT_ECON_MODEL_FEATURES_V1).encode()).hexdigest()
 meta={
  'provider':'short_economic_logistic_v1','status':'research_only','promoted':False,
  'score_semantics':'short_economic_probability','target':'fee_positive_current_lifecycle_proxy',
  'round_trip_fee_pct':0.0008,'leverage':2.0,'tp_arm_roi':0.0055,'giveback_roi':0.0008,'hard_stop_roi':-0.0060,
  'features':SHORT_ECON_MODEL_FEATURES_V1,'feature_count':len(SHORT_ECON_MODEL_FEATURES_V1),'feature_hash':feature_hash,
  'train_rows':int(len(d)),'positive_rate':float(y.mean()),'train_first_ts':str(d.ts.iloc[0]),'train_last_ts':str(d.ts.iloc[-1]),
  'reference_top_fraction':0.20,'reference_probability_threshold_full_train':ref,
  'validation_note':'day-forward selected realized log matches: 27 trades, +6.25 bps net8, win 77.8%, PF 1.64; not production-approved',
 }
 META.write_text(json.dumps(meta,indent=2))
 print('MODEL',MODEL);print('META',META);print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
