#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V45B/SHORT1_BOOKDEPTH_V45B.csv'
OUT=ROOT/'research/short/results/SHORT1_BOOKDEPTH_OOS_V47'
OUT.mkdir(parents=True,exist_ok=True)
MICRO=[
 'score','flow_5s','flow_20s','flow_30s','flow_delta_5_20','flow_delta_20_30',
 'ret_5s','ret_20s','ret_30s','ret_delta_20_30','range_20s','range_30s',
 'volume_pace_mean_20s','volume_pace_mean_30s','trade_count_20s','trade_count_30s',
 'sell_eff_20s','sell_eff_30s'
]
BOOK1=[
 'bd0_1p0_notional_imb','bd30_1p0_notional_imb','bd_1p0_notional_imb_delta_0_30',
 'bd_1p0_bid_notional_ratio_0_30','bd_1p0_ask_notional_ratio_0_30',
 'bd0_1p0_depth_imb','bd30_1p0_depth_imb','bd_1p0_depth_imb_delta_0_30',
 'bd_1p0_bid_depth_ratio_0_30','bd_1p0_ask_depth_ratio_0_30'
]
BOOK02=[
 'bd0_0p2_notional_imb','bd30_0p2_notional_imb','bd_0p2_notional_imb_delta_0_30',
 'bd_0p2_bid_notional_ratio_0_30','bd_0p2_ask_notional_ratio_0_30',
 'bd0_0p2_depth_imb','bd30_0p2_depth_imb','bd_0p2_depth_imb_delta_0_30',
 'bd_0p2_bid_depth_ratio_0_30','bd_0p2_ask_depth_ratio_0_30'
]

def load():
 d=pd.read_csv(SRC);d['entry_ts']=pd.to_datetime(d.entry_ts,utc=True)
 for c in set(MICRO+BOOK1+BOOK02):d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d[d.touch40_5m!='BOTH_SAME_BAR'].copy()
 d['y']=d.touch40_5m.eq('ADVERSE_FIRST').astype(int)
 d=d[(d.ret_20s<=-0.0001) & d.bd30_1p0_notional_imb.notna()].sort_values('entry_ts').reset_index(drop=True)
 return d

def model(kind:str):
 if kind=='logistic':
  return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',LogisticRegression(C=.5,max_iter=2000,class_weight='balanced'))])
 return Pipeline([('imp',SimpleImputer(strategy='median')),('m',DecisionTreeClassifier(max_depth=3,min_samples_leaf=25,class_weight='balanced',random_state=42))])

def run_model(d,features,kind):
 tscv=TimeSeriesSplit(n_splits=4); rows=[]; preds=[]
 X=d[features];y=d.y.to_numpy()
 for fold,(tr,te) in enumerate(tscv.split(X),1):
  m=model(kind);m.fit(X.iloc[tr],y[tr]);p=m.predict_proba(X.iloc[te])[:,1]
  auc=roc_auc_score(y[te],p) if len(np.unique(y[te]))>1 else np.nan
  rows.append({'fold':fold,'train_n':len(tr),'test_n':len(te),'auc':float(auc),'test_rate':float(y[te].mean())})
  for idx,prob in zip(te,p):preds.append({'row':int(idx),'fold':fold,'prob':float(prob),'y':int(y[idx])})
 P=pd.DataFrame(preds)
 # Top 20% risk within each test fold, avoiding cross-regime calibration assumptions.
 top=[]
 for fold,g in P.groupby('fold'):
  q=g.prob.quantile(.8);z=g[g.prob>=q]; top.append({'fold':int(fold),'n':len(z),'adverse_rate':float(z.y.mean()),'base_rate':float(g.y.mean())})
 return {'folds':rows,'mean_auc':float(np.nanmean([r['auc'] for r in rows])),'top20':top}

def main():
 d=load();print('ROWS',len(d),'ADVERSE_RATE',d.y.mean())
 specs={
   'micro_logistic':(MICRO,'logistic'),
   'micro_book1_logistic':(MICRO+BOOK1,'logistic'),
   'book1_logistic':(BOOK1,'logistic'),
   'micro_tree':(MICRO,'tree'),
   'micro_book1_tree':(MICRO+BOOK1,'tree'),
   'book1_tree':(BOOK1,'tree'),
 }
 # 0.2 subset is evaluated separately using the exact same chronology after filtering.
 rep={'rows':len(d),'adverse_rate':float(d.y.mean()),'models':{}}
 for name,(f,k) in specs.items():
  rep['models'][name]=run_model(d,f,k);print(name,'mean_auc',round(rep['models'][name]['mean_auc'],4),'folds',[round(x['auc'],4) for x in rep['models'][name]['folds']])
 d02=d[d.bd30_0p2_notional_imb.notna()].reset_index(drop=True)
 rep['book02_rows']=len(d02);rep['book02_models']={}
 for name,(f,k) in {'micro_logistic':(MICRO,'logistic'),'micro_book02_logistic':(MICRO+BOOK02,'logistic'),'micro_tree':(MICRO,'tree'),'micro_book02_tree':(MICRO+BOOK02,'tree')}.items():
  rep['book02_models'][name]=run_model(d02,f,k);print('02',name,'mean_auc',round(rep['book02_models'][name]['mean_auc'],4),'folds',[round(x['auc'],4) for x in rep['book02_models'][name]['folds']])
 (OUT/'summary.json').write_text(json.dumps(rep,indent=2),encoding='utf-8')
 print('SAVED',OUT/'summary.json')
if __name__=='__main__':main()
