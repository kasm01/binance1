#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_ENTRY_INTELLIGENCE_V32/entry_intelligence_dataset.csv'
OUT=ROOT/'research/short/results/SHORT1_ENTRY_INTELLIGENCE_V32'

def model():
    return Pipeline([
        ('imp',SimpleImputer(strategy='median')),
        ('sc',StandardScaler()),
        ('lr',LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,solver='liblinear')),
    ])

def eval_set(df, feats, target):
    df=df.sort_values('entry_ts').reset_index(drop=True)
    # 4 expanding chronological folds, each test block ~15% after 40% warmup
    n=len(df); cuts=[(.40,.55),(.55,.70),(.70,.85),(.85,1.00)]
    rows=[]; all_oos=[]
    for k,(a,b) in enumerate(cuts,1):
        tr_end=int(n*a); te_end=int(n*b)
        tr=df.iloc[:tr_end]; te=df.iloc[tr_end:te_end].copy()
        if tr[target].nunique()<2 or te[target].nunique()<2: continue
        m=model(); m.fit(tr[feats],tr[target]); p=m.predict_proba(te[feats])[:,1]
        auc=roc_auc_score(te[target],p); ap=average_precision_score(te[target],p)
        te['risk_p']=p; te['fold']=k; all_oos.append(te)
        rows.append({'fold':k,'train_n':len(tr),'test_n':len(te),'base_rate':te[target].mean(),'auc':auc,'ap':ap})
    o=pd.concat(all_oos,ignore_index=True)
    auc=roc_auc_score(o[target],o.risk_p); ap=average_precision_score(o[target],o.risk_p)
    return rows,o,auc,ap

def summarize_gate(o,target):
    z=o.sort_values('risk_p').copy()
    out=[]
    for reject in [0.10,0.20,0.30,0.40]:
        th=z.risk_p.quantile(1-reject)
        keep=z[z.risk_p<th]; rej=z[z.risk_p>=th]
        pnl_keep=pd.to_numeric(keep.pnl_net,errors='coerce').dropna()
        pnl_all=pd.to_numeric(z.pnl_net,errors='coerce').dropna()
        out.append({
            'reject_frac':reject,'threshold':th,'kept_n':len(keep),'rejected_n':len(rej),
            'bad_rate_kept':keep[target].mean(),'bad_rate_rejected':rej[target].mean(),
            'trade_win_kept':pd.to_numeric(keep.trade_win,errors='coerce').mean(),
            'trade_win_rejected':pd.to_numeric(rej.trade_win,errors='coerce').mean(),
            'orig_pnl_all':pnl_all.sum(),'orig_pnl_kept':pnl_keep.sum(),
            'pnl_per_trade_all':pnl_all.mean(),'pnl_per_trade_kept':pnl_keep.mean(),
        })
    return pd.DataFrame(out)

def main():
    d=pd.read_csv(SRC); d['entry_ts']=pd.to_datetime(d.entry_ts,utc=True)
    sets={
      'SCORE':['score'],
      'FLOW':['flow1','flow3','flow5','flow15','flow60','flow_accel_1_5','flow_accel_3_15','buy_ratio','flow_eff_1','flow_eff_3','flow_eff_5'],
      'CANDLE':['ret1','ret3','ret5','ret15','body_pct','range_pct','close_loc','upper_wick_pct','lower_wick_pct','volz120'],
      'CONTEXT':['oi5','oi15','flow15','flow60','ret5','ret15'],
    }
    sets['ALL']=list(dict.fromkeys(sets['SCORE']+sets['FLOW']+sets['CANDLE']+['oi5','oi15']))
    target='bad_entry_3m'
    summary=[]
    best=None
    for name,feats in sets.items():
        folds,o,auc,ap=eval_set(d,feats,target)
        summary.append({'set':name,'features':len(feats),'oos_n':len(o),'auc':auc,'ap':ap,'base_rate':o[target].mean()})
        print('\nSET',name,'AUC',round(auc,4),'AP',round(ap,4),'BASE',round(o[target].mean(),4))
        print(pd.DataFrame(folds).round(4).to_string(index=False))
        g=summarize_gate(o,target)
        print('GATES')
        print(g.round(4).to_string(index=False))
        g.to_csv(OUT/f'gate_{name.lower()}_bad3m.csv',index=False)
        o[['entry_ts','risk_p',target,'trade_win','pnl_net','score']].to_csv(OUT/f'oos_{name.lower()}_bad3m.csv',index=False)
        if best is None or auc>best[0]: best=(auc,name,feats,o)
    S=pd.DataFrame(summary).sort_values('auc',ascending=False); S.to_csv(OUT/'entry_hazard_model_summary.csv',index=False)
    print('\nMODEL SUMMARY')
    print(S.round(4).to_string(index=False))
    # Fit best on all data for coefficient inspection only (not performance claim)
    _,name,feats,o=best
    m=model(); m.fit(d[feats],d[target]); co=m.named_steps['lr'].coef_[0]
    C=pd.DataFrame({'feature':feats,'coef':co,'abs_coef':np.abs(co)}).sort_values('abs_coef',ascending=False)
    C.to_csv(OUT/'best_logistic_coefficients.csv',index=False)
    print('\nBEST',name,'COEFFICIENTS')
    print(C.head(20).round(5).to_string(index=False))

if __name__=='__main__': main()
