#!/usr/bin/env python3
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'
OUT=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V38'; OUT.mkdir(parents=True,exist_ok=True)

def pf(p):
    p=np.asarray(p,float); gp=p[p>0].sum(); gl=-p[p<0].sum(); return gp/gl if gl>0 else np.nan

def metrics_subset(d,name):
    x=d.copy(); p=pd.to_numeric(x.pnl_net,errors='coerce').dropna().to_numpy(float)
    return {'group':name,'n':len(x),'adverse_rate':float((x.touch40_5m=='ADVERSE_FIRST').mean()),
            'favorable_rate':float((x.touch40_5m=='FAVORABLE_FIRST').mean()),
            'win_rate':float(pd.to_numeric(x.final_win,errors='coerce').mean()),'pnl_sum_diag':float(np.nansum(x.pnl_net)),
            'pf_diag':float(pf(p)) if len(p) else np.nan,'mae5_med':float(np.nanmedian(x.mae5)),'mfe5_med':float(np.nanmedian(x.mfe5))}

def feature_sets(cols):
    score=['score']
    f20=[x for x in ['score','flow_20s','ret_20s','range_20s','close_loc_20s','volume_pace_mean_20s','trade_count_20s','avg_trade_quote_20s','flow_delta_5_20','sell_eff_20s'] if x in cols]
    mult=['score']+[x for x in cols if any(x.startswith(p) for p in ['flow_','ret_','range_','close_loc_','volume_pace_mean_','trade_count_','avg_trade_quote_','sell_eff_']) and ('full' not in x and 'baseline' not in x)]
    # remove target/result-derived/non-snapshot fields if prefixes collide
    mult=[x for x in dict.fromkeys(mult) if x not in ['flow15_baseline','flow60_baseline']]
    conf=[x for x in ['score','first_confirm_sec','first_confirm_flow','first_confirm_ret','first_confirm_range','first_confirm_close_loc','first_confirm_quote','first_confirm_agg_count','flow15_baseline','flow60_baseline'] if x in cols]
    allf=list(dict.fromkeys(mult+conf))
    return {'SCORE':score,'FIXED20':f20,'MULTICUT':mult,'FIRST_CONFIRM':conf,'ALL':allf}

def expanding_oos(d,features,target='hazard'):
    z=d[['entry_ts',target,'pnl_net','final_win','touch40_5m']+features].copy().sort_values('entry_ts').reset_index(drop=True)
    z=z[z[target].notna()].reset_index(drop=True)
    n=len(z); train0=max(120,int(n*.40)); edges=np.linspace(train0,n,5,dtype=int)
    preds=[]; folds=[]
    for k in range(4):
        tr=z.iloc[:edges[k]].copy(); te=z.iloc[edges[k]:edges[k+1]].copy()
        if te.empty or tr[target].nunique()<2: continue
        model=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('lr',LogisticRegression(max_iter=2000,class_weight='balanced',C=.5,random_state=1))])
        model.fit(tr[features],tr[target].astype(int))
        pte=model.predict_proba(te[features])[:,1]
        ptr=model.predict_proba(tr[features])[:,1]
        a=te.assign(pred=pte,fold=k+1)
        # Causal risk thresholds defined only from training prediction distribution.
        a['risk_q80']=pte>=np.quantile(ptr,.80); a['risk_q90']=pte>=np.quantile(ptr,.90)
        preds.append(a); folds.append({'fold':k+1,'train_n':len(tr),'test_n':len(te),'base':te[target].mean(),
            'auc':roc_auc_score(te[target],pte) if te[target].nunique()>1 else np.nan,
            'ap':average_precision_score(te[target],pte) if te[target].nunique()>1 else np.nan})
    o=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame(); F=pd.DataFrame(folds)
    return o,F

def summarize_model(name,o,F):
    if o.empty:return {'set':name}
    y=o.hazard.astype(int); p=o.pred
    s={'set':name,'features':None,'oos_n':len(o),'base_rate':float(y.mean()),'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),
       'fold_auc_min':float(F.auc.min()),'fold_auc_med':float(F.auc.median()),'fold_auc_max':float(F.auc.max())}
    for q in ['risk_q80','risk_q90']:
        hi=o[o[q]]; lo=o[~o[q]]
        s[q+'_n']=len(hi); s[q+'_hazard']=float(hi.hazard.mean()) if len(hi) else np.nan; s[q+'_safe_hazard']=float(lo.hazard.mean()) if len(lo) else np.nan
        s[q+'_win']=float(pd.to_numeric(hi.final_win,errors='coerce').mean()) if len(hi) else np.nan
        s[q+'_pnl_diag']=float(np.nansum(hi.pnl_net)) if len(hi) else np.nan
    return s

def main():
    d=pd.read_csv(SRC); d['entry_ts']=pd.to_datetime(d.entry_ts,utc=True)
    d['hazard']=(d.touch40_5m=='ADVERSE_FIRST').astype(int)
    d['early_confirm']=d.first_confirm_found_15s.astype(str).str.lower().isin(['true','1'])
    groups=[metrics_subset(d,'ALL_BASELINE'),metrics_subset(d[d.early_confirm],'EARLY_CONFIRM_15S'),metrics_subset(d[~d.early_confirm],'NO_EARLY_CONFIRM_15S')]
    # Extreme flow timing diagnostics
    if 'flow_20s' in d:
        groups += [metrics_subset(d[d.flow_20s<=-.6],'FLOW20_LE_-0.60'),metrics_subset(d[d.flow_20s>-.6],'FLOW20_GT_-0.60')]
    G=pd.DataFrame(groups); G.to_csv(OUT/'group_diagnostics.csv',index=False)
    print('\nGROUP DIAGNOSTICS'); print(G.round(4).to_string(index=False))
    sets=feature_sets(set(d.columns)); summaries=[]
    # Main hazard modeling only on entries that would be confirmable inside live sniper window.
    live=d[d.early_confirm].copy()
    print('\nLIVE-LIKE SAMPLE',len(live),'of',len(d),'rate',round(len(live)/len(d),4),'hazard',round(live.hazard.mean(),4))
    for name,fs in sets.items():
        if not fs:continue
        o,F=expanding_oos(live,fs)
        if not o.empty:
            o.to_csv(OUT/f'oos_{name.lower()}.csv',index=False); F.to_csv(OUT/f'folds_{name.lower()}.csv',index=False)
        s=summarize_model(name,o,F); s['features']=len(fs); summaries.append(s)
        print('\n',name,'features',len(fs),'AUC',round(s.get('auc',np.nan),4),'AP',round(s.get('ap',np.nan),4),'base',round(s.get('base_rate',np.nan),4),'fold_auc',round(s.get('fold_auc_min',np.nan),4),round(s.get('fold_auc_med',np.nan),4),round(s.get('fold_auc_max',np.nan),4))
        for q in ['risk_q80','risk_q90']:
            print(q,'n',s.get(q+'_n'),'haz',round(s.get(q+'_hazard',np.nan),4),'safe',round(s.get(q+'_safe_hazard',np.nan),4),'win',round(s.get(q+'_win',np.nan),4),'pnl_diag',round(s.get(q+'_pnl_diag',np.nan),2))
    S=pd.DataFrame(summaries).sort_values('auc',ascending=False); S.to_csv(OUT/'model_summary.csv',index=False)
    print('\nMODEL SUMMARY');print(S.round(4).to_string(index=False))
    meta={'rows':len(d),'early_confirm_rows':len(live),'target':'ADVERSE_FIRST within 5m at +/-40bp first touch; NEITHER/FAVORABLE/BOTH are 0','note':'pnl_diag uses original C600 sizing and is not a counterfactual equity replay'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__':main()
