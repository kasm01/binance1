#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1

DATA = ROOT / 'research/short/results/SHORT_CANDIDATE_LIFECYCLE_FEATURES.csv'
MODEL_DIR = ROOT / 'research/short/models'
RESULT_DIR = ROOT / 'research/short/results'
TOP_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
DEFAULT_FRACTION = 0.20
MIN_VALID_CONFIRM = 3


def stats(x: pd.Series) -> dict:
    a = pd.to_numeric(x, errors='coerce').dropna().to_numpy(float)
    if len(a) == 0:
        return {'n': 0, 'bps': None, 'win': None, 'pf': None}
    pos = float(a[a > 0].sum()); neg = float(-a[a < 0].sum())
    return {'n': int(len(a)), 'bps': float(a.mean()*1e4), 'win': float((a > 0).mean()), 'pf': (float(pos/neg) if neg > 0 else None)}


def make_model() -> Pipeline:
    return Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=0.35, penalty='l2', solver='liblinear', max_iter=2000, class_weight='balanced', random_state=42)),
    ])


def threshold(scores: np.ndarray, frac: float) -> float:
    return float(np.quantile(np.asarray(scores, dtype=float), 1.0-frac))


def finite_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[SHORT_ECON_MODEL_FEATURES_V1].apply(pd.to_numeric, errors='coerce')
    ok = np.isfinite(x.to_numpy(float)).all(axis=1)
    return df.loc[ok].copy()


def choose_fraction(train: pd.DataFrame, val: pd.DataFrame) -> tuple[float, dict]:
    if train.empty or val.empty or train['target'].nunique() < 2:
        return DEFAULT_FRACTION, {'reason': 'insufficient_train_or_val'}
    m = make_model().fit(train[SHORT_ECON_MODEL_FEATURES_V1], train['target'])
    ptr = m.predict_proba(train[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    pval = m.predict_proba(val[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    choices=[]
    for frac in TOP_FRACTIONS:
        thr=threshold(ptr,frac)
        sel=val[(pval>=thr)&val['state'].eq('CONFIRM')]
        s=stats(sel['net_return_pct']); choices.append({'frac':frac,'threshold':thr,**s})
    eligible=[c for c in choices if c['n']>=MIN_VALID_CONFIRM and c['bps'] is not None]
    if not eligible:
        return DEFAULT_FRACTION, {'reason':'validation_confirm_too_small','choices':choices}
    best=max(eligible,key=lambda c: ((c['bps']>0), (c['pf'] or 0.0), c['bps'], c['n']))
    return float(best['frac']), {'reason':'validation_selected','selected':best,'choices':choices}


def nested_day_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dates=sorted(df['date'].unique()); rows=[]
    # train dates[:i-1], validation date[i-1], untouched test date[i]
    for i in range(3,len(dates)):
        inner_train=df[df['date'].isin(dates[:i-1])].copy()
        val=df[df['date'].eq(dates[i-1])].copy()
        test=df[df['date'].eq(dates[i])].copy()
        frac,decision=choose_fraction(inner_train,val)
        dev=df[df['date'].isin(dates[:i])].copy()
        if dev['target'].nunique()<2: continue
        m=make_model().fit(dev[SHORT_ECON_MODEL_FEATURES_V1],dev['target'])
        pdev=m.predict_proba(dev[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
        ptest=m.predict_proba(test[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
        thr=threshold(pdev,frac)
        sel=test[(ptest>=thr)&test['state'].eq('CONFIRM')]
        s=stats(sel['net_return_pct']); base=stats(test.loc[test['state'].eq('CONFIRM'),'net_return_pct'])
        rows.append({'test_date':str(dates[i]),'validation_date':str(dates[i-1]),'train_n':len(inner_train),'val_n':len(val),'test_n':len(test),'fraction':frac,'threshold':thr,'selection_reason':decision.get('reason'),'base_n':base['n'],'base_bps':base['bps'],'base_pf':base['pf'],'selected_n':s['n'],'selected_bps':s['bps'],'selected_win':s['win'],'selected_pf':s['pf']})
    out=pd.DataFrame(rows); used=out[out.selected_n>0] if not out.empty else out
    summary={'tested_days':int(len(out)),'selected_days':int(len(used)),'positive_selected_days':int((used.selected_bps>0).sum()) if not used.empty else 0,'selected_trades':int(used.selected_n.sum()) if not used.empty else 0,'weighted_selected_bps':float(np.average(used.selected_bps,weights=used.selected_n)) if not used.empty and used.selected_n.sum()>0 else None,'worst_selected_day_bps':float(used.selected_bps.min()) if not used.empty else None}
    return out,summary


def strict_holdout(df: pd.DataFrame) -> tuple[dict,dict,float,dict]:
    dates=sorted(df['date'].unique())
    if len(dates)<4: raise ValueError('need >=4 independent dates')
    hold_date=dates[-1]; val_date=dates[-2]
    train=df[df['date'].isin(dates[:-2])].copy(); val=df[df['date'].eq(val_date)].copy(); hold=df[df['date'].eq(hold_date)].copy()
    frac,selection=choose_fraction(train,val)
    dev=df[df['date'].isin(dates[:-1])].copy()
    m=make_model().fit(dev[SHORT_ECON_MODEL_FEATURES_V1],dev['target'])
    pdev=m.predict_proba(dev[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    ph=m.predict_proba(hold[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    thr=threshold(pdev,frac)
    selected=hold[(ph>=thr)&hold['state'].eq('CONFIRM')]
    result={'holdout_date':str(hold_date),'validation_date':str(val_date),'fraction':frac,'threshold':thr,**stats(selected['net_return_pct'])}
    baseline=stats(hold.loc[hold['state'].eq('CONFIRM'),'net_return_pct'])
    return result,baseline,frac,selection


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--symbol',required=True); ap.add_argument('--bot-id',default='SHORT1'); ap.add_argument('--interval',default='3m'); args=ap.parse_args()
    symbol=args.symbol.upper().strip()
    d=pd.read_csv(DATA); d=d[d.symbol.astype(str).str.upper().eq(symbol)].copy()
    if d.empty: raise SystemExit(f'no training rows for {symbol}')
    d['ts']=pd.to_datetime(d.ts,utc=True); d['date']=d.ts.dt.date
    d['net_return_pct']=pd.to_numeric(d.net_return_pct,errors='coerce')
    d=d[d['outcome'].astype(str).ne('censored') & d['net_return_pct'].notna()].copy()
    d['target']=(d['net_return_pct']>0).astype(int)
    d=finite_features(d).sort_values('ts').reset_index(drop=True)
    if len(d)<100 or d.target.nunique()<2: raise SystemExit(f'insufficient data rows={len(d)} classes={d.target.nunique()}')

    day_df,day_summary=nested_day_forward(d)
    hold,hold_base,locked_frac,selection=strict_holdout(d)
    final=make_model().fit(d[SHORT_ECON_MODEL_FEATURES_V1],d.target)
    full_scores=final.predict_proba(d[SHORT_ECON_MODEL_FEATURES_V1])[:,1]
    ref_thr=threshold(full_scores,locked_frac)

    MODEL_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    model_path=MODEL_DIR/f'short_model_{symbol}_{args.interval}_best.joblib'; meta_path=MODEL_DIR/f'short_model_meta_{symbol}_{args.interval}.json'
    day_path=RESULT_DIR/f'{args.bot_id}_{symbol}_DAY_FORWARD.csv'; report_path=RESULT_DIR/f'{args.bot_id}_{symbol}_TRAIN_REPORT.json'
    joblib.dump(final,model_path); day_df.to_csv(day_path,index=False)
    fh=hashlib.sha256('\n'.join(SHORT_ECON_MODEL_FEATURES_V1).encode()).hexdigest()
    meta={'bot_id':args.bot_id,'symbol':symbol,'side':'short','interval':args.interval,'provider':'short_economic_logistic_coin_v1','model_type':'StandardScaler+LogisticRegression','score_semantics':'short_economic_probability','target':'fee_positive_no_time_exit_lifecycle_proxy','status':'research_only','promoted':False,'features':SHORT_ECON_MODEL_FEATURES_V1,'feature_count':len(SHORT_ECON_MODEL_FEATURES_V1),'feature_hash':fh,'train_rows':int(len(d)),'train_first_ts':str(d.ts.min()),'train_last_ts':str(d.ts.max()),'train_days':int(d.date.nunique()),'positive_rate':float(d.target.mean()),'confirm_rows':int(d.state.eq('CONFIRM').sum()),'round_trip_fee_pct':0.0008,'leverage':3.0,'tp_arm_roi':0.0120,'giveback_roi':0.0024,'hard_stop_roi':-0.09,'dynamic_stop_enable':True,'dynamic_stop_confirm_sec':60.0,'dynamic_stop_catastrophic_pct':0.045,'time_exit_enable':False,'locked_top_fraction':locked_frac,'reference_probability_threshold_full_train':ref_thr,'strict_holdout':hold,'holdout_confirm_baseline':hold_base,'day_forward':day_summary,'promotion_note':'Single-coin research model only. Requires independent future shadow evidence before promotion.'}
    meta_path.write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
    def _clean_json(v):
        if isinstance(v, dict): return {k:_clean_json(x) for k,x in v.items()}
        if isinstance(v, list): return [_clean_json(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)): return None
        return v
    report={'meta':meta,'fraction_selection':selection,'day_forward_rows':day_df.to_dict(orient='records')}
    report_path.write_text(json.dumps(_clean_json(report),indent=2,allow_nan=False),encoding='utf-8')
    print('MODEL',model_path); print('META',meta_path); print('ROWS',len(d),'DAYS',d.date.nunique(),'CONFIRM',int(d.state.eq('CONFIRM').sum()))
    print('DAY_FORWARD',json.dumps(day_summary,sort_keys=True)); print('HOLD_BASE',json.dumps(hold_base,sort_keys=True)); print('STRICT_HOLDOUT',json.dumps(hold,sort_keys=True)); print('REFERENCE_THRESHOLD',ref_thr); print('STATUS research_only PROMOTED False')

if __name__=='__main__': main()
