#!/usr/bin/env python3
"""SHORT1 3m TP 1.25% ROI / price SL 2.99%, 3x, conservative 60m study.

Offline, chronological train/validation/test, event-window embargo. Does NOT
replace any live/shadow model, does NOT publish signals or Binance requests.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, SHORT_ECON_MODEL_FEATURE_HASH_V1,build_short_economic_features

SOURCE=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
OUT=ROOT/'research/short/results'
MOD=ROOT/'research/short/models'
H=20
LEVERAGE=3.0
TP_ROI=.0125
SL_PRICE=.0299
COST_NOTIONAL=.0014  # 0.08% fees + 0.06% entry/exit slippage assumption
TP_PRICE=TP_ROI/LEVERAGE
def train():
    d=pd.read_csv(SOURCE)
    for field in ('open_time','open','high','low','close','close_time'):
        d[field]=pd.to_numeric(d[field],errors='coerce')
    if d[['open_time','open','high','low','close','close_time']].isna().any().any():
        raise RuntimeError("invalid_3m_market_data")
    if not d['open_time'].is_monotonic_increasing or d.open_time.duplicated().any():
        raise RuntimeError("unsorted_or_duplicated_3m_bars")
    deltas=np.diff(d.open_time.values)
    if not (deltas==180000).all():raise RuntimeError("gaps_in_3m_market_data")
    features=build_short_economic_features(d)
    hi=d.high.to_numpy(dtype=float)
    lo=d.low.to_numpy(dtype=float)
    close=d.close.to_numpy(dtype=float)
    op=d.open.to_numpy(dtype=float)
    rows=np.arange(len(d)-H-1)
    entries=rows+1
    window=entries[:,None]+np.arange(H)[None,:]
    price=op[entries]
    stop=price*(1+SL_PRICE)
    tp=price*(1-TP_PRICE)
    stop_hit=hi[window]>=stop[:,None]
    take_hit=lo[window]<=tp[:,None]
    no_hit=H+1
    stop_i=np.where(stop_hit.any(axis=1),stop_hit.argmax(axis=1),no_hit)
    take_i=np.where(take_hit.any(axis=1),take_hit.argmax(axis=1),no_hit)
    # If same 3m OHLC bar touches stop and TP, count STOP (adverse-first).
    is_stop=(stop_i<=take_i)&(stop_i!=no_hit)
    is_take=(take_i<stop_i)&(take_i!=no_hit)
    exit_price=np.where(is_stop,stop,np.where(is_take,tp,close[window[:,-1]]))
    net_margin_roi=LEVERAGE*((price-exit_price)/price-COST_NOTIONAL)
    y=(net_margin_roi>0).astype(int)
    X=features.iloc[rows][SHORT_ECON_MODEL_FEATURES_V1].copy()
    valid=np.isfinite(X.to_numpy(float)).all(axis=1)
    X=X.loc[valid].reset_index(drop=True)
    y=y[valid]
    outcome=net_margin_roi[valid]
    stopped=is_stop[valid];taken=is_take[valid]
    rows=rows[valid]
    N=len(X)
    a=int(N*.70)
    b=int(N*.85)
    # Time embargo removes last H observations before validation / test.
    tr=np.arange(0,a-H)
    va=np.arange(a,b-H)
    te=np.arange(b,N)
    def model():
        return Pipeline([
            ('scale',StandardScaler()),
            ('clf',LogisticRegression(C=.35,class_weight='balanced',max_iter=1500,
              random_state=42,solver='liblinear'))
        ])
    clf=model().fit(X.iloc[tr],y[tr])
    p_tr=clf.predict_proba(X.iloc[tr])[:,1]
    p_va=clf.predict_proba(X.iloc[va])[:,1]
    p_te=clf.predict_proba(X.iloc[te])[:,1]
    choices=[]
    for frac in (.02,.05,.10,.15,.20,.30):
        threshold=float(np.quantile(p_tr,1-frac))
        selected=p_va>=threshold
        choices.append(dict(frac=frac,threshold=threshold,
            val_selected=int(selected.sum()),
            val_net_mean_pct=float(outcome[va][selected].mean()*100) if selected.any() else None,
            val_net_total_pct=float(outcome[va][selected].sum()*100),
            val_win_rate=float(y[va][selected].mean()) if selected.any() else None))
    eligible=[c for c in choices if c['val_selected']>=50]
    chosen=max(eligible,key=lambda c:(c['val_net_mean_pct'],c['val_net_total_pct'])) if eligible else None
    result=dict(
       study='SHORT1_INJ_tp125roi_sl299price_v7_RESEARCH_ONLY',
       promoted=False,production_authorized=False,
       data_file=str(SOURCE),rows=int(N),
       data_first=str(pd.to_datetime(d.close_time.iloc[0],unit='ms',utc=True)),
       data_last=str(pd.to_datetime(d.close_time.iloc[-1],unit='ms',utc=True)),
       train_rows=int(len(tr)),validation_rows=int(len(va)),test_rows=int(len(te)),
       test_first=str(pd.to_datetime(d.close_time.iloc[rows[te[0]]],unit='ms',utc=True)),
       test_last=str(pd.to_datetime(d.close_time.iloc[rows[te[-1]]],unit='ms',utc=True)),
       embargo_bars=int(H),horizon_bars=int(H),interval='3m',
       target_tp_gross_roi=float(TP_ROI),stop_adverse_price_pct=float(SL_PRICE),
       assumed_cost_notional=float(COST_NOTIONAL),leverage=float(LEVERAGE),
       intrabar_assumption='STOP_FIRST_IF_BOTH_TOUCHED_SAME_3M_CANDLE',
       all_sample_base_net_roi_mean_pct=float(outcome.mean()*100),
       test_base_net_roi_mean_pct=float(outcome[te].mean()*100),
       test_base_winner_rate=float(y[te].mean()),
       test_base_stop_rate=float(stopped[te].mean()),
       test_base_tp_rate=float(taken[te].mean()),
       val_auc=float(roc_auc_score(y[va],p_va)) if len(set(y[va]))==2 else None,
       test_auc=float(roc_auc_score(y[te],p_te)) if len(set(y[te]))==2 else None,
       validation_threshold_grid=choices,
       chosen_from_validation=chosen,
       target_semantics='net_positive_roi_60m_path_with_native_price_SL',
       model_feature_hash=SHORT_ECON_MODEL_FEATURE_HASH_V1,
    )
    if chosen:
        selected=p_te>=chosen['threshold']
        # Adjacent 3m observations share most of their 60m label windows.
        # A trade-like single-position proxy takes at most one 60m setup.
        raw_selected=te[selected]
        nonoverlap=[]
        last_exit_row=-1
        for idx in raw_selected:
            if int(rows[idx]) > last_exit_row:
                nonoverlap.append(int(idx))
                last_exit_row=int(rows[idx])+H
        independent=np.asarray(nonoverlap,dtype=int)
        result.update(
            test_nonoverlap_setup_count=int(len(independent)),
            test_nonoverlap_win_rate=float(y[independent].mean()) if len(independent) else None,
            test_nonoverlap_net_roi_mean_pct=float(outcome[independent].mean()*100) if len(independent) else None,
            test_nonoverlap_net_roi_sum_pct=float(outcome[independent].sum()*100),
            test_nonoverlap_stop_rate=float(stopped[independent].mean()) if len(independent) else None,
            test_nonoverlap_tp_rate=float(taken[independent].mean()) if len(independent) else None,
            break_even_win_rate_two_outcome_approx=float(
                (SL_PRICE*LEVERAGE+COST_NOTIONAL*LEVERAGE) /
                (SL_PRICE*LEVERAGE+COST_NOTIONAL*LEVERAGE+
                 TP_ROI-COST_NOTIONAL*LEVERAGE)
            ),
        )
        result.update(test_selected=int(selected.sum()),
          test_selected_net_roi_mean_pct=float(outcome[te][selected].mean()*100) if selected.any() else None,
          test_selected_net_roi_sum_pct=float(outcome[te][selected].sum()*100),
          test_selected_win_rate=float(y[te][selected].mean()) if selected.any() else None,
          test_selected_stop_rate=float(stopped[te][selected].mean()) if selected.any() else None,
          test_selected_tp_rate=float(taken[te][selected].mean()) if selected.any() else None)
    OUT.mkdir(parents=True,exist_ok=True)
    MOD.mkdir(parents=True,exist_ok=True)
    target=OUT/'SHORT1_INJ_V7_TP125_SL299_OFFLINE_RESEARCH.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False))
    # Train-only fit artifact for reproducibility; NOT replace any existing model.
    model_path=MOD/'short_model_INJUSDT_3m_tp125_sl299_v7_research_only.joblib'
    joblib.dump(clf,model_path)
    print('TRAINING_RESULT',json.dumps({k:v for k,v in result.items() if k in (
      'rows','data_first','data_last','test_first','test_last','train_rows',
      'validation_rows','test_rows','test_auc','val_auc','chosen_from_validation',
      'test_base_net_roi_mean_pct','test_base_winner_rate','test_selected',
      'test_selected_net_roi_mean_pct','test_selected_win_rate','test_selected_stop_rate',
      'test_selected_tp_rate')}))
    print('REPORT',target,'MODEL_RESEARCH_ONLY',model_path)
if __name__=='__main__':train()
