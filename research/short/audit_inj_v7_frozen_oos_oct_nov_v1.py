#!/usr/bin/env python3
"""Frozen V7 SHORT TP1.25/SL2.99 out-of-time Oct13-Nov14 using 3m Binance klines.

No new training, no new threshold selection. Historical model was trained on
train period only with validation cutoff 0.712398...; source data is immutable.
Existing models/services/Redis not written. Calls only public futures KLINES
to fill 3m OOS gaps, bounded to one month; cached file reused.
"""
from pathlib import Path
from datetime import datetime,timezone
from sklearn.metrics import roc_auc_score
from features.short_economic import (
    SHORT_ECON_MODEL_FEATURES_V1, SHORT_ECON_MODEL_FEATURE_HASH_V1,
    build_short_economic_features)
import requests,time,json
import pandas as pd
import numpy as np
import joblib

ROOT=Path('research/short')
OLD=Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
CACHE=ROOT/'data/INJUSDT_3m_20261013_20261114_FROZEN_OOS.csv'
MODEL=ROOT/'models/short_model_INJUSDT_3m_tp125_sl299_v7_research_only.joblib'
PARAM=ROOT/'results/SHORT1_INJ_V7_TP125_SL299_OFFLINE_RESEARCH.json'
REPORT=Path('research/results')
END=datetime(2026,11,14,10,30,tzinfo=timezone.utc)
END_MS=int(END.timestamp()*1000)
FREQUENCY=180000
H=20
def ensure_cache():
    if CACHE.exists():
        print('CACHED_3M_OOS_FILE_REUSED',CACHE,flush=True)
        return
    old=pd.read_csv(OLD,usecols=['open_time','close_time']).iloc[-1]
    next_open=int(old.open_time)+FREQUENCY
    assert int(old.close_time)==next_open-1
    session=requests.Session();rows=[];cursor=next_open
    request_count=0
    while cursor+FREQUENCY<=END_MS:
        req=session.get('https://fapi.binance.com/fapi/v1/klines',params={
            'symbol':'INJUSDT','interval':'3m','startTime':cursor,
            'endTime':END_MS-1,'limit':1500},timeout=18)
        req.raise_for_status()
        payload=req.json()
        if not payload or not isinstance(payload,list):
            raise RuntimeError('missing_historical_3m')
        # Avoid any still-forming candle included at end.
        chunk=[z for z in payload if int(z[6])<END_MS]
        if not chunk:break
        rows.extend(chunk)
        nxt=int(chunk[-1][0])+FREQUENCY
        if nxt<=cursor:raise RuntimeError('stalled_3m_cursor')
        cursor=nxt
        request_count+=1
        print('FETCH_3M_CHUNK',request_count,'rows',len(chunk),'total',len(rows),
              'next',datetime.fromtimestamp(cursor/1000,timezone.utc).isoformat(),flush=True)
        if request_count>16:raise RuntimeError('oos_request_budget_exceeded')
        time.sleep(.17)
    if not rows:raise RuntimeError('no_oos_bars')
    cols=['open_time','open','high','low','close','volume','close_time',
          'quote_asset_volume','number_of_trades','taker_buy_base_volume',
          'taker_buy_quote_volume','ignore']
    df=pd.DataFrame(rows,columns=cols)
    df['open_time']=pd.to_numeric(df.open_time)
    df['close_time']=pd.to_numeric(df.close_time)
    if df.open_time.duplicated().any() or not (df.open_time.diff().dropna()==FREQUENCY).all():
        raise RuntimeError('oos_has_duplicates_or_gap')
    if int(df.open_time.iloc[0])!=next_open or int(df.close_time.iloc[-1])>=END_MS:
        raise RuntimeError('oos_alignment_wrong')
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(CACHE,index=False)
    print('OOS_CACHE_CREATED',len(df),CACHE,flush=True)
def run():
    ensure_cache()
    baseline=json.loads(PARAM.read_text())
    assert baseline['model_feature_hash']==SHORT_ECON_MODEL_FEATURE_HASH_V1
    cutoff=float(baseline['chosen_from_validation']['threshold'])
    train_last=int(pd.Timestamp(baseline['data_last']).value//1_000_000)
    old=pd.read_csv(OLD)
    new=pd.read_csv(CACHE)
    for col in ('open_time','close_time'):
        old[col]=pd.to_numeric(old[col])
        new[col]=pd.to_numeric(new[col])
    assert int(old.close_time.iloc[-1])==train_last
    assert int(new.open_time.iloc[0])==train_last+1
    assert (new.open_time.diff().dropna()==FREQUENCY).all()
    old_tail=old.tail(310)
    raw=pd.concat([old_tail,new],ignore_index=True)
    for col in ('open','high','low','close','volume','taker_buy_base_volume'):
        raw[col]=pd.to_numeric(raw[col])
    features=build_short_economic_features(raw)
    start=old_tail.shape[0]
    # Source V7 train contract: signal at completed bar i; enter NEXT 3m bar open.
    # Leave last H+1 bars without future label, no overlapping lookahead.
    indices=np.arange(start,len(raw)-H-1,dtype=int)
    X=features.iloc[indices][SHORT_ECON_MODEL_FEATURES_V1]
    valid=np.isfinite(X.to_numpy('float64')).all(axis=1)
    X=X.loc[valid]
    indices=indices[valid]
    assert len(indices)>1000
    model=joblib.load(MODEL)
    score=model.predict_proba(X)[:,1]
    high=raw.high.to_numpy(float)
    low=raw.low.to_numpy(float)
    op=raw.open.to_numpy(float)
    closes=raw.close.to_numpy(float)
    future=indices[:,None]+1+np.arange(H)[None,:]
    entry=op[indices+1]
    sl=entry*(1+.0299)
    tp=entry*(1-.0125/3.0)
    stop_hit=high[future]>=sl[:,None]
    tp_hit=low[future]<=tp[:,None]
    sentinel=H+1
    stop_idx=np.where(stop_hit.any(axis=1),stop_hit.argmax(axis=1),sentinel)
    tp_idx=np.where(tp_hit.any(axis=1),tp_hit.argmax(axis=1),sentinel)
    stopped=(stop_idx<=tp_idx)&(stop_idx!=sentinel)
    took=(tp_idx<stop_idx)&(tp_idx!=sentinel)
    exit_price=np.where(stopped,sl,np.where(took,tp,closes[future[:,-1]]))
    net=(entry-exit_price)/entry*3-.0014*3
    y=net>0
    selected=score>=cutoff
    # fixed time-based cooldown to eliminate overlapping labels/trades
    def no_overlap(mask):
        positions=np.flatnonzero(mask)
        chosen=[];last_exit=-1
        for j in positions:
            if indices[j]>last_exit:
                chosen.append(j)
                last_exit=indices[j]+H
        return np.array(chosen,dtype=int)
    independent=no_overlap(selected)
    periods=[
        ('all_oos',np.ones(len(indices),dtype=bool)),
        ('oct13_31',raw.open_time.to_numpy()[indices]<int(pd.Timestamp('2026-11-01T00:00:00Z').value//1_000_000)),
        ('nov01_14',raw.open_time.to_numpy()[indices]>=int(pd.Timestamp('2026-11-01T00:00:00Z').value//1_000_000)),
    ]
    stats={}
    for name,mask in periods:
        chosen=selected&mask
        indep=no_overlap(chosen)
        vals=net[chosen];v2=net[indep]
        stats[name]=dict(
          total=int(mask.sum()),passed=int(chosen.sum()),passed_fraction=float(chosen.sum()/mask.sum()),
          base_mean_net_roi_pct=float(net[mask].mean()*100),
          selected_mean_net_roi_pct=float(vals.mean()*100) if len(vals) else None,
          selected_win_fraction=float((vals>0).mean()) if len(vals) else None,
          selected_stop_fraction=float(stopped[chosen].mean()) if len(vals) else None,
          selected_tp_fraction=float(took[chosen].mean()) if len(vals) else None,
          independent_n=int(len(indep)),
          independent_mean_net_roi_pct=float(v2.mean()*100) if len(v2) else None,
          independent_wins=int(sum(v2>0)),
          independent_stop_count=int(stopped[indep].sum()),
          independent_tp_count=int(took[indep].sum()),
        )
    output=dict(
      model='short_model_INJUSDT_3m_tp125_sl299_v7_research_only.joblib',
      training_unchanged=True,promoted=False,report_only=True,
      cutoff_fixed_from_prior_validation=cutoff,
      data_end=datetime.fromtimestamp(raw.close_time.iloc[-1]/1000,timezone.utc).isoformat(),
      oos_start=datetime.fromtimestamp(raw.close_time.iloc[start]/1000,timezone.utc).isoformat(),
      oos_count=int(len(indices)),
      oos_auc=float(roc_auc_score(y.astype(int),score)) if len(set(y))==2 else None,
      fees_slip_roundtrip_notional=.0014,
      tp_margin_roi=.0125,stop_adverse_underlying_price=.0299,
      label_horizon_3m_bars=20,adverse_stop_first_same_bar=True,
      actual_native_lifecycle_simulated=False,
      sample_stats=stats)
    REPORT.mkdir(parents=True,exist_ok=True)
    path=REPORT/'SHORT1_INJ_V7_FROZEN_OOS_OCT_NOV_20261114.json'
    path.write_text(json.dumps(output,indent=2))
    selected_df=pd.DataFrame(dict(
      completed_close_ms=raw.close_time.to_numpy()[indices],
      source_price_close=closes[indices],
      model_score=score,
      frozen_gate_pass=selected,
      target_positive_after_assumed_cost=y,
      gross_roi3x_pct=100*(net+.0014*3),
      assumed_net_roi3x_pct=100*net,
      stop_first_event=stopped,tp_first_event=took))
    selected_df[selected_df.frozen_gate_pass].to_csv(
      REPORT/'SHORT1_INJ_V7_FROZEN_OOS_SELECTED_20261114.csv',index=False)
    print(json.dumps(output,indent=2))
if __name__=='__main__':run()
