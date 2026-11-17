#!/usr/bin/env python3
"""SHORT V6 observation real-emission timestamp -> next 1m OPEN -> adverse-first TP/SL.

Only historical shadow events with score pass+source confirmation; no Master,
no native executor, no live orders, no inference about source independent
fresh model readiness. Exit at horizon is a RESEARCH artificial time exit.
"""
from pathlib import Path
from collections import Counter
import bisect,json
import numpy as np,pandas as pd
REPORT=Path('research/results')
events=pd.read_csv(REPORT/'SHORT1_INJ_V6_LIVE_CONTRA_96H_EVENTS_20261114.csv')
candles=pd.read_csv('research/short/data/INJUSDT_1m_20261110_20261114_live_contra_audit.csv')
times=candles.open_time.to_numpy('int64')
TP_MARGIN=.0125
LEVERAGE=3
SL=.0299
COST_NOTIONAL=.0014
def replay(event,horizon):
    ms=int(event.event_ms)
    idx=bisect.bisect_right(times,ms)
    if idx>=len(times) or times[idx]-ms>60000:return None
    if idx+horizon>len(candles):return None
    entry=float(candles.open.iloc[idx])
    target=entry*(1-TP_MARGIN/LEVERAGE)
    stop=entry*(1+SL)
    outcome='TIME_EXIT'
    exit_price=float(candles.close.iloc[idx+horizon-1])
    j=idx+horizon-1
    for k in range(idx,idx+horizon):
        hi=float(candles.high.iloc[k]);lo=float(candles.low.iloc[k])
        if hi>=stop:
            outcome='STOP_FIRST';exit_price=stop;j=k;break
        if lo<=target:
            outcome='TP';exit_price=target;j=k;break
    net=100*LEVERAGE*((entry-exit_price)/entry-COST_NOTIONAL)
    return dict(event_id=event.event_id,model_score=float(event.score),
      caution_uptrend=bool(event.caution_uptrend),selling_reversal=bool(event.selling_reversal),
      source_confirmed=bool(event.confirmation),model_pass=bool(event.score_pass),
      enter_ms=int(times[idx]),entry_price=entry,exit_ms=int(candles.close_time.iloc[j]),
      exit_reason=outcome,hold_minutes=j-idx+1,
      gross_roi_pct=round(100*LEVERAGE*((entry-exit_price)/entry),5),
      assumed_net_roi_pct=round(net,5),horizon=horizon)
records=[]
for _,event in events[events.old_shadow_ready].iterrows():
    for horizon in (15,30,60,120):
        res=replay(event,horizon)
        if res is not None:records.append(res)
df=pd.DataFrame(records)
file=REPORT/'SHORT1_INJ_V6_SHADOW_READY_INTRABAR_TP125_SL299_20261114.csv'
df.to_csv(file,index=False)
summary={'total_shadow_ready_events':int(events.old_shadow_ready.sum()),
   'replay_event_rows':int(len(df)),
   'same_fresh_model_intent_required':False,
   'model_target_mismatch':'V6 predicts 4% ROI target; this applies hypothetical 1.25% gross ROI exit',
   'horizons':{}}
for h,g in df.groupby('horizon'):
    parts={}
    for name,sub in [('all',g),('15m60m_uptrend',g[g.caution_uptrend]),
                     ('other_regimes',g[~g.caution_uptrend])]:
        parts[name]=dict(n=int(len(sub)),
          mean_assumed_net_roi_pct=round(float(sub.assumed_net_roi_pct.mean()),5),
          assumed_positive=int((sub.assumed_net_roi_pct>0).sum()),
          exits=dict(Counter(sub.exit_reason)))
    summary['horizons'][str(int(h))]=parts
REPORT.joinpath('SHORT1_INJ_V6_SHADOW_READY_INTRABAR_SUMMARY_20261114.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('FILE',file)
print('LIMIT: 1m candles and arbitrary hold exit; cannot replace central Risk Master, native stop or actual fill.')
