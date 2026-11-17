#!/usr/bin/env python3
"""Research SHORT1 TP+1.25: intrabar touch vs observed closed-1m close.

No claims 2-second application poll could actually capture intra-minute low.
No Binance orders, no Redis writes. All 1m data from existing Nov10-14 cache.
"""
import pandas as pd,numpy as np,bisect,json
from pathlib import Path
out=Path('research/results')
events=pd.read_csv(out/'SHORT1_INJ_V6_LIVE_CONTRA_96H_EVENTS_20261114.csv')
one=pd.read_csv('research/short/data/INJUSDT_1m_20261110_20261114_live_contra_audit.csv')
opens=one.open_time.to_numpy('int64')
records=[]
for _,evt in events[events.old_shadow_ready].iterrows():
    idx=bisect.bisect_right(opens,int(evt.event_ms))
    if idx+120>len(one):continue
    entry=float(one.open.iloc[idx]);tp=entry*(1-.0125/3);sl=entry*(1+.0299)
    for horizon in (15,30,60,120):
        subset=one.iloc[idx:idx+horizon]
        first_touch=None;first_close=None;first_sl=None
        for k,(_,bar) in enumerate(subset.iterrows(),start=1):
            if first_sl is None and float(bar.high)>=sl:first_sl=k
            if first_touch is None and float(bar.low)<=tp:first_touch=k
            if first_close is None and float(bar.close)<=tp:first_close=k
        records.append(dict(
           event_id=evt.event_id,model_score=float(evt.score),
           caution_uptrend=bool(evt.caution_uptrend),
           horizon=horizon,entry=entry,nominal_tp=tp,
           first_touch_min=first_touch,first_1m_close_min=first_close,
           first_sl_touch_min=first_sl,
           closed_below_tp_by_horizon=bool(first_close is not None),
           price_touched_tp_by_horizon=bool(first_touch is not None)))
df=pd.DataFrame(records)
file=out/'SHORT1_INJ_10_REAL_SHADOW_TP_TOUCH_VS_1M_CLOSE_20261114.csv'
df.to_csv(file,index=False)
s={}
for horizon,g in df.groupby('horizon'):
    s[str(int(horizon))]=dict(
      n=len(g),tp_intrabar_touched=int(g.price_touched_tp_by_horizon.sum()),
      tp_1m_close_verified=int(g.closed_below_tp_by_horizon.sum()),
      stop_intrabar_touched=int(g.first_sl_touch_min.notna().sum()),
      median_minute_first_touch=float(g.first_touch_min.median()),
      median_minute_first_close=float(g.first_1m_close_min.median()) if g.first_1m_close_min.notna().any() else None,
    )
(out/'SHORT1_INJ_10_REAL_SHADOW_TP_TOUCH_VS_1M_CLOSE_SUMMARY_20261114.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))
print('FILE',file)
print('LIMIT: 2-second periodic position lifecycle cannot be reconstructed from 1m candle highs/lows or closes')
