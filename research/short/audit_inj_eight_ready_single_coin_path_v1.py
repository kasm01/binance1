#!/usr/bin/env python3
"""8 observed source model + frozen research score + real entry confirm, SHORT only.

No production master/risk permission, so these are conditional price-path
candidate studies; first 1m OPEN after model timestamp, no current 1m lookahead.
No actual 2-second TP poll or exchange fill. Stops before TP when same candle.
"""
import bisect,json
from collections import Counter
from pathlib import Path
import pandas as pd,numpy as np
out=Path('research/results')
candidates=pd.read_csv(out/'SHORT1_INJ_V7_SHADOW_PENDING_CAUSAL_REPLAY_20261114.csv')
candidates=candidates[(candidates.v7_frozen_pass) &
    candidates.shadow_fsm_decision.isin(['SHADOW_READY_DIRECT','SHADOW_READY_FRESH_MODEL']) &
    (candidates.actual_executor_confirmation=='READY')].copy().sort_values('event_time_ms')
assert len(candidates)==8
m=pd.read_csv('research/short/data/INJUSDT_1m_20261110_20261114_live_contra_audit.csv')
o=m.open_time.to_numpy('int64')
outcomes=[]
for _,row in candidates.iterrows():
 idx=bisect.bisect_right(o,int(row.event_time_ms))
 if idx+60>len(m):continue
 entry=float(m.open.iloc[idx]);sl=entry*1.0299;tp=entry*(1-.0125/3)
 touch=None;closing_confirm=None
 for j in range(idx,idx+60):
  hi=float(m.high.iloc[j]);lo=float(m.low.iloc[j])
  if touch is None:
   if hi>=sl:touch=dict(exit_reason='STOP',exit_time_ms=int(m.close_time.iloc[j]),exit_price=sl,minutes=j-idx+1);break
   if lo<=tp:touch=dict(exit_reason='TP',exit_time_ms=int(m.close_time.iloc[j]),exit_price=tp,minutes=j-idx+1)
  if closing_confirm is None and float(m.close.iloc[j])<=tp:
   closing_confirm=dict(confirm_time_ms=int(m.close_time.iloc[j]),price_next_open=float(m.open.iloc[j+1]) if j+1<len(m) else None)
  if touch is not None and closing_confirm is not None:break
 if touch is None:
  j=idx+59
  touch=dict(exit_reason='TIME60',exit_time_ms=int(m.close_time.iloc[j]),exit_price=float(m.close.iloc[j]),minutes=60)
 # touch ideal threshold execution proxy
 touch_roi=300*((entry-touch['exit_price'])/entry-.0014)
 # 1m close confirmed then enter next 1m OPEN exit execution proxy (slower than 2s loop)
 if closing_confirm:
  next_roi=300*((entry-closing_confirm['price_next_open'])/entry-.0014)
 else:next_roi=None
 outcomes.append(dict(source_event_id=row.source_event_id,
   source_model_time_ms=int(row.event_time_ms),
   fsm_state=row.shadow_fsm_decision,model_v7_score=row.v7_frozen_score,
   entry_ms=int(o[idx]),entry_price=entry,
   exit_time_ms=touch['exit_time_ms'],exit_reason=touch['exit_reason'],
   minutes_to_trigger=touch['minutes'],
   touch_net_roi_pct=round(touch_roi,5),
   next_open_after_confirm_net_roi_pct=round(next_roi,5) if next_roi is not None else None,
   close_confirm_available=bool(closing_confirm)))
frame=pd.DataFrame(outcomes).sort_values('entry_ms')
# Conservative single-position rule: no new entry if prior position's potential
# exit is in same or later 1m than new source entry. Since 1m OHLC cannot give
# precise intraminute order, a whole exit candle is reserved until next minute.
last_flat_ms=-1
accepted=[]
for _,row in frame.iterrows():
 if int(row.entry_ms)>last_flat_ms:
  accepted.append(row)
  last_flat_ms=(int(row.exit_time_ms)//60000+1)*60000
actual=pd.DataFrame(accepted)
p=out/'SHORT1_INJ_8_EXECUTOR_READY_SINGLE_COIN_REPLAY_20261114.csv'
frame.to_csv(p,index=False)
s=dict(source_ready=8,covered=len(frame),single_coin_nonoverlapping=len(actual),
 conditional_tp_touch=int((frame.exit_reason=='TP').sum()),
 conditional_stop_touch=int((frame.exit_reason=='STOP').sum()),
 mean_ideal_touch_net_roi_pct=round(float(frame.touch_net_roi_pct.mean()),5),
 single_coin_ideal_touch_net_roi_pct=round(float(actual.touch_net_roi_pct.mean()),5),
 single_coin_ideal_touch_sum_roi_pct=round(float(actual.touch_net_roi_pct.sum()),5),
 close_confirm_available=int(frame.close_confirm_available.sum()),
 mean_one_minute_close_confirm_then_next_open_net_roi_pct=round(float(frame.next_open_after_confirm_net_roi_pct.mean()),5),
 single_coin_one_minute_close_confirm_then_next_open_mean_pct=round(float(actual.next_open_after_confirm_net_roi_pct.mean()),5),
 warning='Price-path diagnostic, 8 selected events; no actual position, central Master/Risk, market-order fill, native Algo stop or 2-second lifecycle timing, and not a full OOS model expectancy.')
(out/'SHORT1_INJ_8_EXECUTOR_READY_SINGLE_COIN_SUMMARY_20261114.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))
print('ROWS')
print(frame[['source_model_time_ms','entry_price','minutes_to_trigger','exit_reason','touch_net_roi_pct','next_open_after_confirm_net_roi_pct']].to_string(index=False))
