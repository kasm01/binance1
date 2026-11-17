#!/usr/bin/env python3
"""Source timestamp verified cross-check V7 frozen score vs real V6 live snapshots.

Read only: never convert V7 offline prediction into a live intent; V6 source
confirmation has model-specific target, V7 was not running at those timestamps.
"""
import redis,json
import pandas as pd
from pathlib import Path
from collections import Counter
root=Path('research/results')
v7=pd.read_csv(root/'SHORT1_INJ_V7_FROZEN_OOS_SELECTED_20261114.csv')
v7['completed_close_ms']=v7.completed_close_ms.astype('int64')
v7=v7.set_index('completed_close_ms')
r=redis.Redis(decode_responses=True,socket_timeout=3)
rows=[]
for sid,fields in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=1800):
 try:
  a=json.loads(fields['json'])
  stamp=int(a['completed_close_time'])
  if stamp not in v7.index:continue
  current=v7.loc[stamp]
  if isinstance(current,pd.DataFrame):raise RuntimeError('duplicate_3m_score')
  prov=a.get('model_provider')=='short_arm_quality_logistic_v6'
  gate=bool(prov and float(a['economic_probability'])>=float(a['reference_threshold']))
  progress=float(a.get('sniper_progress') or -1)
  confirmed=bool(a.get('confirmation_state')=='CONFIRM' and
        float(a.get('sniper_min') or 0)<=progress<=float(a.get('sniper_max') or 0))
  rows.append(dict(event_id=a['event_id'],close_ms=stamp,
       v6_model_score=float(a['economic_probability']),v6_pass=gate,
       source_confirmed=confirmed,
       source_model_target=a.get('model_target'),
       v7_frozen_score=float(current.model_score),v7_frozen_pass=True,
       historical_v7_60m_label_net_roi_pct=float(current.assumed_net_roi3x_pct),
       historical_v7_label_tp=bool(current.tp_first_event),
       historical_v7_label_stop=bool(current.stop_first_event),
       v6_entry_candle_open_ms=int((a.get('confirmation') or {}).get('current_open_time') or 0),
       v6_emitted_ms=int(float(a['ts'])*1000)))
 except (ValueError,TypeError,KeyError):continue
df=pd.DataFrame(rows)
file=root/'SHORT1_INJ_V7_V6_REAL_CONFIRMATION_OVERLAP_20261114.csv'
df.to_csv(file,index=False)
report=dict(v7_frozen_selected_total=int(len(v7)),
  v6_matched_model_events=int(len(df)),
  v6_score_pass=int(df.v6_pass.sum()) if len(df) else 0,
  v6_confirmation_pass=int(df.source_confirmed.sum()) if len(df) else 0,
  v6_and_confirmation_pass=int((df.v6_pass&df.source_confirmed).sum()) if len(df) else 0,
  v7_not_was_running_at_signal=True,
  proof_of_canonical_master_approval=False)
(root/'SHORT1_INJ_V7_V6_REAL_CONFIRMATION_OVERLAP_SUMMARY_20261114.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if len(df):print('CROSS_MATCHED_READY',df[df.v6_pass&df.source_confirmed][['event_id','close_ms','v6_model_score','v7_frozen_score','historical_v7_60m_label_net_roi_pct']].to_string(index=False))
