#!/usr/bin/env python3
"""Pure SHORT1 SignalBus replay, use REAL model event time, no Redis writes.

Join canonical source V5/V6 close IDs; check independent fresh source model,
then historical V7 score and archived partial 1m real executor confirmation.
Source older than 8m rejected, no sample look-ahead, all events shadow only.
"""
import json
from collections import Counter
from pathlib import Path
import redis,pandas as pd
from orchestration.shadow.short1_inj_shadow_signal_bus_v1 import process
root=Path('research/results')
v7=pd.read_csv(root/'SHORT1_INJ_V7_FROZEN_OOS_SELECTED_20261114.csv').set_index('completed_close_ms')
confirmation=pd.read_csv(root/'SHORT1_INJ_V7_V6_REAL_EXECUTOR_CONFIRM_SNAPSHOT_20261114.csv')
cstate=dict(zip(confirmation.event_id,confirmation.executor_state))
r=redis.Redis(decode_responses=True,socket_timeout=3)
events={}
for ver in ('v5','v6'):
    for sid,raw in r.xrevrange('telemetry:short1:inj:model_shadow_'+ver,count=1900):
        try:
            obj=json.loads(raw['json']);ms=int(obj['completed_close_time'])
            events.setdefault(ms,{})[ver]=obj
        except (ValueError,TypeError,KeyError):continue
state={};rows=[]
for ms,pair in sorted(events.items()):
    if 'v5' not in pair or 'v6' not in pair:continue
    a=pair['v6'];b=pair['v5']
    # Model event may occur after bar+5 seconds; using real emission avoids a
    # replay event supposedly happening BEFORE its source data existed.
    actual_ms=max(int(float(a['ts'])*1000),int(float(b['ts'])*1000))
    try:event,state=process(a,b,state,now_ms=actual_ms)
    except ValueError:continue
    hit=ms in v7.index
    rows.append(dict(close_ms=ms,source_event_id=a['event_id'],
         event_time_ms=actual_ms,
         shadow_fsm_decision=event['decision'],fsm_reason=event['reason'],
         v7_frozen_score=float(v7.loc[ms,'model_score']) if hit else None,
         v7_frozen_pass=bool(hit),
         actual_executor_confirmation=cstate.get(a['event_id'],'NOT_EVALUATED')))
df=pd.DataFrame(rows)
p=root/'SHORT1_INJ_V7_SHADOW_PENDING_CAUSAL_REPLAY_20261114.csv'
df.to_csv(p,index=False)
ready=df[df.shadow_fsm_decision.isin(['SHADOW_READY_DIRECT','SHADOW_READY_FRESH_MODEL'])]
gate=ready[ready.v7_frozen_pass]
stats=dict(
 source_paired_events=len(df),
 all_fsm_decisions=dict(Counter(df.shadow_fsm_decision)),
 shadow_fsm_ready=len(ready),
 fsm_ready_with_v7_frozen_pass=len(gate),
 fsm_ready_v7_executor_confirm_ready=int((gate.actual_executor_confirmation=='READY').sum()),
 fsm_ready_v7_executor_confirm_pending=int((gate.actual_executor_confirmation=='PENDING').sum()),
 fsm_ready_v7_executor_confirm_unknown=int((gate.actual_executor_confirmation=='NOT_EVALUATED').sum()),
 ready_decisions=dict(Counter(ready.shadow_fsm_decision)),
 ready_and_v7_states=dict(Counter(gate.actual_executor_confirmation)),
 source_time_policy='max(original V5/V6 model event ts), NOT artificial bar+5000ms',
 all_shadow_only=True,not_master_approved=True)
res=root/'SHORT1_INJ_V7_SHADOW_PENDING_CAUSAL_REPLAY_SUMMARY_20261114.json'
res.write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
print('READY_V7_ROWS')
print(gate[['close_ms','source_event_id','shadow_fsm_decision','actual_executor_confirmation','v7_frozen_score']].to_string(index=False))
