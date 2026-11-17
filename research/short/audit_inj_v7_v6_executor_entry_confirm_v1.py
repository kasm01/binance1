#!/usr/bin/env python3
"""Call ACTUAL SHORT executor entry confirmation method with as-of source snapshot.

H+L reconstructed from archived V6 partial 1m hazard; not future 1m candle.
Read only Redis and CSV. No executor init, orders, services or Redis writes.
"""
from pathlib import Path
from collections import Counter
import json,redis
import pandas as pd
from core.trade_executor_short import TradeExecutorShort

root=Path('research/results')
cross=pd.read_csv(root/'SHORT1_INJ_V7_V6_REAL_CONFIRMATION_OVERLAP_20261114.csv')
ids=set(cross.loc[cross.v6_pass & cross.source_confirmed,'event_id'])
r=redis.Redis(decode_responses=True,socket_timeout=3)
sources={}
for sid,data in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=1800):
 try:
  e=json.loads(data['json'])
  if e.get('event_id') in ids:sources[e['event_id']]=e
 except:pass
class CandleStub:
    def __init__(self,row):self.row=row
    def get_last_rows(self,symbol,interval):return [self.row,self.row]
class ExecutorStub:
    logger=None
    entry_candle_dir_interval="1m"
    short_current_reversal_min_pct=.0001
    _make_short_entry_decision=TradeExecutorShort._make_short_entry_decision
    _evaluate_short_entry_confirmation_stage=TradeExecutorShort._evaluate_short_entry_confirmation_stage
    def __init__(self,previous,current,row):
        self.previous=previous
        self.current=current
        self.short_entry_candle_service=CandleStub(row)
    def _get_entry_candle_pair(self,*,symbol,interval):
        return self.previous,self.current

rows=[]
for eid in ids:
 e=sources.get(eid)
 if not e:continue
 h=e.get('entry_hazard_snapshot') or {}
 conf=e.get('confirmation') or {}
 if h.get('available') is not True:continue
 co=float(conf['curr_open']); cc=float(conf['curr_close'])
 current_range=float(h['curr_range_pct'])*co
 lo=cc-float(h['curr_close_loc'])*current_range
 hi=lo+current_range
 if current_range<=0:continue
 assert lo<=min(co,cc)+1e-9 and hi>=max(co,cc)-1e-9,(eid,lo,hi,co,cc)
 actual=ExecutorStub(
  (float(conf['prev_open']),float(conf['prev_close'])),
  (co,cc),
  [int(conf['current_open_time']),co,hi,lo,cc])
 score=float(e['economic_probability'])
 result=actual._evaluate_short_entry_confirmation_stage(
  symbol='INJUSDT',signal_score=score,
  final_min_score=float(e['reference_threshold']),
  progress=float(e['sniper_progress']))
 rows.append(dict(event_id=eid,model_score=score,
  archived_shadow_confirm=str(e['confirmation_state']),
  executor_state=result['state'],executor_reason=result['reason'],
  entry_quality=result.get('entry_quality'),
  body_ratio=result.get('body_ratio'),
  retrace_ratio=result.get('retrace_ratio'),
  previous_green=result.get('previous_green'),
  curr_open=co,curr_close=cc,reconstructed_hi=hi,reconstructed_lo=lo,
  score_gate_pass=score>=float(e['reference_threshold']),
  source_event_time_ms=int(float(e['ts'])*1000)))
df=pd.DataFrame(rows).sort_values('source_event_time_ms')
p=root/'SHORT1_INJ_V7_V6_REAL_EXECUTOR_CONFIRM_SNAPSHOT_20261114.csv'
df.to_csv(p,index=False)
summ=dict(expected_real_confirm_model_eligible=len(ids),
  snapshot_reconstructed=len(df),
  executor_states=dict(Counter(df.executor_state)),
  executor_reasons=dict(Counter(df.executor_reason)),
  eligible_real_executor_ready=int((df.executor_state=='READY').sum()),
  caveat='Partial 1m OHLC reconstructed from archived as-of V6 hazard snapshot; source V6 confirmation does not contain entire official real executor snapshot and no Master/Risk authorization')
(root/'SHORT1_INJ_V7_V6_REAL_EXECUTOR_CONFIRM_SNAPSHOT_SUMMARY_20261114.json').write_text(json.dumps(summ,indent=2))
print(json.dumps(summ,indent=2))
print(df[['event_id','model_score','executor_state','executor_reason','entry_quality','body_ratio','retrace_ratio']].to_string(index=False))
