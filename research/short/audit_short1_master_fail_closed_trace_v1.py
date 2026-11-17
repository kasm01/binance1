#!/usr/bin/env python3
"""Read-only SHORT1 admission trace on actual recorded V6 source snapshots.

Calls ACTUAL executor market/context and score-gate stages WITHOUT injecting
upstream market_context_passed or model_score_gate_passed. A distinct strict
context contract audits real Master required fields to catch default masking.
Never places orders, writes Redis, changes strategies or launches workers.
"""
from pathlib import Path
from collections import Counter
import json,redis
import pandas as pd
from core.trade_executor_short import TradeExecutorShort
from orchestration.shadow.short1_v6_master_contract_v1 import audit

ROOT=Path('research/results')
candidates=pd.read_csv(ROOT/'SHORT1_INJ_V7_SHADOW_PENDING_CAUSAL_REPLAY_20261114.csv')
candidates=candidates[(candidates.v7_frozen_pass)&
    candidates.shadow_fsm_decision.isin(['SHADOW_READY_DIRECT','SHADOW_READY_FRESH_MODEL'])&
    (candidates.actual_executor_confirmation=='READY')]
assert len(candidates)==8
r=redis.Redis(decode_responses=True,socket_timeout=3)
wanted=set(candidates.source_event_id)
source={}
for sid,raw in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=2200):
    try:
        obj=json.loads(raw.get('json','{}'))
        if obj.get('event_id') in wanted:
            source[obj['event_id']]=obj
    except (ValueError,TypeError,KeyError):continue
assert len(source)==8,('missing_source_events',len(source))
executor=TradeExecutorShort.__new__(TradeExecutorShort)
rows=[]
for _,candidate in candidates.sort_values('event_time_ms').iterrows():
    obj=source[candidate.source_event_id]
    trace=dict(event_id=candidate.source_event_id,source_completed_close_ms=int(obj['completed_close_time']),
       observed_at_ms=int(obj['ts']*1000),
       v7_frozen_pass=True,
       v6_independent_model_fsm=candidate.shadow_fsm_decision,
       executor_1m_confirmation=candidate.actual_executor_confirmation)
    # Source-only checks (strict), real V6 event timestamp, not artificial model time.
    strict=audit(obj,now_ms=int(obj['ts']*1000))
    trace.update(strict_context_state=strict['state'],strict_context_reason=strict['reason'],
        strict_context_missing_fields=','.join(strict.get('missing',[])))
    # Source V6 is OFF, not a canonically admitted MasterExecutor intent.
    # Do NOT modify execution_source to orchestrated or set pass flags to true.
    extra=dict(
        execution_source='shadow',
        model_provider=obj['model_provider'],
        model_score=float(obj['economic_probability']),
        model_score_gate_threshold=float(obj['reference_threshold']),
    )
    ctx=executor._build_short_market_context_stage(probs={},extra=extra)
    score=executor._evaluate_short_model_score_gate_stage(symbol='INJUSDT',market_context=ctx)
    trace.update(executor_market_state=ctx['state'],executor_market_reason=ctx['reason'],
        executor_score_state=score['state'],executor_score_reason=score['reason'])
    trace['master_gate_state']='NOT_AUTHORIZED'
    trace['master_gate_reason']='missing_canonical_master_context_and_authoritative_score_contract'
    trace['execution_risk_state']='NOT_REACHED'
    trace['final_safety_state']='NOT_REACHED'
    trace['exchange_order_authorized']=False
    assert trace['strict_context_state']=='BLOCK_CONTEXT'
    assert trace['executor_score_state']=='REJECT'
    rows.append(trace)
frame=pd.DataFrame(rows)
path=ROOT/'SHORT1_INJ_8_REAL_READY_FAIL_CLOSED_MASTER_TRACE_20261114.csv'
frame.to_csv(path,index=False)
summary=dict(
  scope='8 real recorded shadow V6 events with retrospective V7 score and archived executor 1m READY',
  tested=len(rows),source_event_ids_unique=len(set(frame.event_id)),
  strict_master_context=dict(Counter(frame.strict_context_state)),
  actual_executor_market_context=dict(Counter(frame.executor_market_state)),
  actual_executor_score_gate=dict(Counter(frame.executor_score_state)),
  actual_executor_rejection_reasons=dict(Counter(frame.executor_score_reason)),
  all_stopped_at_or_before_master=True,
  executed_risk_stage_count=0,final_safety_pass_count=0,
  authorized_exchange_orders=0,
  missing_master_fields=sorted({s for v in frame.strict_context_missing_fields for s in v.split(',') if s}),
  warning='Dry-run admission trace, not a simulation of actual orchestrated MasterExecutor. Missing context not converted to defaults, no model score spoofing.',
)
res=ROOT/'SHORT1_INJ_8_REAL_READY_FAIL_CLOSED_MASTER_SUMMARY_20261114.json'
res.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('FILES',path,res)
