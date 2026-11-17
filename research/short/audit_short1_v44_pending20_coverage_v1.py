#!/usr/bin/env python3
"""Read-only coverage V6 event at ~06s, V44 20s book arrives ~20s.
A pending observation is NOT a validated new model or executable intent.
"""
import json,csv,redis,statistics
from collections import Counter
from pathlib import Path
from orchestration.context.short1_asof_market_join import build_pending_20s_index,evaluate_pending_20s
r=redis.Redis(decode_responses=True,socket_timeout=3)
v6=[]
for sid,raw in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=480):
    v6.append(json.loads(raw['json']))
v44=[]
for sid,raw in r.xrevrange('telemetry:short1:inj:entry_hazard_orderbook_v44_shadow',count=2400):
    v44.append(json.loads(raw['json']))
groups=build_pending_20s_index(v44)
res=[]
for e in v6:
    out=evaluate_pending_20s(e,groups)
    out['model_event_id']=e.get('event_id')
    out['model_score_pass']=float(e['economic_probability'])>=float(e['reference_threshold'])
    out['source_confirmation']=e.get('confirmation_state')
    res.append(out)
counts=Counter(x['state'] for x in res)
ready=[x for x in res if x['state']=='PENDING_BOOK_EVIDENCE_AVAILABLE']
summary=dict(v6_events=len(v6),v44_pending_20s_minute_count=len(groups),
    result_states=dict(counts),
    available_rate_pct=round(len(ready)/len(res)*100,2),
    median_delay_after_v6_ms=statistics.median([x['delay_after_model_ms'] for x in ready]) if ready else None,
    median_book_age_at_publication_ms=statistics.median([x['book_age_on_publish_ms'] for x in ready]) if ready else None,
    score_pass_total=sum(x['model_score_pass'] for x in res),
    score_pass_fresh_book=sum(x['model_score_pass'] for x in ready),
    confirmed_v6_with_fresh_book=sum(x['source_confirmation']=='CONFIRM' for x in ready),
    actual_20s_entry_confirmation_revalidated=0,
    independent_new_model_event_revalidated=0,
    market_master_authorized=0,
    caveat='20s V44 data become available AFTER original V6 model event. Not eligible at 6s; only a future PENDING revalidation may use them. No whale or canonical liquidity score.')
root=Path('research/results')
(root/'SHORT1_INJ_V44_PENDING20_COVERAGE_20261114.json').write_text(json.dumps(summary,indent=2))
with (root/'SHORT1_INJ_V44_PENDING20_EVENTS_20261114.csv').open('w',newline='') as f:
    fields=['model_event_id','model_ms','minute_ms','state','delay_after_model_ms','book_age_on_publish_ms','spread_pct','bid_qty5','ask_qty5','model_score_pass','source_confirmation','executor_authorized']
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
    writer.writeheader()
    for row in res:writer.writerow(row)
print(json.dumps(summary,indent=2))
