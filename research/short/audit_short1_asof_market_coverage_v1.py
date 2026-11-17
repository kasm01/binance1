#!/usr/bin/env python3
"""Short1 V6 480 recent events, read-only as-of market coverage, no Redis writes."""
from pathlib import Path
from collections import Counter
import json,redis,csv,time,statistics
from orchestration.context.short1_asof_market_join import build_indexes,join
ROOT=Path('research/results')
r=redis.Redis(decode_responses=True,socket_timeout=3)
v6=[]
for sid,raw in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=480):
    try: v6.append(json.loads(raw['json']))
    except (ValueError,KeyError):continue
v44=[]
for sid,raw in r.xrevrange('telemetry:short1:inj:entry_hazard_orderbook_v44_shadow',count=2400):
    try:v44.append(json.loads(raw['json']))
    except (ValueError,KeyError):continue
index,times=build_indexes(v44)
rows=[]
for evt in v6:
    x=join(evt,index,times)
    if x.get('state')=='REJECT_SOURCE':continue
    ev=x['field_evidence']
    rows.append(dict(
        event_id=x['model_source_event_id'],
        event_ms=x['model_event_ms'],
        v6_score=evt.get('economic_probability'),
        v6_source_confirm=evt.get('confirmation_state'),
        master_state=x['state'],
        price='price' in x['available_fields'],
        atr='atr_pct' in x['available_fields'],
        spread='spread_pct' in x['available_fields'],
        ob_depth=('bid_qty_5' in x['values'] and 'ask_qty_5' in x['values']),
        oi='oi_delta_15m' in x['values'],
        flow15='money_flow_15m' in x['available_fields'],
        flow60='money_flow_60m' in x['available_fields'],
        return15='price_return_15m' in x['available_fields'],
        whale_dir='whale_dir' in x['available_fields'],
        whale_score='whale_score' in x['available_fields'],
        liq_score='liq_score' in x['available_fields'],
        ob_observation_lag_ms=ev.get('spread_pct',{}).get('age_ms'),
        ob_publication_lag_ms=(x['model_event_ms']-ev['spread_pct']['published_ms']) if 'spread_pct' in ev else None,
        missing=','.join(x['missing_fields'])))
out=ROOT/'SHORT1_INJ_ASOF_MARKET_480_COVERAGE_20261114.csv'
with out.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows)
fields=('price','atr','spread','ob_depth','oi','flow15','flow60','return15','whale_dir','whale_score','liq_score')
summ=dict(
    v6_loaded=len(v6),v44_loaded=len(v44),v44_valid_orderbook_events=len(index),
    audited=len(rows),
    valid_master=int(sum(row['master_state']!='BLOCK_MASTER' for row in rows)),
    available={field:sum(bool(row[field]) for row in rows) for field in fields},
    missing_combinations=dict(Counter(row['missing'] for row in rows)),
    orderbook_age_ms_median=statistics.median([row['ob_observation_lag_ms'] for row in rows if row['ob_observation_lag_ms'] is not None]) if any(row['ob_observation_lag_ms'] is not None for row in rows) else None,
    assumptions='Strict as-of: event publish timestamp AND orderbook observation/source <= model emission; no future; <=120s orderbook age; <=900s OI age. No fake whale or invented liq_score.',
    model_production_approved=False,
    all_routes='AUDIT_ONLY')
result=ROOT/'SHORT1_INJ_ASOF_MARKET_480_SUMMARY_20261114.json'
result.write_text(json.dumps(summ,indent=2))
print(json.dumps(summ,indent=2))
print('FILES',out,result)
