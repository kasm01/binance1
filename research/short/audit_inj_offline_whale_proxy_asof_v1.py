#!/usr/bin/env python3
"""Existing WhaleDetector on historical closed 3m OHLCV, as-of V6 sources.

This detector is price/volume PATTERN heuristic; it cannot identify actual
large traders or replace source-verified whale_dir/whale_score in production.
No REST, Redis writes, services or orders.
"""
from pathlib import Path
from collections import Counter
import json,redis,bisect,statistics,csv
import pandas as pd
from core.whale_detector import WhaleDetector
root=Path('research/results')
r=redis.Redis(decode_responses=True,socket_timeout=3)
events=[json.loads(fields['json']) for sid,fields in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=480)]
a=pd.read_csv('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
b=pd.read_csv('research/short/data/INJUSDT_3m_20261013_20261114_FROZEN_OOS.csv')
market=pd.concat([a.tail(350),b],ignore_index=True)
detector=WhaleDetector()
features=detector.calculate_features(market)
times=market.close_time.astype('int64').tolist()
rows=[]
for e in events:
 source_ms=int(e['completed_close_time'])
 i=bisect.bisect_right(times,source_ms)-1
 if i<120 or times[i]!=source_ms:continue
 # Exactly the source completed bars, no feature from future in signal.
 signal=detector._generate_signal(features.iloc[:i+1],multiplier=1.0)
 vote=detector.final_whale_vote(signal,trade_side='short')
 if signal.direction not in ('long','short','none'):raise AssertionError('invalid direction')
 rows.append(dict(event_id=e['event_id'],source_completed_close_ms=source_ms,
   derived_signal_direction=signal.direction,derived_signal_score=signal.score,
   derived_action=vote.action,derived_regime=vote.regime,
   derived_reason=signal.reason,
   v6_source_model_score=float(e['economic_probability']),
   v6_score_pass=float(e['economic_probability'])>=float(e['reference_threshold']),
   source_whale_usable_for_master=False,
   output='RESEARCH_ONLY_OHLCV_PATTERN'))
if not rows:raise RuntimeError('no_whale_proxy_coverage')
csvfile=root/'SHORT1_INJ_WHALE_OHLCV_PROXY_480_20261114.csv'
with csvfile.open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
scores=[x['derived_signal_score'] for x in rows]
summary=dict(source_events=len(events),covered=len(rows),
  directions=dict(Counter(x['derived_signal_direction'] for x in rows)),
  votes=dict(Counter(x['derived_action'] for x in rows)),
  median_score=round(statistics.median(scores),5),
  v6_score_pass=sum(x['v6_score_pass'] for x in rows),
  master_approved=0,
  actual_whale_identity_known=False,
  warning='WhaleDetector uses OHLCV price+volume proxies, NOT independent whale source. If used as separate master whale proof may be circular with economic model; require independent OOS validation.')
report=root/'SHORT1_INJ_WHALE_OHLCV_PROXY_480_SUMMARY_20261114.json'
report.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print('FILES',csvfile,report)
