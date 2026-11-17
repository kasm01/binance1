#!/usr/bin/env python3
"""One-to-one independent READY intent replay from existing fixed proxy data.

No external fetch, no Redis write, no actual order.
"""
from pathlib import Path
from statistics import mean,median
import csv
data=Path('research/results/SHORT1_INJ_PENDING_TO_FRESH_READY_5M_PROXY_20261113.csv')
rows=list(csv.DictReader(data.open()))
# Pair most recent pending to each distinct later READY (new model evidence).
by_ready={}
for r in rows:
 k=r['fresh_intent']
 if k not in by_ready or int(r['pending_ms'])>int(by_ready[k]['pending_ms']):
  by_ready[k]=r
unique=sorted(by_ready.values(),key=lambda x:int(x['fresh_ms']))
result=Path('research/results/SHORT1_INJ_PENDING_TO_FRESH_READY_ONE_TO_ONE_20261113.csv')
with result.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['fresh_intent']);w.writeheader();w.writerows(unique)
print('PENDING_READY_PAIRS_RAW',len(rows),'UNIQUE_READY',len(unique),'DUPLICATED_PAIRS_REMOVED',len(rows)-len(unique))
for m in (15,30,60):
 a=[float(x[f'pending_short_{m}m_pct']) for x in unique if x[f'pending_short_{m}m_pct']]
 b=[float(x[f'fresh_short_{m}m_pct']) for x in unique if x[f'fresh_short_{m}m_pct']]
 print('HORIZON_MIN',m,'n',len(b),'OLD_MEAN',round(mean(a),4) if a else None,
       'NEW_MEAN',round(mean(b),4) if b else None,
       'NEW_MEDIAN',round(median(b),4) if b else None,
       'NEW_POSITIVE',sum(x>0 for x in b))
print('SAVED',result)
