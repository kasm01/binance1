#!/usr/bin/env python3
"""Read-only SHORT1 Redis upstream intent temporal-contract audit."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv,json,redis
r=redis.Redis(host='localhost',port=6379,decode_responses=True)
out=[]
for eid,fields in r.xrange('exec_events_stream_short1'):
 try:o=json.loads(fields.get('json') or '{}')
 except:continue
 for i in o.get('items') or []:
  if not isinstance(i,dict) or str(i.get('symbol') or '').upper()!='INJUSDT':continue
  raw=i.get('raw')
  try:raw=json.loads(raw) if isinstance(raw,str) else raw
  except:raw={}
  raw=raw if isinstance(raw,dict) else {}
  nested=raw.get('raw')
  try:nested=json.loads(nested) if isinstance(nested,str) else nested
  except:nested={}
  nested=nested if isinstance(nested,dict) else {}
  meta=nested.get('meta') if isinstance(nested.get('meta'),dict) else {}
  out.append(dict(stream_id=eid,intent_id=i.get('intent_id'),side=i.get('side'),
    score=i.get('model_score'),gate_state=i.get('model_score_gate_state'),
    score_threshold=i.get('model_score_gate_threshold'),provider=i.get('model_provider'),
    raw_keys=','.join(raw.keys()),has_canonical_source_close=bool(
      i.get('completed_close_time') or raw.get('completed_close_time') or
      nested.get('completed_close_time') or meta.get('completed_close_time')),
    has_independent_revalidation_id=bool(i.get('revalidation_source_intent_id') or raw.get('revalidation_source_intent_id'))))
dst=Path('research/results/SHORT1_PENDING_INTENT_CAUSAL_AUDIT_20261113.csv')
dst.parent.mkdir(exist_ok=True,parents=True)
with dst.open('w',newline='') as f:
 cols=list(out[0]) if out else ['stream_id','intent_id']
 w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(out)
print('EXEC_INTENTS',len(out),'GATE_COUNTS',dict(Counter(x['gate_state'] for x in out)))
print('SOURCE_CLOSED_BAR_PROOF',sum(x['has_canonical_source_close'] for x in out),'INDEPENDENT_REVALIDATION_PROOF',sum(x['has_independent_revalidation_id'] for x in out))
print('FIRST_LAST',out[0]['stream_id'] if out else None,out[-1]['stream_id'] if out else None,'RESULT',dst)
