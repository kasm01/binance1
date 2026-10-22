#!/usr/bin/env python3
import json, os
from collections import Counter
import redis
A=os.getenv('V1_STREAM','telemetry:short1:inj:entry_hazard_policy_v1')
B=os.getenv('V43_STREAM','telemetry:short1:inj:entry_hazard_policy_v43_shadow')

def f(v):
 try:return float(v)
 except:return None

def load(r,s):
 out={}
 for sid,fields in r.xrange(s):
  try:o=json.loads(fields.get('json','{}'))
  except:continue
  c=o.get('candidate') if isinstance(o.get('candidate'),dict) else {}
  key=(str(c.get('source_event_id') or ''),str(o.get('stage') or ''))
  if key[0]:out[key]=(sid,o)
 return out

def main():
 r=redis.Redis(decode_responses=True); a=load(r,A); b=load(r,B); keys=sorted(set(a)&set(b))
 print('V1',len(a),'V43',len(b),'MATCH',len(keys))
 states=Counter(); diffs=[]
 for k in keys[-20:]:
  sa,oa=a[k]; sb,ob=b[k]
  eq=oa.get('state')==ob.get('state'); states['state_equal' if eq else 'state_diff']+=1
  a20=oa.get('snapshot_20s') or {}; b20=ob.get('snapshot_20s') or {}; a30=oa.get('snapshot_30s') or {}; b30=ob.get('snapshot_30s') or {}
  df=(None if f(a20.get('flow')) is None or f(b20.get('flow')) is None else f(b20.get('flow'))-f(a20.get('flow')))
  dr20=(None if f(a20.get('return_from_open')) is None or f(b20.get('return_from_open')) is None else f(b20.get('return_from_open'))-f(a20.get('return_from_open')))
  dr30=(None if f(a30.get('return_from_open')) is None or f(b30.get('return_from_open')) is None else f(b30.get('return_from_open'))-f(a30.get('return_from_open')))
  diffs.append((df,dr20,dr30))
  print(k,'V1',oa.get('state'),'V43',ob.get('state'),'flow20_delta',df,'ret20_delta',dr20,'ret30_delta',dr30)
 print('STATES',dict(states))
 vals=[x for x in diffs if x[0] is not None]
 if vals:
  print('MAX_ABS flow20',max(abs(x[0]) for x in vals),'ret20',max(abs(x[1]) for x in vals if x[1] is not None),'ret30',max(abs(x[2]) for x in vals if x[2] is not None))
if __name__=='__main__':main()
