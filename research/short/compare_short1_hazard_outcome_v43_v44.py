#!/usr/bin/env python3
import json,os,redis
r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
S43='telemetry:short1:inj:entry_hazard_policy_v43_outcomes_ws'; S44='telemetry:short1:inj:entry_hazard_orderbook_v44_outcomes'
def load(s):
 out={}
 for sid,f in r.xrange(s):
  try:o=json.loads(f.get('json','{}'))
  except:continue
  k=str(o.get('source_event_id') or '')
  if k:out[k]=o
 return out
a,b=load(S43),load(S44);ks=sorted(set(a)&set(b));print('V43',len(a),'V44',len(b),'MATCH',len(ks))
for k in ks[-10:]:
 x,y=a[k],b[k]; print('\n',k,x.get('policy_state'),y.get('policy_state'),x.get('model_gate_shadow_state'),y.get('model_gate_shadow_state'))
 for m in ('1m','3m','5m'):
  lx=(x.get('labels_from_30s_decision') or {}).get(m) or {}; ly=(y.get('labels_from_30s_decision') or {}).get(m) or {}
  print(m,'touch',lx.get('touch_40bp'),ly.get('touch_40bp'),'mae_delta',abs(float(lx.get('mae_pct',0))-float(ly.get('mae_pct',0))),
        'mfe_delta',abs(float(lx.get('mfe_pct',0))-float(ly.get('mfe_pct',0))),'ticks',lx.get('tick_count'),ly.get('tick_count'))
