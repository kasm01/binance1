#!/usr/bin/env python3
import json,redis
A='telemetry:short1:inj:entry_hazard_policy_v1_outcomes_ws'
B='telemetry:short1:inj:entry_hazard_policy_v43_outcomes_ws'
def load(r,s):
 out={}
 for sid,f in r.xrange(s):
  try:o=json.loads(f.get('json','{}'))
  except:continue
  k=str(o.get('source_event_id') or '')
  if k:out[k]=(sid,o)
 return out
def main():
 r=redis.Redis(decode_responses=True);a=load(r,A);b=load(r,B);ks=sorted(set(a)&set(b));print('V42',len(a),'V43',len(b),'MATCH',len(ks))
 for k in ks[-10:]:
  _,x=a[k];_,y=b[k];print('\nSOURCE',k,'state',x.get('policy_state'),y.get('policy_state'),'gate',x.get('model_gate_shadow_state'),y.get('model_gate_shadow_state'))
  for h in ('1m','3m','5m'):
   lx=(x.get('labels_from_30s_decision') or {}).get(h) or {};ly=(y.get('labels_from_30s_decision') or {}).get(h) or {}
   print(h,'touch',lx.get('touch_40bp'),ly.get('touch_40bp'),'mae_delta',float(ly.get('mae_pct') or 0)-float(lx.get('mae_pct') or 0),'mfe_delta',float(ly.get('mfe_pct') or 0)-float(lx.get('mfe_pct') or 0),'ticks',lx.get('tick_count'),ly.get('tick_count'))
if __name__=='__main__':main()
