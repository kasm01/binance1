#!/usr/bin/env python3
"""One-shot read-only V5/V6 INJ matching 3m closed-bar diagnostics (last 24h)."""
import redis,json,time,csv,collections
from pathlib import Path
r=redis.Redis(decode_responses=True,socket_timeout=2)
cut=int(time.time()*1000)-86400000
s={}
for version in ('v5','v6'):
 key=f'telemetry:short1:inj:model_shadow_{version}'
 out={}
 for sid,rec in r.xrevrange(key,count=600):
  if int(sid.split('-')[0])<cut:break
  try:x=json.loads(rec.get('json') or '{}')
  except:continue
  if x.get('symbol')!='INJUSDT' or not x.get('completed_close_time'):continue
  out[int(x['completed_close_time'])]=x
 s[version]=out
timepoints=sorted(set(s['v5'])&set(s['v6']))
data=[]
for t in timepoints:
 a,b=s['v5'][t],s['v6'][t]
 try:
  va=float(a['economic_probability'])>=float(a['reference_threshold'])
  vb=float(b['economic_probability'])>=float(b['reference_threshold'])
 except (TypeError,ValueError,KeyError):continue
 data.append(dict(completed_close_ms=t,
 v5_score=round(float(a['economic_probability']),6),v5_threshold=round(float(a['reference_threshold']),6),
 v6_score=round(float(b['economic_probability']),6),v6_threshold=round(float(b['reference_threshold']),6),
 v5_pass=va,v6_pass=vb,confirm_v5=a.get('confirmation_state'),
 confirm_v6=b.get('confirmation_state'),both_pass=va and vb,
 both_pass_confirm=va and vb and a.get('confirmation_state')=='CONFIRM' and b.get('confirmation_state')=='CONFIRM'))
p=Path('research/results/SHORT1_INJ_V5_V6_24H_NONROUTING_AGREEMENT_20261113.csv')
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['completed_close_ms'])
 w.writeheader();w.writerows(data)
print('MATCHED_COMPLETED_3M_BARS',len(data),'V5_SCORE_PASS',sum(a['v5_pass'] for a in data),
      'V6_SCORE_PASS',sum(a['v6_pass'] for a in data),
      'BOTH_PASS',sum(a['both_pass'] for a in data),
      'BOTH_PASS_AND_CONFIRM',sum(a['both_pass_confirm'] for a in data))
print('ROUTING_OFF_BOTH_STREAMS','RESULT',p)
