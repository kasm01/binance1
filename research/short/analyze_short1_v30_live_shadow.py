#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from collections import Counter
import redis

host=os.getenv('REDIS_HOST','localhost'); port=int(os.getenv('REDIS_PORT','6379')); db=int(os.getenv('REDIS_DB','0'))
r=redis.Redis(host=host,port=port,db=db,decode_responses=True)
stream=os.getenv('SHORT1_V30_SHADOW_STREAM','telemetry:short1:v30:lifecycle_shadow_v5')
state_key=os.getenv('SHORT1_V30_SHADOW_STATE_KEY','shadow:short1:v30:v5:state')
source_stream=os.getenv('SHORT1_V30_SHADOW_SOURCE_STREAM','telemetry:short1:inj:model_shadow_v5')
count=int(os.getenv('SHORT1_V30_ANALYZE_COUNT','5000'))
expected_provider=os.getenv('SHORT1_V30_EXPECTED_PROVIDER','short_entry_quality_fresh_logistic_v5')
rows=list(reversed(r.xrevrange(stream,max='+',min='-',count=count))); objs=[]
for sid,f in rows:
    try:
        o=json.loads((f or {}).get('json') or '{}'); o['_sid']=sid; objs.append(o)
    except Exception: pass
k=Counter(str(o.get('kind') or '') for o in objs)
dec=Counter(str(o.get('reason') or '') for o in objs if o.get('kind')=='SOURCE_DECISION')
providers=Counter(str(o.get('model_provider') or '') for o in objs if o.get('kind')=='SOURCE_DECISION')
ready=sum(1 for o in objs if o.get('kind')=='SOURCE_DECISION' and o.get('decision')=='READY')
soft_ready=sum(1 for o in objs if o.get('kind')=='SOURCE_DECISION' and o.get('reason')=='soft_short_build_confirmed')
hard_ready=sum(1 for o in objs if o.get('kind')=='SOURCE_DECISION' and o.get('reason')=='hard_score_confirmed')
state={}
try: state=json.loads(r.get(state_key) or '{}')
except Exception: pass
hb=next((o for o in reversed(objs) if o.get('kind')=='HEARTBEAT'),{})
src=[]
for sid,f in r.xrevrange(source_stream,max='+',min='-',count=3):
    try:
        o=json.loads((f or {}).get('json') or '{}'); o['_sid']=sid; src.append(o)
    except Exception: pass
latest_src=src[0] if src else {}
print('SHORT1 V30 LIVE SHADOW V5')
print('events',len(objs),'kinds',dict(k))
print('source_ready',ready,'hard_ready',hard_ready,'soft_ready',soft_ready)
print('decision_reasons',dict(dec)); print('providers',dict(providers))
print('expected_provider',expected_provider,'provider_ok',bool(providers) and set(providers)=={expected_provider})
print('equity',state.get('equity'),'peak',state.get('equity_peak'),'risk_mode',state.get('risk_mode'))
print('position',json.dumps(state.get('position'),sort_keys=True)); print('closed',state.get('closed'),'wins',state.get('wins'),'losses',state.get('losses'))
print('reentry_remaining_sec',max(0.0,float(state.get('reentry_until') or 0)-time.time()))
if hb: print('last_heartbeat_age_sec',time.time()-float(hb.get('ts') or 0),'mark_price',hb.get('mark_price'),'mark_age_sec',hb.get('mark_age_sec'))
if latest_src:
 print('source_latest_age_sec',time.time()-float(latest_src.get('ts') or 0),'source_provider',latest_src.get('model_provider'),'score',latest_src.get('economic_probability'),'confirm',latest_src.get('confirmation_state'),'context_available',bool((latest_src.get('entry_market_context') or {}).get('available',False)))
else: print('source_latest_age_sec',None,'source_provider',None)
