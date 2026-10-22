#!/usr/bin/env python3
import json, os
from collections import Counter
import redis
r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
S43='telemetry:short1:inj:entry_hazard_policy_v43_shadow'; S44='telemetry:short1:inj:entry_hazard_orderbook_v44_shadow'
def load(s):
    out={}
    for sid,f in r.xrange(s):
        try:o=json.loads(f.get('json','{}'))
        except:continue
        c=o.get('candidate') or {}; key=(str(c.get('source_event_id') or ''),str(o.get('stage') or ''))
        if key[0]:out[key]=o
    return out
a,b=load(S43),load(S44); ks=sorted(set(a)&set(b)); print('V43',len(a),'V44',len(b),'MATCH',len(ks))
state=Counter(); fd=[];r20=[];r30=[]
for k in ks[-10:]:
    x,y=a[k],b[k]; sx=x.get('snapshot_20s') or {}; sy=y.get('snapshot_20s') or {}; x30=x.get('snapshot_30s') or {}; y30=y.get('snapshot_30s') or {}
    df=abs(float(sx.get('flow',0))-float(sy.get('flow',0))); dr20=abs(float(sx.get('return_from_open',0))-float(sy.get('return_from_open',0))); dr30=abs(float(x30.get('return_from_open',0))-float(y30.get('return_from_open',0))) if x30 and y30 else None
    fd.append(df);r20.append(dr20); r30 += [] if dr30 is None else [dr30]
    state['equal' if x.get('state')==y.get('state') else 'different']+=1
    ob=y.get('orderbook_20s') or {}; dob=y.get('orderbook_delta_20_30') or {}
    print(k,'state',x.get('state'),y.get('state'),'df',round(df,12),'dr20',round(dr20,12),'dr30',None if dr30 is None else round(dr30,12),
          'spread',round(float(ob.get('spread_bps',0)),4),'l1',round(float(ob.get('l1_imbalance',0)),4),'d5',round(float(ob.get('depth5_imbalance',0)),4),'d20',round(float(ob.get('depth20_imbalance',0)),4),'micro',round(float(ob.get('microprice_dev_bps',0)),4),'d5_20_30',dob.get('depth5_delta'))
print('STATES',dict(state),'MAX',{'flow20':max(fd) if fd else None,'ret20':max(r20) if r20 else None,'ret30':max(r30) if r30 else None})
