#!/usr/bin/env python3
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
import numpy as np
import redis

STREAM=os.getenv('SHORT1_HAZARD_OUT_STREAM','telemetry:short1:inj:entry_hazard_v1_outcomes')

def f(x,d=0.0):
    try:return float(x)
    except:return d

def flags(s):
    cur=f(s.get('curr_flow_partial')); f15=f(s.get('flow15_prev')); f3=f(s.get('flow3_prev')); acc15=f(s.get('flow_accel_partial_vs15'))
    body=f(s.get('curr_body_pct')); lw=f(s.get('curr_lower_wick_pct')); pace=f(s.get('curr_volume_pace'))
    out=[]
    if f15 < 0 and cur > 0: out.append('FLOW_FLIP_UP')
    if acc15 >= 0.50: out.append('FLOW_SHOCK_UP')
    if cur <= -0.20 and acc15 <= -0.15 and body < 0: out.append('SELL_CONTINUATION')
    if cur <= -0.30 and body < 0 and lw >= max(abs(body)*0.60, 0.00005): out.append('POSSIBLE_SELL_ABSORPTION')
    if pace >= 3.0: out.append('VOLUME_SHOCK')
    if f3 > 0 and cur < 0: out.append('FLOW_FLIP_DOWN_FROM_3M')
    return out or ['NEUTRAL']

def main():
    r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
    rows=[]
    for sid,fields in r.xrange(STREAM):
        try:
            o=json.loads(fields.get('json','{}')); s=o.get('entry_hazard_snapshot') or {}; lab=o.get('labels') or {}
            if not s.get('available'): continue
            rec={'id':sid,'score':f(o.get('economic_probability')),'flags':flags(s),'snapshot':s,'labels':lab}
            rows.append(rec)
        except Exception: continue
    print('ENTRY_HAZARD_LIVE_V1 sample=',len(rows))
    if not rows: return
    counts=Counter(x for r0 in rows for x in r0['flags'])
    print('FLAGS',dict(counts))
    groups=defaultdict(list)
    for r0 in rows:
        for fl in r0['flags']: groups[fl].append(r0)
    for fl,arr in sorted(groups.items(), key=lambda kv:(-len(kv[1]),kv[0])):
        print('\nFLAG',fl,'n',len(arr))
        for n in (1,3,5):
            vals=[x['labels'].get(f'{n}m',{}) for x in arr if x['labels'].get(f'{n}m',{}).get('available')]
            if not vals: continue
            mae=np.array([f(v.get('mae_pct')) for v in vals]); mfe=np.array([f(v.get('mfe_pct')) for v in vals]); touch=Counter(v.get('touch_40bp') for v in vals)
            print(f' {n}m mae_med={np.median(mae)*100:.3f}% mfe_med={np.median(mfe)*100:.3f}% touches={dict(touch)}')
    print('\nRECENT')
    for r0 in rows[-10:]:
        s=r0['snapshot']; print(r0['id'],'score',round(r0['score'],4),'flags',','.join(r0['flags']),'curflow',round(f(s.get('curr_flow_partial')),3),'f15',round(f(s.get('flow15_prev')),3),'acc15',round(f(s.get('flow_accel_partial_vs15')),3),'5m',r0['labels'].get('5m'))
if __name__=='__main__':main()
