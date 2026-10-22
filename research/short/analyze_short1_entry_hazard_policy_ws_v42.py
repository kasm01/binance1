#!/usr/bin/env python3
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import redis

POLICY=os.getenv('SHORT1_HAZARD_POLICY_STREAM','telemetry:short1:inj:entry_hazard_policy_v1')
OUT=os.getenv('SHORT1_HAZARD_POLICY_OUTCOME_STREAM','telemetry:short1:inj:entry_hazard_policy_v1_outcomes_ws')
DEST=Path(os.getenv('SHORT1_HAZARD_POLICY_WS_REPORT','research/short/results/SHORT1_ENTRY_HAZARD_POLICY_WS_V42.json'))
ELIGIBLE={'READY','PENDING_CONTEXT'}

def f(v,d=0.0):
    try:return float(v)
    except:return d

def stats(arr:List[Dict[str,Any]])->Dict[str,Any]:
    if not arr:return {'n':0}
    l5=[x.get('labels_from_30s_decision',{}).get('5m',{}) for x in arr]
    l5=[x for x in l5 if x.get('available')]
    touch=Counter(x.get('touch_40bp') for x in l5)
    return {
      'n':len(arr),'labeled_5m':len(l5),
      'adverse_first_rate': touch.get('ADVERSE_FIRST',0)/len(l5) if l5 else None,
      'favorable_first_rate': touch.get('FAVORABLE_FIRST',0)/len(l5) if l5 else None,
      'neither_rate': touch.get('NEITHER',0)/len(l5) if l5 else None,
      'mae5_median_pct': float(np.median([f(x.get('mae_pct')) for x in l5])) if l5 else None,
      'mfe5_median_pct': float(np.median([f(x.get('mfe_pct')) for x in l5])) if l5 else None,
      'touch_counts': dict(touch),
    }

def main():
    r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
    policies={}
    stages=Counter(); states=Counter(); gates=Counter()
    for sid,fields in r.xrange(POLICY):
        try:o=json.loads(fields.get('json','{}'))
        except:continue
        stages[str(o.get('stage'))]+=1; states[str(o.get('state'))]+=1
        cand=o.get('candidate') if isinstance(o.get('candidate'),dict) else {}
        gates[str(cand.get('model_gate_shadow_state') or 'UNKNOWN')]+=1
        if o.get('stage')=='30S_REVALIDATION' and o.get('event_id'):
            policies[str(o['event_id'])]=o
    outcomes=[]
    for sid,fields in r.xrange(OUT):
        try:o=json.loads(fields.get('json','{}'))
        except:continue
        if str(o.get('policy_event_id') or '') in policies:
            outcomes.append(o)
    groups=defaultdict(list); eligible=[]
    for o in outcomes:
        groups[str(o.get('policy_state') or 'UNKNOWN')].append(o)
        if str(o.get('model_gate_shadow_state') or '') in ELIGIBLE: eligible.append(o)
    report={
      'policy_events':sum(stages.values()),'policy_30s_events':len(policies),'outcome_events':len(outcomes),
      'stages':dict(stages),'states':dict(states),'model_gate_states':dict(gates),
      'all_labeled':stats(outcomes),
      'by_policy_state':{k:stats(v) for k,v in sorted(groups.items())},
      'eligible_model_gate_labeled':stats(eligible),
      'eligible_model_gate_states':sorted(ELIGIBLE),
    }
    DEST.parent.mkdir(parents=True,exist_ok=True); DEST.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('SAVED',DEST)
if __name__=='__main__':main()
