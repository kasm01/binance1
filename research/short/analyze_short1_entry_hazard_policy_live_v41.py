#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import redis

POLICY = os.getenv('SHORT1_HAZARD_POLICY_STREAM','telemetry:short1:inj:entry_hazard_policy_v1')
OUTCOME = os.getenv('SHORT1_HAZARD_OUTCOME_STREAM','telemetry:short1:inj:entry_hazard_v1_outcomes')
RESULT = Path('/home/kasm920/binance1/research/short/results/SHORT1_ENTRY_HAZARD_POLICY_LIVE_V41.json')


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _load_stream(r: redis.Redis, stream: str):
    out=[]
    for sid, fields in r.xrange(stream):
        try:
            obj=json.loads(fields.get('json','{}'))
            if isinstance(obj,dict): out.append((sid,obj))
        except Exception:
            pass
    return out


def main() -> None:
    r=redis.Redis(host=os.getenv('REDIS_HOST','127.0.0.1'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
    policy=_load_stream(r,POLICY)
    outcomes=_load_stream(r,OUTCOME)
    by_event={str(o.get('event_id') or ''):o for _,o in outcomes if o.get('event_id')}

    stages=Counter(); states=Counter(); joined=[]
    for sid,p in policy:
        stage=str(p.get('stage') or '')
        state=str(p.get('state') or '')
        stages[stage]+=1; states[state]+=1
        if stage!='30S_REVALIDATION':
            continue
        cand=p.get('candidate') if isinstance(p.get('candidate'),dict) else {}
        source_event_id=str(cand.get('source_event_id') or '')
        out=by_event.get(source_event_id)
        if not out:
            continue
        lab=(out.get('labels') or {}).get('5m',{})
        if not lab.get('available'):
            continue
        joined.append({
            'policy_id':sid,
            'source_event_id':source_event_id,
            'state':state,
            'score':_f(cand.get('score')),
            'model_gate_shadow_state':str(cand.get('model_gate_shadow_state') or ''),
            'touch_5m':str(lab.get('touch_40bp') or ''),
            'mae_5m':_f(lab.get('mae_pct')),
            'mfe_5m':_f(lab.get('mfe_pct')),
            'decision20':p.get('decision20') if isinstance(p.get('decision20'),dict) else {},
            'decision30':p.get('decision') if isinstance(p.get('decision'),dict) else {},
        })

    groups=defaultdict(list)
    for x in joined: groups[x['state']].append(x)
    summary={
        'policy_events':len(policy),
        'outcome_events':len(outcomes),
        'joined_30s':len(joined),
        'stages':dict(stages),
        'states':dict(states),
        'groups':{},
        'model_gate_states':dict(Counter(x['model_gate_shadow_state'] for x in joined)),
        'eligible_joined_30s':sum(1 for x in joined if x['model_gate_shadow_state'] in ('READY','PENDING_CONTEXT')),
    }
    for state, arr in sorted(groups.items()):
        touches=Counter(x['touch_5m'] for x in arr)
        mae=np.array([x['mae_5m'] for x in arr],float)
        mfe=np.array([x['mfe_5m'] for x in arr],float)
        summary['groups'][state]={
            'n':len(arr),
            'touches':dict(touches),
            'adverse_first_rate':touches.get('ADVERSE_FIRST',0)/len(arr),
            'favorable_first_rate':touches.get('FAVORABLE_FIRST',0)/len(arr),
            'mae_5m_median':float(np.median(mae)),
            'mfe_5m_median':float(np.median(mfe)),
        }

    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('SAVED',RESULT)

if __name__=='__main__': main()
