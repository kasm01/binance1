#!/usr/bin/env python3
from __future__ import annotations
import json, os, time, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from orchestration.event_bus.redis_bus import RedisBus
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/'research/short/results/SHORT1_INJ_V6_SHADOW_REPORT.json'
STREAM=os.getenv('SHORT1_V6_SHADOW_OUTCOME_STREAM','telemetry:short1:inj:model_shadow_v6_outcomes')
EXPECTED=os.getenv('SHORT1_V6_EXPECTED_PROVIDER','short_arm_quality_logistic_v6')
MAXN=int(os.getenv('SHORT1_V6_ANALYZER_MAXN','20000'))

def metrics(rows):
    if not rows:
        return {'n':0,'base_hit_rate':None,'selected_n':0,'selected_hit_rate':None,'selected_lift':None,'auc':None,'days':0}
    y=np.array([1 if r.get('entry_quality_hit') is True else 0 for r in rows],dtype=int)
    p=np.array([float(r.get('economic_probability') or 0.0) for r in rows],dtype=float)
    thr=np.array([float(r.get('reference_threshold') or 0.0) for r in rows],dtype=float)
    sel=p>=thr
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
    base=float(y.mean())
    sh=float(y[sel].mean()) if sel.any() else None
    days=len({time.strftime('%Y-%m-%d',time.gmtime(float(r.get('source_ts') or 0.0))) for r in rows if float(r.get('source_ts') or 0)>0})
    return {'n':int(len(rows)),'days':int(days),'base_hit_rate':base,'selected_n':int(sel.sum()),'selected_hit_rate':sh,'selected_lift':(float(sh-base) if sh is not None else None),'auc':auc,'score_mean':float(p.mean()),'reference_threshold_mean':float(thr.mean())}

bus=RedisBus(); rows=[]
for _,fields in reversed(bus.r.xrevrange(STREAM,max='+',min='-',count=MAXN)):
    try:
        o=json.loads((fields or {}).get('json') or '{}')
        if str(o.get('model_provider') or '')==EXPECTED and o.get('entry_quality_hit') is not None: rows.append(o)
    except Exception: pass
allm=metrics(rows); conf=metrics([r for r in rows if str(r.get('confirmation_state') or '')=='CONFIRM'])
status='WARMUP'
if allm['n']>=100 and allm['days']>=5: status='OBSERVE'
if allm['selected_n']>=100 and allm['days']>=10 and allm['selected_lift'] is not None and allm['selected_lift']>0: status='CANDIDATE_REVIEW'
report={'schema_version':'short1_v6_shadow_report.v1','generated_ts':time.time(),'routing':'OFF','provider':EXPECTED,'stream':STREAM,'status':status,'all':allm,'confirm':conf,'note':'Research-only prospective validation. Never auto-promotes or changes execution thresholds.'}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report,indent=2,allow_nan=False))
