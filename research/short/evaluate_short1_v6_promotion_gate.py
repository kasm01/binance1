#!/usr/bin/env python3
from __future__ import annotations
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/short/results/SHORT1_INJ_V6_SHADOW_REPORT.json'
OUT=ROOT/'research/short/results/SHORT1_V6_PROMOTION_GATE.json'

def num(d,k):
    v=d.get(k)
    try: return float(v) if v is not None else None
    except Exception: return None
r=json.loads(SRC.read_text()) if SRC.exists() else {}
a=r.get('all') if isinstance(r.get('all'),dict) else {}
c=r.get('confirm') if isinstance(r.get('confirm'),dict) else {}
selected_n=int(a.get('selected_n') or 0); days=int(a.get('days') or 0)
lift=num(a,'selected_lift'); auc=num(a,'auc')
confirm_n=int(c.get('selected_n') or 0); confirm_days=int(c.get('days') or 0); confirm_lift=num(c,'selected_lift')
reasons=[]; status='WARMUP'
if selected_n < 100: reasons.append('selected_n_lt_100')
if days < 10: reasons.append('days_lt_10')
if lift is None: reasons.append('lift_missing')
elif lift < 0.10: reasons.append('lift_lt_0p10')
if auc is None: reasons.append('auc_missing')
elif auc < 0.60: reasons.append('auc_lt_0p60')
# Confirmation subset is a safety veto once it has minimally useful evidence.
confirm_veto = confirm_n >= 30 and confirm_days >= 5 and confirm_lift is not None and confirm_lift <= 0.0
if confirm_veto: reasons.append('confirm_lift_nonpositive')
if selected_n >= 30 and days >= 5 and ((lift is not None and lift <= 0.0) or (auc is not None and auc < 0.50)):
    status='REJECT'
elif not reasons:
    status='PRODUCTION_ELIGIBLE'
elif selected_n >= 50 and days >= 7 and lift is not None and lift >= 0.05 and auc is not None and auc >= 0.55 and not confirm_veto:
    status='CANDIDATE'
else:
    status='WARMUP'
out={
 'schema_version':'short1_v6_promotion_gate.v1','generated_ts':time.time(),'routing':'OFF','auto_promote':False,
 'provider':r.get('provider'),'promotion_status':status,'reasons':reasons,
 'metrics':{'selected_n':selected_n,'days':days,'selected_lift':lift,'auc':auc,'confirm_selected_n':confirm_n,'confirm_days':confirm_days,'confirm_selected_lift':confirm_lift},
 'criteria':{'candidate_selected_n_gte':50,'candidate_days_gte':7,'candidate_lift_gte':0.05,'candidate_auc_gte':0.55,'production_selected_n_gte':100,'production_days_gte':10,'production_lift_gte':0.10,'production_auc_gte':0.60,'confirm_veto_after_selected_n':30,'confirm_veto_after_days':5,'confirm_lift_must_be_gt':0.0},
 'note':'Research-only gate. Never promotes, arms, or changes execution thresholds.'
}
OUT.write_text(json.dumps(out,indent=2,allow_nan=False)); print(json.dumps(out,indent=2,allow_nan=False))
