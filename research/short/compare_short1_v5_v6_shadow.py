#!/usr/bin/env python3
from __future__ import annotations
import json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
V5=ROOT/'research/short/results/SHORT1_INJ_SHADOW_PROMOTION.json'
V6=ROOT/'research/short/results/SHORT1_INJ_V6_SHADOW_REPORT.json'
G6=ROOT/'research/short/results/SHORT1_V6_PROMOTION_GATE.json'
OUT=ROOT/'research/short/results/SHORT1_V5_V6_COMPARISON.json'
def load(p):
    try:return json.loads(p.read_text()) if p.exists() else {}
    except Exception:return {}
v5,v6,g6=load(V5),load(V6),load(G6)
e5=v5.get('entry_quality') if isinstance(v5.get('entry_quality'),dict) else {}
a6=v6.get('all') if isinstance(v6.get('all'),dict) else {}
c6=v6.get('confirm') if isinstance(v6.get('confirm'),dict) else {}
out={
 'schema_version':'short1_v5_v6_shadow_compare.v1','generated_ts':time.time(),'routing':'OFF',
 'v5':{'provider':v5.get('model_provider'),'status':v5.get('promotion_status'),'confirm_n':e5.get('confirm_n'),'selected_n':e5.get('selected_n'),'base_hit_rate':e5.get('confirm_hit_rate'),'selected_hit_rate':e5.get('selected_hit_rate'),'selected_lift':e5.get('selected_lift'),'auc':e5.get('score_auc')},
 'v6':{'provider':v6.get('provider'),'status':g6.get('promotion_status',v6.get('status')),'n':a6.get('n'),'days':a6.get('days'),'selected_n':a6.get('selected_n'),'base_hit_rate':a6.get('base_hit_rate'),'selected_hit_rate':a6.get('selected_hit_rate'),'selected_lift':a6.get('selected_lift'),'auc':a6.get('auc'),'confirm_selected_n':c6.get('selected_n'),'confirm_selected_lift':c6.get('selected_lift')},
 'decision':{'v6_better_lift':(a6.get('selected_lift') is not None and e5.get('selected_lift') is not None and float(a6.get('selected_lift'))>float(e5.get('selected_lift'))),'v6_better_auc':(a6.get('auc') is not None and e5.get('score_auc') is not None and float(a6.get('auc'))>float(e5.get('score_auc')))},
 'note':'Monitoring only. Different targets mean absolute hit rates are not directly comparable; lift/AUC and stability are primary.'
}
OUT.write_text(json.dumps(out,indent=2,allow_nan=False)); print(json.dumps(out,indent=2,allow_nan=False))
