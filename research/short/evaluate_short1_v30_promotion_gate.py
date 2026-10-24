#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, time
from pathlib import Path
from collections import Counter
import redis
ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/'research/short/results'; RES.mkdir(parents=True,exist_ok=True)
OUT=RES/'SHORT1_V30_LIVE_PROMOTION_GATE.json'
MODEL_QUALITY=RES/'SHORT1_INJ_SHADOW_PROMOTION.json'
r=redis.Redis(host=os.getenv('REDIS_HOST','localhost'),port=int(os.getenv('REDIS_PORT','6379')),db=int(os.getenv('REDIS_DB','0')),decode_responses=True)
stream=os.getenv('SHORT1_V30_SHADOW_STREAM','telemetry:short1:v30:lifecycle_shadow_v5'); expected=os.getenv('SHORT1_V30_EXPECTED_PROVIDER','short_entry_quality_fresh_logistic_v5')
objs=[]
for sid,f in reversed(r.xrevrange(stream,max='+',min='-',count=20000)):
 try:
  o=json.loads((f or {}).get('json') or '{}'); o['_sid']=sid; objs.append(o)
 except Exception: pass
src=[o for o in objs if o.get('kind')=='SOURCE_DECISION']; opens=[o for o in objs if o.get('kind')=='OPEN']; closes=[o for o in objs if o.get('kind')=='CLOSE']; hbs=[o for o in objs if o.get('kind')=='HEARTBEAT']
providers=Counter(str(o.get('model_provider') or '') for o in src); provider_ok=bool(src) and set(providers)=={expected}
# Only CLOSE events are authoritative realized paper outcomes.
pnls=[float(o.get('pnl_net') or 0) for o in closes]; wins=sum(x>0 for x in pnls); losses=sum(x<=0 for x in pnls); gp=sum(x for x in pnls if x>0); gl=-sum(x for x in pnls if x<0); pf=(gp/gl if gl>0 else (999.0 if gp>0 else None))
# Unique UTC days observed by source decisions and closes.
def day(ts):
 import datetime as dt
 return dt.datetime.fromtimestamp(float(ts),dt.timezone.utc).date().isoformat()
days=sorted({day(o.get('ts') or 0) for o in src if float(o.get('ts') or 0)>0}); close_days=sorted({day(o.get('ts') or 0) for o in closes if float(o.get('ts') or 0)>0})
# Reconstruct closed-equity series from START heartbeat/open state. State equity is easier and includes fees/open position; realized p&l sum is used for gate.
net=sum(pnls); start_eq=1000.0; realized_return=net/start_eq*100.0
# Approx drawdown over CLOSE cumulative pnl only; heartbeat/state remains supplemental.
eq=start_eq; peak=eq; maxdd=0.0
for p in pnls:
 eq+=p; peak=max(peak,eq); maxdd=min(maxdd,(eq/peak-1)*100.0)
latest_hb=hbs[-1] if hbs else {}; hb_age=(time.time()-float(latest_hb.get('ts') or 0)) if latest_hb else None
# Independent model-discrimination evidence. A few lucky lifecycle closes must
# never be enough to promote a model whose live shadow ranking has collapsed.
model_quality={}
try:
 model_quality=json.loads(MODEL_QUALITY.read_text()) if MODEL_QUALITY.exists() else {}
except Exception:
 model_quality={}
quality_status=str(model_quality.get('promotion_status') or 'MISSING').upper()
quality_provider=str(model_quality.get('model_provider') or '')
quality_entry=model_quality.get('entry_quality') if isinstance(model_quality.get('entry_quality'),dict) else {}
quality_ref=model_quality.get('reference_selection_lifecycle') if isinstance(model_quality.get('reference_selection_lifecycle'),dict) else {}
quality_selected_n=int(quality_entry.get('selected_n') or 0)
quality_selected_days=int(quality_entry.get('selected_days') or 0)
quality_lift=quality_entry.get('selected_lift')
try: quality_lift=float(quality_lift) if quality_lift is not None else None
except Exception: quality_lift=None
quality_ref_n=int(quality_ref.get('n') or 0)
quality_ref_days=int(quality_ref.get('days') or 0)
quality_ref_net=quality_ref.get('net_bps')
quality_ref_pf=quality_ref.get('pf')
try: quality_ref_net=float(quality_ref_net) if quality_ref_net is not None else None
except Exception: quality_ref_net=None
try: quality_ref_pf=float(quality_ref_pf) if quality_ref_pf is not None else None
except Exception: quality_ref_pf=None
quality_provider_ok=(quality_provider==expected)
quality_candidate=(
 quality_status in {'CANDIDATE','CANDIDATE_REVIEW'} and quality_provider_ok
 and quality_selected_n>=100 and quality_selected_days>=10 and quality_lift is not None and quality_lift>=0.02
 and quality_ref_n>=100 and quality_ref_days>=10 and quality_ref_net is not None and quality_ref_net>0.0
 and quality_ref_pf is not None and quality_ref_pf>1.10
)

reasons=[]; status='OBSERVE'
if not provider_ok:
 reasons.append('provider_not_verified')
if not model_quality:
 reasons.append('model_quality_report_missing')
elif not quality_provider_ok:
 reasons.append('model_quality_provider_mismatch')
elif quality_status=='REJECT':
 reasons.append('model_quality_reject')
elif not quality_candidate:
 reasons.append('model_quality_not_candidate')

hard_quality_fail=any(x in reasons for x in {
 'provider_not_verified','model_quality_report_missing',
 'model_quality_provider_mismatch','model_quality_reject'
})
if hard_quality_fail:
 status='REJECT'
elif len(closes)<15 or len(days)<10:
 reasons.append('insufficient_live_sample')
 status='OBSERVE'
else:
 lifecycle_reasons=[]
 if realized_return<=0: lifecycle_reasons.append('realized_return_nonpositive')
 if pf is None or pf<=1.0: lifecycle_reasons.append('pf_not_above_1')
 if maxdd<=-15.0: lifecycle_reasons.append('drawdown_below_limit')
 reasons.extend(lifecycle_reasons)
 if lifecycle_reasons:
  status='REJECT'
 elif not quality_candidate:
  status='OBSERVE'
 else:
  status='CANDIDATE'
  if len(closes)>=40 and len(days)>=20 and pf>1.15 and maxdd>-15.0:
   status='PRODUCTION_ELIGIBLE'
report={
 'schema_version':'short1_v30_promotion_gate.v1','generated_ts':time.time(),'routing':'OFF','auto_promote':False,
 'expected_provider':expected,'providers':dict(providers),'provider_ok':provider_ok,'source_decisions':len(src),'opens':len(opens),'closed':len(closes),'wins':wins,'losses':losses,
 'observed_days':len(days),'close_days':len(close_days),'realized_pnl':net,'realized_return_pct':realized_return,'pf':pf,'realized_maxdd_pct':maxdd,
 'heartbeat_age_sec':hb_age,'promotion_status':status,'reasons':reasons,
 'model_quality':{
  'report_status':quality_status,'provider':quality_provider,'provider_ok':quality_provider_ok,
  'selected_n':quality_selected_n,'selected_days':quality_selected_days,'selected_lift':quality_lift,
  'reference_lifecycle_n':quality_ref_n,'reference_lifecycle_days':quality_ref_days,
  'reference_lifecycle_net_bps':quality_ref_net,'reference_lifecycle_pf':quality_ref_pf,
  'candidate_ok':quality_candidate,
 },
 'criteria':{
  'minimum_reject_sample_closed':15,'minimum_reject_days':10,'candidate_pf_gt':1.0,
  'production_closed_gte':40,'production_days_gte':20,'production_pf_gt':1.15,'maxdd_gt_pct':-15.0,
  'model_quality_status_in':['CANDIDATE','CANDIDATE_REVIEW'],'model_quality_selected_n_gte':100,
  'model_quality_selected_days_gte':10,'model_quality_selected_lift_gte':0.02,
  'model_quality_reference_lifecycle_n_gte':100,'model_quality_reference_lifecycle_days_gte':10,
  'model_quality_reference_lifecycle_net_bps_gt':0.0,'model_quality_reference_lifecycle_pf_gt':1.10,
 }
}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report,indent=2,allow_nan=False))
