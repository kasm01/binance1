#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, time
from pathlib import Path
from collections import Counter
import redis
ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/'research/short/results'; RES.mkdir(parents=True,exist_ok=True)
OUT=RES/'SHORT1_V30_LIVE_PROMOTION_GATE.json'
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
reasons=[]; status='OBSERVE'
# Never auto-promote. Reject only after non-trivial sample; before then OBSERVE even if negative.
if not provider_ok:
 reasons.append('provider_not_verified')
elif len(closes)<15 or len(days)<10:
 reasons.append('insufficient_live_sample')
else:
 if realized_return<=0: reasons.append('realized_return_nonpositive')
 if pf is None or pf<=1.0: reasons.append('pf_not_above_1')
 if maxdd<=-15.0: reasons.append('drawdown_below_limit')
 if reasons: status='REJECT'
 else:
  status='CANDIDATE'
  if len(closes)>=40 and len(days)>=20 and pf>1.15 and maxdd>-15.0:
   status='PRODUCTION_ELIGIBLE'
report={
 'schema_version':'short1_v30_promotion_gate.v1','generated_ts':time.time(),'routing':'OFF','auto_promote':False,
 'expected_provider':expected,'providers':dict(providers),'provider_ok':provider_ok,'source_decisions':len(src),'opens':len(opens),'closed':len(closes),'wins':wins,'losses':losses,
 'observed_days':len(days),'close_days':len(close_days),'realized_pnl':net,'realized_return_pct':realized_return,'pf':pf,'realized_maxdd_pct':maxdd,
 'heartbeat_age_sec':hb_age,'promotion_status':status,'reasons':reasons,
 'criteria':{'minimum_reject_sample_closed':15,'minimum_reject_days':10,'candidate_pf_gt':1.0,'production_closed_gte':40,'production_days_gte':20,'production_pf_gt':1.15,'maxdd_gt_pct':-15.0}
}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report,indent=2,allow_nan=False))
