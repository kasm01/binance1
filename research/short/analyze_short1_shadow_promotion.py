#!/usr/bin/env python3
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
from typing import Any, Dict, List

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np, pandas as pd
from orchestration.event_bus.redis_bus import RedisBus

SRC=os.getenv('SHORT1_SHADOW_STREAM') or os.getenv('SHORT_SHADOW_STREAM') or 'telemetry:short1:inj:model_shadow'
OUTCOME=os.getenv('SHORT1_SHADOW_OUTCOME_STREAM') or os.getenv('SHORT_SHADOW_OUTCOME_STREAM') or 'telemetry:short1:inj:model_shadow_outcomes'
RES=ROOT/'research/short/results'; RES.mkdir(parents=True,exist_ok=True)
REPORT=RES/'SHORT1_INJ_SHADOW_PROMOTION.json'
BUCKETS=RES/'SHORT1_INJ_SHADOW_SCORE_BUCKETS.csv'
DAILY=RES/'SHORT1_INJ_SHADOW_DAILY.csv'


def read_stream(bus: RedisBus, stream: str, count: int=20000)->pd.DataFrame:
    rows=[]
    for sid,fields in reversed(bus.r.xrevrange(stream,max='+',min='-',count=count)):
        try:
            d=json.loads((fields or {}).get('json') or '{}')
            if isinstance(d,dict): d['_stream_id']=sid; rows.append(d)
        except Exception: pass
    return pd.DataFrame(rows)


def metrics(r)->Dict[str,Any]:
    x=np.asarray(pd.to_numeric(pd.Series(r),errors='coerce').dropna(),dtype=float)
    if len(x)==0: return {'n':0,'net_bps':None,'win':None,'pf':None,'ci95_lo_bps':None,'ci95_hi_bps':None}
    pos=x[x>0].sum(); neg=-x[x<0].sum(); mean=float(x.mean())
    if len(x)>=2:
        se=float(x.std(ddof=1)/math.sqrt(len(x))); lo=(mean-1.96*se)*1e4; hi=(mean+1.96*se)*1e4
    else: lo=hi=None
    return {'n':int(len(x)),'net_bps':mean*1e4,'win':float((x>0).mean()),'pf':(float(pos/neg) if neg>0 else None),'ci95_lo_bps':lo,'ci95_hi_bps':hi}


def classify(confirm_m: Dict[str,Any], days: int, positive_days: int)->str:
    n=int(confirm_m.get('n') or 0); net=confirm_m.get('net_bps'); pf=confirm_m.get('pf'); ci=confirm_m.get('ci95_lo_bps')
    if n<30: return 'OBSERVE'
    if net is None or net<=0: return 'REJECT'
    if n<100: return 'PRELIMINARY'
    if days<10 or positive_days<max(3,math.ceil(days*0.6)): return 'PRELIMINARY'
    if pf is None or pf<=1.10: return 'PRELIMINARY'
    if ci is None or ci<=0: return 'PRELIMINARY'
    if n<300 or days<30: return 'CANDIDATE'
    return 'PRODUCTION_ELIGIBLE'


def classify_entry_quality(n: int, hit: float | None, base_hit: float | None, days: int) -> str:
    if n < 30: return 'OBSERVE'
    if hit is None or base_hit is None or hit <= base_hit: return 'REJECT'
    lift = hit - base_hit
    if n < 100 or days < 10: return 'PRELIMINARY'
    if lift < 0.02: return 'PRELIMINARY'
    if n < 300 or days < 30: return 'CANDIDATE'
    # Deliberately never auto-promote from entry-quality alone.
    return 'CANDIDATE_REVIEW'

def main():
    bus=RedisBus()
    if not bus.ping(): raise SystemExit('Redis unavailable')
    s=read_stream(bus,SRC); o=read_stream(bus,OUTCOME)
    report={'bot_id':'SHORT1','symbol':'INJUSDT','routing':'OFF','shadow_events':int(len(s)),'outcomes':int(len(o)),'promotion_status':'OBSERVE'}
    if len(s):
        s['ts_dt']=pd.to_datetime(pd.to_numeric(s['ts'],errors='coerce'),unit='s',utc=True)
        report['shadow_first_ts']=str(s.ts_dt.min()); report['shadow_last_ts']=str(s.ts_dt.max())
        report['confirmation_counts']={str(k):int(v) for k,v in s['confirmation_state'].value_counts(dropna=False).items()}
        p=pd.to_numeric(s['economic_probability'],errors='coerce')
        report['score_summary']={'mean':float(p.mean()),'median':float(p.median()),'min':float(p.min()),'max':float(p.max()),'reference_threshold_mean':float(pd.to_numeric(s['reference_threshold'],errors='coerce').mean())}
    if len(o)==0:
        REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2)); return

    o['source_dt']=pd.to_datetime(pd.to_numeric(o['source_ts'],errors='coerce'),unit='s',utc=True)
    o['date']=o.source_dt.dt.date.astype(str)
    o['net_return_pct']=pd.to_numeric(o['net_return_pct'],errors='coerce')
    o['economic_probability']=pd.to_numeric(o['economic_probability'],errors='coerce')
    if 'model_target' in o.columns and len(o):
        report['model_target']=str(o['model_target'].dropna().iloc[-1]) if len(o['model_target'].dropna()) else ''
    if 'model_provider' in o.columns and len(o):
        report['model_provider']=str(o['model_provider'].dropna().iloc[-1]) if len(o['model_provider'].dropna()) else ''
    allm=metrics(o.net_return_pct); report['all_outcomes']=allm
    by_state={}
    for st,g in o.groupby('confirmation_state'): by_state[str(st)]=metrics(g.net_return_pct)
    report['by_confirmation_state']=by_state
    c=o[o.confirmation_state=='CONFIRM'].copy(); cm=metrics(c.net_return_pct); report['confirm']=cm

    daily=[]
    for day,g in c.groupby('date'):
        m=metrics(g.net_return_pct); daily.append({'date':day,**m})
    ddf=pd.DataFrame(daily); ddf.to_csv(DAILY,index=False)
    days=int(len(ddf)); posdays=int((pd.to_numeric(ddf.get('net_bps',pd.Series(dtype=float)),errors='coerce')>0).sum()) if days else 0
    report['confirm_days']=days; report['confirm_positive_days']=posdays

    if len(c)>=5:
        try:
            c['score_bucket']=pd.qcut(c.economic_probability,q=min(5,max(2,len(c)//5)),duplicates='drop')
            br=[]
            for b,g in c.groupby('score_bucket',observed=True): br.append({'bucket':str(b),'score_min':float(g.economic_probability.min()),'score_max':float(g.economic_probability.max()),**metrics(g.net_return_pct)})
            pd.DataFrame(br).to_csv(BUCKETS,index=False); report['score_buckets']=br
        except Exception: pass

    thresholds=[]
    if len(c):
        ref=float(pd.to_numeric(c.reference_threshold,errors='coerce').median())
        for th in sorted(set([ref,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90])):
            g=c[c.economic_probability>=th]; thresholds.append({'threshold':float(th),**metrics(g.net_return_pct)})
    report['thresholds']=thresholds

    quality_status = None
    if 'entry_quality_hit' in o.columns:
        q = c.copy()
        q['entry_quality_hit_num'] = q['entry_quality_hit'].map(lambda v: 1.0 if str(v).lower() in ('true','1','1.0') else (0.0 if str(v).lower() in ('false','0','0.0') else np.nan))
        q = q[q.entry_quality_hit_num.notna()].copy()
        if len(q):
            base_hit = float(q.entry_quality_hit_num.mean())
            selected = q[q.economic_probability >= pd.to_numeric(q.reference_threshold,errors='coerce')].copy()
            sel_hit = float(selected.entry_quality_hit_num.mean()) if len(selected) else None
            sel_days = int(selected['date'].nunique()) if len(selected) else 0
            report['entry_quality'] = {
                'confirm_n': int(len(q)),
                'confirm_hit_rate': base_hit,
                'selected_n': int(len(selected)),
                'selected_hit_rate': sel_hit,
                'selected_lift': (float(sel_hit-base_hit) if sel_hit is not None else None),
                'selected_days': sel_days,
                'observation_min': int(pd.to_numeric(q.get('entry_quality_observation_min'),errors='coerce').median()) if 'entry_quality_observation_min' in q.columns else 60,
            }
            quality_status = classify_entry_quality(int(len(selected)), sel_hit, base_hit, sel_days)
    report['promotion_status']=quality_status or classify(cm,days,posdays)
    report['auto_promote']=False
    REPORT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__': main()
