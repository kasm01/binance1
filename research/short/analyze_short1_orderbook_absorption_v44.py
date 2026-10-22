#!/usr/bin/env python3
from __future__ import annotations
import json, os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import redis

STREAM=os.getenv('SHORT1_ORDERBOOK_V44_OUTCOME_STREAM','telemetry:short1:inj:entry_hazard_orderbook_v44_outcomes')
DEST=Path(os.getenv('SHORT1_ORDERBOOK_V44_REPORT','research/short/results/SHORT1_ORDERBOOK_ABSORPTION_V44.json'))
ELIGIBLE={'READY','PENDING_CONTEXT'}

def f(v,d=0.0):
    try:return float(v)
    except:return d

def shift_from(o:Dict[str,Any])->Dict[str,Any]:
    z=o.get('book_shift_20_30') if isinstance(o.get('book_shift_20_30'),dict) else None
    if z and z.get('available'):
        return z
    d=o.get('orderbook_delta_20_30') or {}
    if not d.get('available'): return {'available':False}
    vals=[f(d.get('l1_delta')),f(d.get('depth5_delta')),f(d.get('depth20_delta')),f(d.get('microprice_delta_bps'))]
    bull=sum(v>0 for v in vals); bear=sum(v<0 for v in vals)
    return {'available':True,'bullish_shift_count':bull,'bearish_shift_count':bear,'net_shift_count':bull-bear,'majority_bullish':bull>=3,'all_bullish':bull==4}


def flow_price_quadrant(o:Dict[str,Any])->str:
    s=o.get('snapshot_20s') or {}
    flow=f(s.get('flow')); ret=f(s.get('return_from_open'))
    if flow>0 and ret>0:return 'BUY_FLOW_PRICE_UP'
    if flow>0 and ret<=0:return 'BUY_FLOW_PRICE_DOWN'
    if flow<0 and ret<0:return 'SELL_FLOW_PRICE_DOWN'
    if flow<0 and ret>=0:return 'SELL_FLOW_PRICE_UP'
    return 'NEUTRAL'

def book_reversal_watch(o:Dict[str,Any])->bool:
    sh=o.get('_shift') or shift_from(o)
    if not sh.get('available') or not sh.get('majority_bullish'):
        return False
    s20=o.get('snapshot_20s') or {}; s30=o.get('snapshot_30s') or {}
    r20=f(s20.get('return_from_open'),np.nan); r30=f(s30.get('return_from_open'),np.nan)
    return bool(np.isfinite(r20) and np.isfinite(r30) and r30>r20)

def event_day(o:Dict[str,Any])->str:
    try:
        ms=int(o.get('minute_ms') or o.get('decision_ts_ms') or 0)
        if not ms:return ''
        import datetime as _dt
        return _dt.datetime.fromtimestamp(ms/1000,tz=_dt.timezone.utc).date().isoformat()
    except Exception:return ''

def sample_status(n:int,days:int)->str:
    if n<20 or days<5:return 'COLLECT_MORE'
    if n<40 or days<10:return 'EXPLORATORY'
    return 'REVIEW_CANDIDATE'

def summarize(arr:List[Dict[str,Any]])->Dict[str,Any]:
    if not arr:return {'n':0}
    touch=[]; maes=[]; mfes=[]
    for o in arr:
        l=(o.get('labels_from_30s_decision') or {}).get('5m',{})
        if l.get('available'):
            touch.append(l.get('touch_40bp')); maes.append(f(l.get('mae_pct'))); mfes.append(f(l.get('mfe_pct')))
    c=Counter(touch)
    return {'n':len(arr),'labeled_5m':len(touch),'adverse_first_rate':c.get('ADVERSE_FIRST',0)/len(touch) if touch else None,
            'favorable_first_rate':c.get('FAVORABLE_FIRST',0)/len(touch) if touch else None,
            'neither_rate':c.get('NEITHER',0)/len(touch) if touch else None,
            'mae5_median_pct':float(np.median(maes)) if maes else None,'mfe5_median_pct':float(np.median(mfes)) if mfes else None,
            'touch_counts':dict(c)}

def main():
    r=redis.Redis(decode_responses=True)
    rows=[]
    for sid,fields in r.xrange(STREAM):
        try:
            o=json.loads(fields.get('json','{}')); o['_sid']=sid; o['_shift']=shift_from(o); rows.append(o)
        except Exception: pass
    model_gate_eligible=[o for o in rows if o.get('model_gate_shadow_state') in ELIGIBLE]
    actionable_states={'NORMAL','READY_AFTER_REVALIDATION','REJECT_CANDIDATE'}
    eligible=[o for o in model_gate_eligible if o.get('policy_state') in actionable_states]
    mb=[o for o in eligible if (o.get('_shift') or {}).get('majority_bullish')]
    notmb=[o for o in eligible if (o.get('_shift') or {}).get('available') and not (o.get('_shift') or {}).get('majority_bullish')]
    reversal_watch=[o for o in eligible if book_reversal_watch(o)]
    reversal_other=[o for o in eligible if not book_reversal_watch(o)]
    recent=[]
    for o in rows[-12:]:
        l=(o.get('labels_from_30s_decision') or {}).get('5m',{})
        a=o.get('absorption_features_20s') or {}
        sh=o.get('_shift') or {}
        recent.append({'source_event_id':o.get('source_event_id'),'state':o.get('policy_state'),'gate':o.get('model_gate_shadow_state'),'score':f(o.get('score')),
            'flow20':f((o.get('snapshot_20s') or {}).get('flow')),'ret20':f((o.get('snapshot_20s') or {}).get('return_from_open')),
            'd5':a.get('depth5_support'),'d20':a.get('depth20_support'),'micro':a.get('microprice_support_bps'),
            'sell_pressure':a.get('sell_pressure'),'eff':a.get('down_bps_per_sell_flow'),
            'flow_price_quadrant':flow_price_quadrant(o),'bullish_shift_count':sh.get('bullish_shift_count'),'majority_bullish':sh.get('majority_bullish'),
            'book_reversal_watch':book_reversal_watch(o),
            'trace20_30_bullish_persistence':(o.get('orderbook_trace_20_30') or {}).get('bullish_persistence'),
            'trace20_30_depth5_positive_frac':(o.get('orderbook_trace_20_30') or {}).get('depth5_positive_frac'),
            'trace20_30_micro_positive_frac':(o.get('orderbook_trace_20_30') or {}).get('micro_positive_frac'),
            'trace20_30_depth5_sign_flips':(o.get('orderbook_trace_20_30') or {}).get('depth5_sign_flips'),
            'touch5':l.get('touch_40bp')})
    quadrant_counts=dict(Counter(flow_price_quadrant(o) for o in eligible))
    quadrant_stats={q:summarize([o for o in eligible if flow_price_quadrant(o)==q]) for q in quadrant_counts}
    trace_eligible=[o for o in eligible if (o.get('orderbook_trace_20_30') or {}).get('available')]
    trace_tertiles={}
    if len(trace_eligible)>=15:
        for key in ['bullish_persistence','depth5_positive_frac','depth20_positive_frac','l1_positive_frac','micro_positive_frac']:
            vals=np.array([f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan) for o in trace_eligible],dtype=float)
            good=np.isfinite(vals)
            if good.sum()<12:continue
            q33,q67=np.quantile(vals[good],[1/3,2/3])
            low=[o for o in trace_eligible if np.isfinite(f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)) and f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)<=q33]
            high=[o for o in trace_eligible if np.isfinite(f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)) and f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)>=q67]
            mid=[o for o in trace_eligible if np.isfinite(f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)) and q33<f((o.get('orderbook_trace_20_30') or {}).get(key),np.nan)<q67]
            trace_tertiles[key]={'q33':float(q33),'q67':float(q67),'low':summarize(low),'mid':summarize(mid),'high':summarize(high)}
    actionable_days=len({event_day(o) for o in eligible if event_day(o)})
    out={'stream':STREAM,'n':len(rows),'model_gate_eligible_n':len(model_gate_eligible),'actionable_entry_n':len(eligible),'actionable_days':actionable_days,
         'sample_status':sample_status(len(eligible),actionable_days),
         'sample_gate':{'collect_more_below_n':20,'exploratory_below_n':40,'min_days_for_review':10},
         'trace_actionable_n':len(trace_eligible),'trace_tertiles':trace_tertiles,
         'policy_states':dict(Counter(o.get('policy_state') for o in rows)),'actionable_quadrants':quadrant_counts,'actionable_quadrant_stats':quadrant_stats,
         'eligible_summary':summarize(eligible),'eligible_majority_bullish':summarize(mb),'eligible_not_majority_bullish':summarize(notmb),
         'book_reversal_watch_n':len(reversal_watch),'book_reversal_watch_summary':summarize(reversal_watch),'book_reversal_other_summary':summarize(reversal_other),
         'recent':recent,'feature_quantiles':{}}
    # quantiles only when enough labeled rows
    if len(eligible)>=15:
        keys=['depth5_support','depth20_support','l1_support','microprice_support_bps','bid5_replenishment_ratio','ask5_replenishment_ratio','depth5_delta','depth20_delta','microprice_delta_bps']
        trace_keys=['bullish_persistence','depth5_positive_frac','depth20_positive_frac','l1_positive_frac','micro_positive_frac','depth5_sign_flips','micro_sign_flips']
        fq={}
        for k in keys:
            vals=[]
            for o in eligible:
                a=o.get('absorption_features_20s') or {}; v=a.get(k)
                if v is not None: vals.append(f(v))
            if len(vals)>=9:
                q=np.quantile(vals,[1/3,2/3]).tolist(); fq[k]={'q33':q[0],'q67':q[1],'n':len(vals)}
        for k in trace_keys:
            vals=[]
            for o in eligible:
                a=o.get('orderbook_trace_20_30') or {}; v=a.get(k)
                if v is not None: vals.append(f(v))
            if len(vals)>=9:
                q=np.quantile(vals,[1/3,2/3]).tolist(); fq['trace20_30_'+k]={'q33':q[0],'q67':q[1],'n':len(vals)}
        out['feature_quantiles']=fq
    DEST.parent.mkdir(parents=True,exist_ok=True); DEST.write_text(json.dumps(out,indent=2,sort_keys=True))
    print(json.dumps(out,indent=2,sort_keys=False)); print('SAVED',DEST)
if __name__=='__main__':main()
