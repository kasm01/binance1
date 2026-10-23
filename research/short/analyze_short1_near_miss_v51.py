#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
import requests, redis

ROOT=Path(__file__).resolve().parents[2]
OUTDIR=ROOT/'research/short/results/SHORT1_NEAR_MISS_V51'
OUTDIR.mkdir(parents=True,exist_ok=True)
CACHE=OUTDIR/'INJUSDT_1m_shadow_window.csv'
STREAM='telemetry:short1:v30:lifecycle_shadow_v5'
SRC_STREAM='telemetry:short1:inj:model_shadow_v5'
SYMBOL='INJUSDT'
LOW,HIGH=0.52,0.54
LEV=3.0
TP_ROI=.04; SOFT_STOP_ROI=-.12; CATA_PRICE=.06
TP_FAV=TP_ROI/LEV
SOFT_ADV=abs(SOFT_STOP_ROI)/LEV
FEE_RT=.0008
HORIZONS=(30,60,120,360)

def q(vals,p):
    vals=sorted(float(x) for x in vals if x is not None and math.isfinite(float(x)))
    if not vals:return None
    i=(len(vals)-1)*p; lo=int(i); hi=min(lo+1,len(vals)-1); w=i-lo
    return vals[lo]*(1-w)+vals[hi]*w

def load_decisions(r):
    rows=r.xrange(STREAM,'-','+')
    out=[]
    for sid,f in rows:
        try:o=json.loads((f or {}).get('json') or '{}')
        except:continue
        if o.get('kind')!='SOURCE_DECISION':continue
        s=float(o.get('score') or 0)
        if not (LOW <= s < HIGH):continue
        reason=str(o.get('reason') or '')
        if reason in {'position_exists','stop_reentry_cooldown'}:continue
        srcid=str(o.get('source_id') or '')
        if not srcid:continue
        srcrows=r.xrange(SRC_STREAM,min=srcid,max=srcid,count=1)
        if not srcrows:continue
        try:src=json.loads((srcrows[0][1] or {}).get('json') or '{}')
        except:continue
        c=src.get('confirmation') or {}; ctx=src.get('entry_market_context') or {}
        entry=float(c.get('curr_close') or 0)
        tms=int(float(c.get('current_open_time') or src.get('completed_close_time') or 0))
        if entry<=0 or tms<=0:continue
        out.append({
          'source_id':srcid,'ts_ms':tms,'entry':entry,'score':s,'reason':reason,
          'confirmation':str(o.get('confirmation_state') or src.get('confirmation_state') or ''),
          'flow60':o.get('flow60'),'regime':str(o.get('regime') or ''),'build_direction':str(o.get('build_direction') or ''),
          'v50_triggered':bool((o.get('v50_context') or {}).get('triggered',False)),
          'v50_mult':float((o.get('v50_context') or {}).get('multiplier') or 1.0),
          'ctx_available':bool(o.get('context_available',False)),
          'flow15':((ctx.get('money_flow') or {}).get('flow_15m')),
          'oi15':((ctx.get('open_interest') or {}).get('oi_delta_15m')),
        })
    return out

def fetch_klines(start_ms,end_ms):
    sess=requests.Session(); cur=start_ms; rows=[]
    while cur<=end_ms:
        resp=sess.get('https://fapi.binance.com/fapi/v1/klines',params={'symbol':SYMBOL,'interval':'1m','startTime':cur,'endTime':end_ms,'limit':1500},timeout=10)
        resp.raise_for_status(); batch=resp.json()
        if not batch:break
        rows.extend(batch); nxt=int(batch[-1][0])+60000
        if nxt<=cur:break
        cur=nxt
        if len(batch)<1500:break
        time.sleep(.05)
    ded={int(x[0]):x for x in rows}
    out=[]
    for t,x in sorted(ded.items()):
        out.append({'open_time':t,'open':float(x[1]),'high':float(x[2]),'low':float(x[3]),'close':float(x[4])})
    with CACHE.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=out[0].keys() if out else ['open_time','open','high','low','close']);w.writeheader();w.writerows(out)
    return out

def load_cache():
    if not CACHE.exists():return []
    with CACHE.open() as f:return [{**r,'open_time':int(r['open_time']),'open':float(r['open']),'high':float(r['high']),'low':float(r['low']),'close':float(r['close'])} for r in csv.DictReader(f)]

def main():
    r=redis.Redis(decode_responses=True)
    ev=load_decisions(r)
    if not ev:raise SystemExit('no near miss decisions')
    start=min(x['ts_ms'] for x in ev)-60000
    desired=max(x['ts_ms'] for x in ev)+max(HORIZONS)*60000
    now=int(time.time()*1000)//60000*60000
    end=min(desired,now-60000)
    bars=load_cache()
    if not bars or bars[0]['open_time']>start or bars[-1]['open_time']<end-120000:
        bars=fetch_klines(start,end)
    bmap={b['open_time']:b for b in bars}
    last_bar=max(bmap) if bmap else 0
    analyzed=[]
    for x in ev:
        e=x['entry']; t0=x['ts_ms']//60000*60000
        row=dict(x)
        max_end=min(t0+max(HORIZONS)*60000,last_bar)
        win=[bmap[t] for t in range(t0,max_end+1,60000) if t in bmap]
        if not win:continue
        row['available_min']=int((win[-1]['open_time']-t0)/60000)+1
        row['mfe_360']=max((e-b['low'])/e for b in win)
        row['mae_360']=max((b['high']-e)/e for b in win)
        # first barrier touch: TP arm vs soft stop vs catastrophic; ambiguous if TP+stop same bar
        tp=e*(1-TP_FAV); soft=e*(1+SOFT_ADV); cata=e*(1+CATA_PRICE)
        outcome='NONE'; first_min=None
        for b in win:
            hit_tp=b['low']<=tp; hit_soft=b['high']>=soft; hit_cata=b['high']>=cata
            if hit_tp and (hit_soft or hit_cata):outcome='AMBIGUOUS'; first_min=int((b['open_time']-t0)/60000);break
            if hit_cata:outcome='CATA_FIRST';first_min=int((b['open_time']-t0)/60000);break
            if hit_soft:outcome='SOFT_FIRST';first_min=int((b['open_time']-t0)/60000);break
            if hit_tp:outcome='TP_ARM_FIRST';first_min=int((b['open_time']-t0)/60000);break
        row['first_barrier']=outcome;row['first_barrier_min']=first_min
        for h in HORIZONS:
            target=t0+(h-1)*60000
            if target<=last_bar and target in bmap:
                c=bmap[target]['close']; ret=(e-c)/e
                row[f'ret_{h}m']=ret;row[f'net_ret_{h}m']=ret-FEE_RT
            else:
                row[f'ret_{h}m']=None;row[f'net_ret_{h}m']=None
        analyzed.append(row)
    fields=sorted({k for x in analyzed for k in x})
    with (OUTDIR/'near_miss_events.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(analyzed)
    def summarize(name,rows):
        d={'group':name,'n':len(rows),'reasons':dict(Counter(x['reason'] for x in rows)),'confirmations':dict(Counter(x['confirmation'] for x in rows)),'barriers':dict(Counter(x['first_barrier'] for x in rows))}
        for key in ['score','mfe_360','mae_360']+[f'ret_{h}m' for h in HORIZONS]+[f'net_ret_{h}m' for h in HORIZONS]:
            vals=[x.get(key) for x in rows if x.get(key) is not None]
            d[key]={'n':len(vals),'mean':mean(vals) if vals else None,'median':median(vals) if vals else None,'q25':q(vals,.25),'q75':q(vals,.75),'positive_pct':(sum(v>0 for v in vals)/len(vals) if vals else None)}
        return d
    groups={
      'ALL_052_054':analyzed,
      'LOW_052_053':[x for x in analyzed if x['score']<.53],
      'SOFT_053_054':[x for x in analyzed if x['score']>=.53],
      'CONFIRM':[x for x in analyzed if x['confirmation']=='CONFIRM'],
      'NOT_CONFIRM':[x for x in analyzed if x['confirmation']!='CONFIRM'],
      'CONFIRMED_SHORT_BUILD':[x for x in analyzed if x['confirmation']=='CONFIRM' and x['regime']=='confirmed_build' and x['build_direction']=='short'],
    }
    report={
      'schema':'short1_near_miss_v51','generated_ts':time.time(),'window':{'first_ms':min(x['ts_ms'] for x in analyzed),'last_ms':max(x['ts_ms'] for x in analyzed),'last_price_bar_ms':last_bar},
      'policy':{'score_band':[LOW,HIGH],'leverage':LEV,'tp_arm_roi':TP_ROI,'tp_favorable_price_pct':TP_FAV,'soft_stop_roi':SOFT_STOP_ROI,'soft_adverse_price_pct':SOFT_ADV,'catastrophic_adverse_price_pct':CATA_PRICE,'fee_roundtrip':FEE_RT},
      'groups':[summarize(k,v) for k,v in groups.items()]
    }
    (OUTDIR/'summary.json').write_text(json.dumps(report,indent=2))
    print('near_miss_n',len(analyzed),'price_bars',len(bars),'cache',CACHE)
    for g in report['groups']:
        print('\nGROUP',g['group'],'n',g['n'])
        print(' reasons',g['reasons']);print(' confirms',g['confirmations']);print(' barriers',g['barriers'])
        for h in HORIZONS:
            z=g[f'net_ret_{h}m']; print(f' net{h}m n={z["n"]} mean={z["mean"]} median={z["median"]} positive={z["positive_pct"]}')
        print(' mfe360',g['mfe_360']);print(' mae360',g['mae_360'])
if __name__=='__main__':main()
