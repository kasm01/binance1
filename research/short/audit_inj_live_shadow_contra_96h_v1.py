#!/usr/bin/env python3
"""SHORT1 nonrouting 96h shadow regime study, sourced from actual V6/5 events.

Closed 1m candles ONLY for pre-signal trend; post-signal markouts separated.
Decision time actual V6 emission timestamp, entry at next 1m OPEN, no claim of
native model/Master approval. Read-only Redis, public Binance GET only when
fixed-window file absent, report writes under research only.
"""
import json,csv,time,bisect
from datetime import datetime,timezone,timedelta
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import redis,requests

ROOT=Path('research/short/data')
REPORT=Path('research/results')
START=datetime(2026,11,10,0,0,tzinfo=timezone.utc)
END=datetime(2026,11,14,10,15,tzinfo=timezone.utc)
START_MS=int(START.timestamp()*1000);END_MS=int(END.timestamp()*1000)
FILE=ROOT/'INJUSDT_1m_20261110_20261114_live_contra_audit.csv'
if not FILE.exists():
    sess=requests.Session();data=[];cursor=START_MS
    while cursor<END_MS:
        response=sess.get('https://fapi.binance.com/fapi/v1/klines',
           params={'symbol':'INJUSDT','interval':'1m','startTime':cursor,'endTime':END_MS-1,'limit':1500},
           timeout=16)
        response.raise_for_status()
        payload=response.json()
        if not isinstance(payload,list) or not payload:raise RuntimeError('missing_1m_kline_history')
        data.extend(payload)
        nxt=int(payload[-1][0])+60000
        if nxt<=cursor:raise RuntimeError('no_kline_progress')
        cursor=nxt
        print('PAGE_RECEIVED',len(payload),'CURSOR',datetime.fromtimestamp(cursor/1000,timezone.utc),flush=True)
        time.sleep(.2)
    FILE.parent.mkdir(parents=True,exist_ok=True)
    with FILE.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['open_time','open','high','low','close','close_time','quote_volume','taker_buy_quote_volume'])
        for k in data:w.writerow([k[0],k[1],k[2],k[3],k[4],k[6],k[7],k[10]])
else: print('REUSE_PUBLIC_CANDLE_FILE',FILE)
m=pd.read_csv(FILE)
for col in m.columns:m[col]=pd.to_numeric(m[col],errors='raise')
assert len(m)>1000
assert (m.open_time.diff().dropna()==60000).all(),'1m candle discontinuity'
assert int(m.open_time.iloc[0])<=START_MS and int(m.open_time.iloc[-1])<END_MS
c=m.close_time.to_numpy('int64');o=m.open_time.to_numpy('int64')
v6={};v5={}
r=redis.Redis(decode_responses=True,socket_timeout=3)
for version,dest in [('v6',v6),('v5',v5)]:
    stream='telemetry:short1:inj:model_shadow_'+version
    for sid,raw in r.xrevrange(stream,count=2600):
        if int(sid.split('-')[0])<START_MS:break
        try:z=json.loads(raw['json']);ms=int(z['completed_close_time'])
        except Exception:continue
        if START_MS<=ms<END_MS and ms not in dest and z.get('routing')=='OFF':
            dest[ms]=z
common=sorted(set(v6)&set(v5))
def past_context(event_ms):
    i=bisect.bisect_left(c,event_ms)-1
    if i<65:return None
    close=float(m.close.iloc[i])
    ret15=100*(close/float(m.close.iloc[i-15])-1)
    ret60=100*(close/float(m.close.iloc[i-60])-1)
    # Causal fully-closed last 1m, aggregate recent 3m & 5m via close-to-close.
    ret1=100*(close/float(m.close.iloc[i-1])-1)
    ret3=100*(close/float(m.close.iloc[i-3])-1)
    ret5=100*(close/float(m.close.iloc[i-5])-1)
    window=m.iloc[i-14:i+1]
    q=float(window.quote_volume.sum())
    flow15=2*float(window.taker_buy_quote_volume.sum())/q-1 if q>0 else np.nan
    return dict(prev_closed_1m_ms=int(c[i]),ret1_pct=ret1,ret3_pct=ret3,ret5_pct=ret5,
      ret15_pct=ret15,ret60_pct=ret60,flow15=flow15,
      caution_uptrend=bool(ret15>0 and ret60>0),
      selling_reversal=bool(ret1<0 and ret3<0 and ret5<0 and flow15<0))
results=[]
for bar in common:
    e=v6[bar];f=v5[bar]
    ts=int(float(e['ts'])*1000)
    if not START_MS<=ts<END_MS:continue
    if f.get('model_feature_hash')!=e.get('model_feature_hash'):continue
    try:
        ctx=past_context(ts)
        if ctx is None:continue
        score=float(e['economic_probability']);threshold=float(e['reference_threshold'])
        valid=(e['model_provider']=='short_arm_quality_logistic_v6' and
            e['symbol']=='INJUSDT' and e['interval']=='3m' and bool(e['event_id']) and
            int(e['completed_close_time'])==int(f['completed_close_time']))
        if not valid:continue
        gate=score>=threshold
        progress=float(e.get('sniper_progress') or -1)
        confirm=(e.get('confirmation_state')=='CONFIRM'
          and float(e.get('sniper_min') or 0)<=progress<=float(e.get('sniper_max') or 0))
        # next minute open AFTER emission timestamp to avoid measuring past fills
        entry_idx=bisect.bisect_right(o,ts)
        entry_idx=entry_idx if entry_idx<len(m) and int(o[entry_idx])-ts<=60000 else None
        row=dict(event_ms=ts,source_bar_ms=bar,event_id=e['event_id'],
          score=score,threshold=threshold,score_pass=bool(gate),
          confirmation=bool(confirm),old_shadow_ready=bool(gate and confirm),
          v5_score=float(f['economic_probability']),v5_pass=float(f['economic_probability'])>=float(f['reference_threshold']),
          **ctx)
        if entry_idx is not None:
            entry=float(m.open.iloc[entry_idx]);row['entry_ms']=int(o[entry_idx]);row['entry_price']=entry
            for horizon in (15,30,60):
                j=entry_idx+horizon-1
                if j>=len(m):continue
                # SHORT signed return from entry to closing 1m candle AFTER horizon.
                px=float(m.close.iloc[j])
                row[f'short_price_{horizon}m_pct']=100*(entry-px)/entry
                row[f'net_roi3x_{horizon}m_assumed_pct']=3*(100*(entry-px)/entry-.14)
        results.append(row)
    except (ValueError,TypeError,KeyError,IndexError):continue
df=pd.DataFrame(results).sort_values('event_ms')
if df.empty:raise RuntimeError('no_completed_shadow_events')
REPORT.mkdir(exist_ok=True,parents=True)
csv_path=REPORT/'SHORT1_INJ_V6_LIVE_CONTRA_96H_EVENTS_20261114.csv'
df.to_csv(csv_path,index=False)
summary={'time_window_utc':f'{START.isoformat()} — {END.isoformat()}',
 'bars_1m':len(m),'v6_events':len(v6),'paired_v5v6':len(common),
 'valid_event_count':len(df),'score_pass':int(df.score_pass.sum()),
 'model_score_and_confirmation':int(df.old_shadow_ready.sum()),
 'caution_uptrend_all':int(df.caution_uptrend.sum()),
 'selling_reversal_all':int(df.selling_reversal.sum()),
 'groups':{}}
for name,group in [('all',df),('score_pass',df[df.score_pass]),('shadow_ready',df[df.old_shadow_ready])]:
    parts={}
    for k,sub in [('uptrend',group[group.caution_uptrend]),('not_uptrend',group[~group.caution_uptrend]),
                  ('uptrend_reversal',group[group.caution_uptrend&group.selling_reversal]),
                  ('uptrend_no_reversal',group[group.caution_uptrend&~group.selling_reversal])]:
        parts[k]={'n':len(sub),'score_pass':int(sub.score_pass.sum()),'ready':int(sub.old_shadow_ready.sum())}
        for h in (15,30,60):
            vals=sub[f'net_roi3x_{h}m_assumed_pct'].dropna()
            parts[k][f'net_roi3x_{h}m_mean_pct']=round(float(vals.mean()),4) if len(vals) else None
            parts[k][f'priced_n_{h}m']=len(vals)
            parts[k][f'priced_positive_{h}m']=int((vals>0).sum())
    summary['groups'][name]=parts
REPORT.joinpath('SHORT1_INJ_V6_LIVE_CONTRA_96H_SUMMARY_20261114.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('FILES',csv_path,FILE)
print('CAUTION: all results are source shadow observations, NOT executable orders or current V7 model profits.')
