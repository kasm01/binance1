#!/usr/bin/env python3
from __future__ import annotations
import bisect, json, statistics, urllib.request
from datetime import datetime
import redis

STREAM='telemetry:short1:hard_entry_quality:v1'
SYMBOL='INJUSDT'
EPISODE_GAP_SEC=900

def fetch_klines(start_ms:int,end_ms:int):
    out=[]; cur=start_ms
    while cur<=end_ms:
        url=(f'https://fapi.binance.com/fapi/v1/klines?symbol={SYMBOL}&interval=1m&'
             f'startTime={cur}&endTime={end_ms}&limit=1500')
        with urllib.request.urlopen(url,timeout=10) as resp: batch=json.load(resp)
        if not batch: break
        out.extend(batch); last=int(batch[-1][0]); cur=last+60000
        if len(batch)<1500: break
    return out

def main():
    r=redis.Redis(decode_responses=True)
    rows=[]
    for sid,f in r.xrange(STREAM,'-','+'):
        try:
            o=json.loads(f.get('json','{}')); score=float(o.get('model_score') or 0); price=float(o.get('price') or 0)
            if score<0.555 or price<=0 or not o.get('ts_utc'): continue
            ts=datetime.fromisoformat(str(o['ts_utc']).replace('Z','+00:00')).timestamp(); rows.append((ts,o))
        except Exception: pass
    if not rows:
        print('SHORT1 HARD QUALITY: no records'); return
    start=int((min(t for t,_ in rows)-120)*1000)//60000*60000
    end=int((max(t for t,_ in rows)+3700)*1000)//60000*60000
    ks=fetch_klines(start,end); opens=[int(x[0]) for x in ks]
    for _,o in rows:
        et=int(datetime.fromisoformat(str(o['ts_utc']).replace('Z','+00:00')).timestamp()*1000)
        i=bisect.bisect_left(opens,et//60000*60000+60000); ep=float(o['price'])
        for m in (15,30,60):
            j=min(i+m-1,len(ks)-1)
            if i>=len(ks) or j<i: continue
            seg=ks[i:j+1]; o[f'ret{m}']=(ep-float(ks[j][4]))/ep
            o[f'mae{m}']=max(float(x[2])/ep-1 for x in seg)
            o[f'mfe{m}']=max(1-float(x[3])/ep for x in seg)
    print(f'SHORT1 HARD ENTRY QUALITY | records={len(rows)} | threshold>=0.555')
    for g in ('STRONG','WATCH','WEAK'):
        a=[o for _,o in rows if o.get('quality_shadow_grade')==g and 'ret60' in o]
        if not a: continue
        print(f'\n{g} n={len(a)}')
        for m in (15,30,60):
            v=[o[f'ret{m}'] for o in a if f'ret{m}' in o]
            print(f' {m:02d}m win={100*sum(x>0 for x in v)/len(v):.1f}% mean={statistics.mean(v):+.5f} median={statistics.median(v):+.5f} MAE={statistics.mean(o[f"mae{m}"] for o in a):.5f} MFE={statistics.mean(o[f"mfe{m}"] for o in a):.5f}')
    acc=sorted((t,o) for t,o in rows if o.get('final_decision')=='INTENT' and 'ret60' in o)
    clusters=[]
    for item in acc:
        if not clusters or item[0]-clusters[-1][-1][0]>EPISODE_GAP_SEC: clusters.append([item])
        else: clusters[-1].append(item)
    rank={'WEAK':0,'WATCH':1,'STRONG':2}
    chosen=[max(c,key=lambda z:(rank.get(z[1].get('quality_shadow_grade'),-1),float(z[1].get('model_score') or 0)))[1] for c in clusters]
    print(f'\nEPISODES={len(clusters)} sizes={[len(c) for c in clusters]} selector=best_grade_then_score')
    for g in ('STRONG','WATCH','WEAK'):
        a=[o for o in chosen if o.get('quality_shadow_grade')==g]
        if a: print(f' {g}: n={len(a)} win60={100*sum(o["ret60"]>0 for o in a)/len(a):.1f}% mean60={statistics.mean(o["ret60"] for o in a):+.5f}')
    print('\nPOLICY=RESEARCH_ONLY; no routing/execution changes')

if __name__=='__main__': main()
