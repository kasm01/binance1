#!/usr/bin/env python3
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd, requests
ROOT=Path('/home/kasm920/binance1')
CANDS=ROOT/'research/short/results/SHORT1_V47_ALL_CONFIRM_CANDIDATES.csv'
CACHE=ROOT/'research/short/data/v47_candidate_micro_daily'
OUT=ROOT/'research/short/results/SHORT1_V47_CANDIDATE_MICRO.csv'
META=ROOT/'research/short/results/SHORT1_V47_CANDIDATE_MICRO_META.json'
CACHE.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/futures/um/daily/aggTrades/INJUSDT'
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'binance1-v47-research'})
CUTS=(5,20,30)

def maker_bool(s):
    if s.dtype==bool:return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(['true','1','t','yes']).to_numpy(bool)

def day_url(day): return f'{BASE}/INJUSDT-aggTrades-{day}.zip'
def read_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        with z.open(z.namelist()[0]) as f:d=pd.read_csv(f)
    d=d.rename(columns={c:c.strip().lower() for c in d.columns})
    t=pd.to_numeric(d['transact_time'],errors='coerce').astype('Int64')
    if float(t.dropna().median())>1e14:t=(t//1000).astype('Int64')
    d['transact_time_ms']=t
    return d

def aggregate_day(day, mins):
    cache=CACHE/f'{day}.csv.gz'
    if cache.exists(): return pd.read_csv(cache)
    r=SESSION.get(day_url(day),timeout=40)
    if r.status_code!=200:
        print('[WARN]',day,'status',r.status_code,flush=True); pd.DataFrame().to_csv(cache,index=False,compression='gzip'); return pd.DataFrame()
    d=read_zip(r.content)
    minute=(pd.to_numeric(d.transact_time_ms,errors='coerce')//60000)*60000
    wanted=set(int(x) for x in mins)
    z=d[minute.isin(wanted) & (pd.to_numeric(d.transact_time_ms,errors='coerce') < minute+30000)].copy()
    if z.empty:
        pd.DataFrame().to_csv(cache,index=False,compression='gzip'); return pd.DataFrame()
    z['_minute_ms']=((pd.to_numeric(z.transact_time_ms,errors='coerce')//60000)*60000).astype(np.int64)
    rows=[]
    for ms,g in z.groupby('_minute_ms',sort=False):
        g=g.sort_values('transact_time_ms'); t=pd.to_numeric(g.transact_time_ms,errors='coerce').to_numpy(np.int64)
        p=pd.to_numeric(g.price,errors='coerce').to_numpy(float); q=pd.to_numeric(g.quantity,errors='coerce').to_numpy(float)
        mk=maker_bool(g.is_buyer_maker); quote=p*q; signed=np.where(mk,-quote,quote)
        o={'minute_ms':int(ms)}
        # opening price of minute = first trade observed; close enough to real candle open, but replay ret uses 1m open separately
        for sec in CUTS:
            m=t<int(ms+sec*1000)
            if not m.any():
                o[f'flow_{sec}s']=np.nan; o[f'last_{sec}s']=np.nan; o[f'quote_{sec}s']=np.nan; o[f'trades_{sec}s']=0
            else:
                den=float(quote[m].sum()); o[f'flow_{sec}s']=float(signed[m].sum()/den) if den>0 else np.nan
                o[f'last_{sec}s']=float(p[m][-1]); o[f'quote_{sec}s']=den; o[f'trades_{sec}s']=int(m.sum())
        rows.append(o)
    out=pd.DataFrame(rows); out.to_csv(cache,index=False,compression='gzip'); return out

def main():
    c=pd.read_csv(CANDS); c['confirm_ts']=pd.to_datetime(c.confirm_ts,utc=True); c['day']=c.confirm_ts.dt.strftime('%Y-%m-%d'); c['minute_ms']=(c.confirm_ts.astype('int64')//1_000_000).astype(np.int64)
    days=sorted(c.day.unique()); parts=[]; missing=[]
    print('[V47MICRO] candidates',len(c),'days',len(days),flush=True)
    for i,day in enumerate(days,1):
        x=c[c.day==day]; a=aggregate_day(day,x.minute_ms.to_numpy(np.int64))
        if a.empty:missing.append(day)
        else:parts.append(a)
        if i%10==0 or i==len(days):
            print(f'[V47MICRO] {i}/{len(days)} rows={sum(len(x) for x in parts)} missing={len(missing)} cache_mb={sum(p.stat().st_size for p in CACHE.glob("*.gz"))/1024/1024:.2f}',flush=True)
    m=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    out=c.merge(m,on='minute_ms',how='left').drop(columns=['day'])
    # returns use authoritative 1m open from historical kline
    one=[]
    for f in sorted((ROOT/'research/short/data/inj_1m').glob('*.csv')):
        q=pd.read_csv(f,usecols=['open_time','open']); one.append(q)
    k=pd.concat(one,ignore_index=True); k['minute_ms']=pd.to_numeric(k.open_time,errors='coerce').astype('Int64'); k['open']=pd.to_numeric(k.open,errors='coerce'); k=k.drop_duplicates('minute_ms').set_index('minute_ms')['open']
    out['candle_open']=out.minute_ms.map(k)
    for sec in CUTS: out[f'ret_{sec}s']=pd.to_numeric(out[f'last_{sec}s'],errors='coerce')/pd.to_numeric(out.candle_open,errors='coerce')-1
    out['flow_delta_5_20']=pd.to_numeric(out.flow_20s,errors='coerce')-pd.to_numeric(out.flow_5s,errors='coerce')
    out['ret_delta_20_30']=pd.to_numeric(out.ret_30s,errors='coerce')-pd.to_numeric(out.ret_20s,errors='coerce')
    out.to_csv(OUT,index=False)
    meta={'candidates':len(c),'rows':len(out),'days':len(days),'missing_days':missing,'coverage20':int(out.flow_20s.notna().sum()),'coverage30':int(out.flow_30s.notna().sum()),'cache_bytes':sum(p.stat().st_size for p in CACHE.glob('*.gz'))}
    META.write_text(json.dumps(meta,indent=2)); print('[V47MICRO] SAVED',OUT,'coverage20',meta['coverage20'],'missing',len(missing),'cache_mb',round(meta['cache_bytes']/1024/1024,2),flush=True)
if __name__=='__main__':main()
