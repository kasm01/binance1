#!/usr/bin/env python3
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'
CACHE=ROOT/'research/short/data/bookdepth_entry_cache_v45b'
OUT=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V45B'
CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/futures/um/daily/bookDepth/INJUSDT'
BANDS=(0.2,1.0)
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'binance1-research-v45b'})
MAX_NEAREST_SEC=5.0

def vision_url(day:str)->str:
    return f'{BASE}/INJUSDT-bookDepth-{day}.zip'

def nearest_times(all_ts:pd.DatetimeIndex,targets:List[pd.Timestamp])->Dict[pd.Timestamp,pd.Timestamp|None]:
    out={}
    if len(all_ts)==0:
        return {t:None for t in targets}
    vals=all_ts.asi8
    for target in targets:
        x=target.value; pos=int(np.searchsorted(vals,x)); cand=[]
        if pos<len(vals):cand.append(pos)
        if pos>0:cand.append(pos-1)
        if not cand:out[target]=None;continue
        best=min(cand,key=lambda i:abs(vals[i]-x)); dt=abs(vals[best]-x)/1e9
        out[target]=pd.Timestamp(vals[best],tz='UTC') if dt<=MAX_NEAREST_SEC else None
    return out

def load_or_extract_day(day:str,targets:List[pd.Timestamp])->pd.DataFrame:
    cache=CACHE/f'{day}.csv.gz'
    if cache.exists():
        d=pd.read_csv(cache)
        if not d.empty:
            d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='coerce')
            d['percentage']=pd.to_numeric(d['percentage'],errors='coerce')
            d['depth']=pd.to_numeric(d['depth'],errors='coerce')
            d['notional']=pd.to_numeric(d['notional'],errors='coerce')
        return d
    r=SESSION.get(vision_url(day),timeout=30)
    if r.status_code!=200:
        print(f'[V45B][WARN] day={day} status={r.status_code}',flush=True)
        pd.DataFrame(columns=['timestamp','percentage','depth','notional']).to_csv(cache,index=False,compression='gzip')
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            d=pd.read_csv(f)
    d.columns=[str(c).strip().lower() for c in d.columns]
    d['timestamp']=pd.to_datetime(d['timestamp'],utc=True,errors='coerce')
    d['percentage']=pd.to_numeric(d['percentage'],errors='coerce')
    d['depth']=pd.to_numeric(d['depth'],errors='coerce')
    d['notional']=pd.to_numeric(d['notional'],errors='coerce')
    d=d.dropna(subset=['timestamp','percentage'])
    all_ts=pd.DatetimeIndex(d['timestamp'].drop_duplicates().sort_values())
    mp=nearest_times(all_ts,targets)
    wanted={t for t in mp.values() if t is not None}
    z=d[d.timestamp.isin(wanted) & d.percentage.abs().isin(BANDS)][['timestamp','percentage','depth','notional']].copy()
    z.to_csv(cache,index=False,compression='gzip')
    return z

def pick_snapshot(d:pd.DataFrame,target:pd.Timestamp)->Tuple[pd.Timestamp|None,pd.DataFrame,float]:
    if d.empty:return None,d,np.nan
    all_ts=pd.DatetimeIndex(d['timestamp'].drop_duplicates().sort_values())
    mp=nearest_times(all_ts,[target]); t=mp[target]
    if t is None:return None,d.iloc[0:0],np.nan
    age=abs((t-target).total_seconds())
    return t,d[d.timestamp==t],age

def snap_features(s:pd.DataFrame,prefix:str,age:float)->Dict[str,float]:
    out={f'{prefix}_age_sec':float(age) if np.isfinite(age) else np.nan}
    vals={float(r.percentage):r for r in s.itertuples(index=False)} if not s.empty else {}
    for b in BANDS:
        tag=str(b).replace('.','p')
        rb=vals.get(-b); ra=vals.get(b)
        bn=float(rb.notional) if rb is not None else np.nan
        an=float(ra.notional) if ra is not None else np.nan
        bd=float(rb.depth) if rb is not None else np.nan
        ad=float(ra.depth) if ra is not None else np.nan
        out[f'{prefix}_{tag}_bid_notional']=bn; out[f'{prefix}_{tag}_ask_notional']=an
        out[f'{prefix}_{tag}_bid_depth']=bd; out[f'{prefix}_{tag}_ask_depth']=ad
        out[f'{prefix}_{tag}_notional_imb']=(bn-an)/(bn+an) if np.isfinite(bn) and np.isfinite(an) and bn+an>0 else np.nan
        out[f'{prefix}_{tag}_depth_imb']=(bd-ad)/(bd+ad) if np.isfinite(bd) and np.isfinite(ad) and bd+ad>0 else np.nan
    return out

def change_features(a:Dict[str,float],b:Dict[str,float])->Dict[str,float]:
    out={}
    for band in BANDS:
        tag=str(band).replace('.','p')
        for kind in ['notional_imb','depth_imb']:
            x=a.get(f'bd0_{tag}_{kind}',np.nan); y=b.get(f'bd30_{tag}_{kind}',np.nan)
            out[f'bd_{tag}_{kind}_delta_0_30']=y-x if np.isfinite(x) and np.isfinite(y) else np.nan
        for kind in ['bid_notional','ask_notional','bid_depth','ask_depth']:
            x=a.get(f'bd0_{tag}_{kind}',np.nan); y=b.get(f'bd30_{tag}_{kind}',np.nan)
            out[f'bd_{tag}_{kind}_ratio_0_30']=y/x if np.isfinite(x) and np.isfinite(y) and x>0 else np.nan
    return out

def main():
    src=pd.read_csv(SRC)
    src['confirm_ts']=pd.to_datetime(src.confirm_ts,utc=True)
    days=sorted(src.confirm_ts.dt.strftime('%Y-%m-%d').unique())
    rows:List[dict]=[]; missing=[]
    print(f'[V45B] entries={len(src)} days={len(days)}',flush=True)
    for di,day in enumerate(days,1):
        sub=src[src.confirm_ts.dt.strftime('%Y-%m-%d')==day]
        targets=[]
        for t in sub.confirm_ts:
            targets.extend([pd.Timestamp(t),pd.Timestamp(t)+pd.Timedelta(seconds=30)])
        dd=load_or_extract_day(day,targets)
        if dd.empty:missing.append(day)
        for r in sub.itertuples(index=False):
            t0=pd.Timestamp(r.confirm_ts); t30=t0+pd.Timedelta(seconds=30)
            ts0,s0,age0=pick_snapshot(dd,t0); ts30,s30,age30=pick_snapshot(dd,t30)
            f0=snap_features(s0,'bd0',age0); f30=snap_features(s30,'bd30',age30)
            out={'entry_ts':r.entry_ts,'confirm_ts':r.confirm_ts,
                 'bd0_ts':ts0.isoformat() if ts0 is not None else '',
                 'bd30_ts':ts30.isoformat() if ts30 is not None else ''}
            out.update(f0); out.update(f30); out.update(change_features(f0,f30)); rows.append(out)
        if di%10==0 or di==len(days):
            size=sum(p.stat().st_size for p in CACHE.glob('*.csv.gz'))
            print(f'[V45B] days={di}/{len(days)} rows={len(rows)} missing={len(missing)} cache_mb={size/1024/1024:.2f}',flush=True)
    feat=pd.DataFrame(rows)
    out=src.merge(feat,on=['entry_ts','confirm_ts'],how='left',validate='one_to_one')
    path=OUT/'SHORT1_BOOKDEPTH_V45B.csv'; out.to_csv(path,index=False)
    cov1=int(out['bd30_1p0_notional_imb'].notna().sum())
    cov02=int(out['bd30_0p2_notional_imb'].notna().sum())
    meta={'source_rows':len(src),'rows':len(out),'days':len(days),'missing_days':missing,
          'valid_bd30_1p0':cov1,'valid_bd30_0p2':cov02,'bands_pct':list(BANDS),'max_nearest_sec':MAX_NEAREST_SEC,
          'cache_bytes':sum(p.stat().st_size for p in CACHE.glob('*.csv.gz')),'created_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'[V45B] SAVED {path} rows={len(out)} cov1={cov1} cov02={cov02} cache_mb={meta["cache_bytes"]/1024/1024:.2f}',flush=True)

if __name__=='__main__':main()
