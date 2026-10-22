#!/usr/bin/env python3
from __future__ import annotations
import io, os, sys, time, zipfile, json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import requests

ROOT=Path('/home/kasm920/binance1')
ENTRY=ROOT/'research/short/results/SHORT1_COMBINED_V30/C600/entries.csv'
CLOSED=ROOT/'research/short/results/SHORT1_COMBINED_V30/C600/closed_positions.csv'
ONE=ROOT/'research/short/data/inj_1m'
CACHE=ROOT/'research/short/data/aggtrades_entry_cache_v36'
OUT=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V36'
CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/data/futures/um/daily/aggTrades/INJUSDT'
CUTS=[5,10,15,20,30]
SESSION=requests.Session(); SESSION.headers.update({'User-Agent':'binance1-research-v36'})

def load_1m()->pd.DataFrame:
    parts=[pd.read_csv(p) for p in sorted(ONE.glob('*.csv'))]
    d=pd.concat(parts,ignore_index=True)
    d['ts']=pd.to_datetime(pd.to_numeric(d.open_time),unit='ms',utc=True)
    for c in ['open','high','low','close','quote_volume','taker_buy_quote_volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.sort_values('ts').drop_duplicates('ts').reset_index(drop=True)
    d['qmean20_prev']=d.quote_volume.rolling(20,min_periods=5).mean().shift(1)
    d['qmed20_prev']=d.quote_volume.rolling(20,min_periods=5).median().shift(1)
    sell=d.quote_volume-d.taker_buy_quote_volume
    d['flow1_full']=(d.taker_buy_quote_volume-sell)/d.quote_volume.replace(0,np.nan)
    return d

def load_entries()->pd.DataFrame:
    e=pd.read_csv(ENTRY); c=pd.read_csv(CLOSED)
    e['entry_ts']=pd.to_datetime(e.ts,utc=True); e['confirm_ts']=e.entry_ts-pd.Timedelta(minutes=1)
    c['open_ts']=pd.to_datetime(c.open_ts,utc=True); c['close_ts']=pd.to_datetime(c.close_ts,utc=True)
    keep=['open_ts','close_ts','pnl_net','reason']
    c=c[keep].drop_duplicates('open_ts',keep='last')
    e=e.merge(c,left_on='entry_ts',right_on='open_ts',how='left')
    e['final_win']=(pd.to_numeric(e.pnl_net,errors='coerce')>0).astype('Int64')
    return e

def vision_url(day:str)->str:
    return f'{BASE}/INJUSDT-aggTrades-{day}.zip'

def read_daily_zip_bytes(blob:bytes)->pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name=z.namelist()[0]
        with z.open(name) as f:
            d=pd.read_csv(f)
    # normalize Binance Vision headers
    ren={c:c.strip().lower() for c in d.columns}; d=d.rename(columns=ren)
    return d

def extract_day(day:str, minute_ms:np.ndarray)->pd.DataFrame:
    cache=CACHE/f'{day}.csv.gz'
    if cache.exists():
        return pd.read_csv(cache)
    # Reuse prior test ZIP if available; otherwise download ephemeral bytes only.
    old=ROOT/f'research/short/data/aggtrades_small/INJUSDT-aggTrades-{day}.zip'
    if old.exists():
        blob=old.read_bytes()
    else:
        r=SESSION.get(vision_url(day),timeout=30)
        if r.status_code!=200:
            print(f'[WARN] day={day} status={r.status_code}',flush=True)
            pd.DataFrame().to_csv(cache,index=False,compression='gzip'); return pd.DataFrame()
        blob=r.content
    d=read_daily_zip_bytes(blob)
    if d.empty:
        d.to_csv(cache,index=False,compression='gzip'); return d
    t=pd.to_numeric(d['transact_time'],errors='coerce').astype('Int64')
    # Some future Binance files may switch time precision. Normalize if necessary.
    med=float(t.dropna().median()) if t.notna().any() else 0.0
    if med>1e14: t=(t//1000).astype('Int64')
    d['transact_time_ms']=t
    wanted=set(int(x) for x in minute_ms)
    minute=(d.transact_time_ms//60000)*60000
    m=minute.isin(wanted) & (d.transact_time_ms < minute+30000)
    z=d.loc[m,['agg_trade_id','price','quantity','first_trade_id','last_trade_id','transact_time_ms','is_buyer_maker']].copy()
    z.to_csv(cache,index=False,compression='gzip')
    return z

def bool_maker(s:pd.Series)->np.ndarray:
    if s.dtype==bool: return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(['true','1','t','yes']).to_numpy(bool)

def first_touch_5m(d1:pd.DataFrame, entry_ts:pd.Timestamp, entry_px:float)->Dict[str,float|str]:
    w=d1[(d1.ts>=entry_ts)&(d1.ts<entry_ts+pd.Timedelta(minutes=5))]
    if w.empty or not np.isfinite(entry_px) or entry_px<=0: return {'touch40_5m':'MISSING','mae5':np.nan,'mfe5':np.nan}
    adv=entry_px*1.004; fav=entry_px*0.996; touch='NEITHER'
    for r in w.itertuples():
        a=float(r.high)>=adv; f=float(r.low)<=fav
        if a and f: touch='BOTH_SAME_BAR'; break
        if a: touch='ADVERSE_FIRST'; break
        if f: touch='FAVORABLE_FIRST'; break
    return {'touch40_5m':touch,'mae5':float(w.high.max()/entry_px-1),'mfe5':float(1-w.low.min()/entry_px)}

def snapshot_features(z:pd.DataFrame, minute_ms:int, candle_open:float, qmean:float, qmed:float)->Dict[str,float]:
    out:Dict[str,float]={}
    if z.empty: return out
    t=pd.to_numeric(z.transact_time_ms,errors='coerce').to_numpy(np.int64)
    p=pd.to_numeric(z.price,errors='coerce').to_numpy(float)
    q=pd.to_numeric(z.quantity,errors='coerce').to_numpy(float)
    maker=bool_maker(z.is_buyer_maker)
    quote=p*q
    # maker buyer => aggressive seller; non-maker buyer => aggressive buyer
    signed=np.where(maker,-quote,quote)
    ntr=(pd.to_numeric(z.last_trade_id,errors='coerce')-pd.to_numeric(z.first_trade_id,errors='coerce')+1).fillna(1).to_numpy(float)
    for sec in CUTS:
        m=t < int(minute_ms+sec*1000)
        if not m.any():
            for name in ['flow','buy_ratio','quote','volume_pace_mean','volume_pace_med','ret','range','close_loc','agg_count','trade_count','avg_trade_quote']:
                out[f'{name}_{sec}s']=np.nan
            continue
        pp=p[m]; qq=quote[m]; ss=signed[m]; nn=ntr[m]
        den=float(qq.sum()); flow=float(ss.sum()/den) if den>0 else np.nan
        last=float(pp[-1]); hi=float(np.max(pp)); lo=float(np.min(pp)); rng=hi-lo
        out[f'flow_{sec}s']=flow
        out[f'buy_ratio_{sec}s']=float((flow+1)/2) if np.isfinite(flow) else np.nan
        out[f'quote_{sec}s']=den
        scaled=den*(60.0/sec)
        out[f'volume_pace_mean_{sec}s']=float(scaled/qmean) if qmean and np.isfinite(qmean) and qmean>0 else np.nan
        out[f'volume_pace_med_{sec}s']=float(scaled/qmed) if qmed and np.isfinite(qmed) and qmed>0 else np.nan
        out[f'ret_{sec}s']=float(last/candle_open-1) if candle_open>0 else np.nan
        out[f'range_{sec}s']=float((hi-lo)/candle_open) if candle_open>0 else np.nan
        out[f'close_loc_{sec}s']=float((last-lo)/rng) if rng>0 else 0.5
        out[f'agg_count_{sec}s']=float(m.sum())
        out[f'trade_count_{sec}s']=float(nn.sum())
        out[f'avg_trade_quote_{sec}s']=float(den/max(nn.sum(),1.0))
    # Dynamics/efficiency; all causal at <=30s.
    for a,b in [(5,10),(10,20),(5,20),(15,30),(5,30),(20,30)]:
        out[f'flow_delta_{a}_{b}']=out.get(f'flow_{b}s',np.nan)-out.get(f'flow_{a}s',np.nan)
        out[f'ret_delta_{a}_{b}']=out.get(f'ret_{b}s',np.nan)-out.get(f'ret_{a}s',np.nan)
    f20=out.get('flow_20s',np.nan); r20=out.get('ret_20s',np.nan)
    f30=out.get('flow_30s',np.nan); r30=out.get('ret_30s',np.nan)
    out['sell_eff_20s']=float(max(0,-r20)/(max(0,-f20)+0.05)) if np.isfinite(f20) and np.isfinite(r20) else np.nan
    out['sell_eff_30s']=float(max(0,-r30)/(max(0,-f30)+0.05)) if np.isfinite(f30) and np.isfinite(r30) else np.nan
    out['flow_abs_20s']=abs(f20) if np.isfinite(f20) else np.nan
    out['flow_abs_30s']=abs(f30) if np.isfinite(f30) else np.nan
    return out

def main():
    e=load_entries(); d1=load_1m(); d1i=d1.set_index('ts')
    days=sorted(e.confirm_ts.dt.strftime('%Y-%m-%d').unique())
    rows:List[dict]=[]; missing_days=[]
    print(f'[V36] entries={len(e)} unique_days={len(days)}',flush=True)
    for di,day in enumerate(days,1):
        ed=e[e.confirm_ts.dt.strftime('%Y-%m-%d')==day].copy()
        mins=(ed.confirm_ts.astype('int64')//1_000_000).astype(np.int64).to_numpy()
        z=extract_day(day,mins)
        if z.empty: missing_days.append(day)
        for r in ed.itertuples():
            cts=pd.Timestamp(r.confirm_ts); ms=int(cts.value//1_000_000)
            try: cr=d1i.loc[cts]
            except KeyError: continue
            zz=z[((pd.to_numeric(z.transact_time_ms,errors='coerce')//60000)*60000)==ms] if not z.empty else z
            feat=snapshot_features(zz,ms,float(cr.open),float(cr.qmean20_prev),float(cr.qmed20_prev))
            out={'entry_ts':r.entry_ts,'confirm_ts':cts,'score':float(r.score),'entry_price':float(r.price),'notional':float(r.notional),
                 'pnl_net':float(r.pnl_net) if pd.notna(r.pnl_net) else np.nan,'reason':r.reason if isinstance(r.reason,str) else '',
                 'final_win':int(r.final_win) if pd.notna(r.final_win) else np.nan,'flow15_baseline':float(r.flow15),'flow60_baseline':float(r.flow60),
                 'confirm_open':float(cr.open),'confirm_close_full':float(cr.close),'confirm_flow_full':float(cr.flow1_full),
                 'agg_rows_30s':int(len(zz))}
            out.update(feat); out.update(first_touch_5m(d1,pd.Timestamp(r.entry_ts),float(r.price)))
            rows.append(out)
        if di%10==0 or di==len(days):
            print(f'[V36] days={di}/{len(days)} rows={len(rows)} cache={len(list(CACHE.glob("*.csv.gz")))} missing={len(missing_days)}',flush=True)
    R=pd.DataFrame(rows).sort_values('entry_ts')
    path=OUT/'SHORT1_ENTRY_MICRO_V36.csv'; R.to_csv(path,index=False)
    meta={'entries_source':len(e),'rows':len(R),'days':len(days),'missing_days':missing_days,'cuts_seconds':CUTS,
          'cache_bytes':sum(p.stat().st_size for p in CACHE.glob('*.csv.gz')),'created_utc':pd.Timestamp.now(tz='UTC').isoformat()}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('[V36] SAVED',path,'rows',len(R),'cache_mb',round(meta['cache_bytes']/1024/1024,2),'missing_days',len(missing_days),flush=True)

if __name__=='__main__': main()
