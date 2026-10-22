#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1')
SRC=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V36/SHORT1_ENTRY_MICRO_V36.csv'
CACHE=ROOT/'research/short/data/aggtrades_entry_cache_v36'
OUT=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'
MIN_SEC=1.8; MAX_SEC=15.0; REV=0.0001

def maker_bool(s):
    if s.dtype==bool:return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(['true','1','t','yes']).to_numpy(bool)

def snap(z,ms,open_px,sec):
    z=z[pd.to_numeric(z.transact_time_ms,errors='coerce')<=ms+int(sec*1000)]
    if z.empty:return {}
    p=pd.to_numeric(z.price,errors='coerce').to_numpy(float); q=pd.to_numeric(z.quantity,errors='coerce').to_numpy(float)
    mk=maker_bool(z.is_buyer_maker); quote=p*q; signed=np.where(mk,-quote,quote); den=quote.sum(); rng=p.max()-p.min(); last=p[-1]
    return {
      'first_confirm_flow':float(signed.sum()/den) if den>0 else np.nan,
      'first_confirm_buy_ratio':float((signed.sum()/den+1)/2) if den>0 else np.nan,
      'first_confirm_ret':float(last/open_px-1) if open_px>0 else np.nan,
      'first_confirm_range':float((p.max()-p.min())/open_px) if open_px>0 else np.nan,
      'first_confirm_close_loc':float((last-p.min())/rng) if rng>0 else .5,
      'first_confirm_quote':float(den),
      'first_confirm_agg_count':int(len(z)),
    }

def main():
    d=pd.read_csv(SRC); d['confirm_ts']=pd.to_datetime(d.confirm_ts,utc=True)
    caches={}
    rows=[]
    for r in d.itertuples(index=False):
        day=r.confirm_ts.strftime('%Y-%m-%d')
        if day not in caches:
            p=CACHE/f'{day}.csv.gz'; caches[day]=pd.read_csv(p) if p.exists() else pd.DataFrame()
        z=caches[day]; ms=int(r.confirm_ts.value//1_000_000)
        if not z.empty:
            zz=z[((pd.to_numeric(z.transact_time_ms,errors='coerce')//60000)*60000)==ms].sort_values('transact_time_ms')
        else: zz=z
        confirm_sec=np.nan; state_px=np.nan
        if not zz.empty:
            t=pd.to_numeric(zz.transact_time_ms,errors='coerce').to_numpy(np.int64); p=pd.to_numeric(zz.price,errors='coerce').to_numpy(float)
            rel=(t-ms)/1000.0; threshold=float(r.confirm_open)*(1-REV)
            # State at minimum sniper time: last trade known at or before MIN_SEC.
            idx=np.where(rel<=MIN_SEC)[0]
            if len(idx) and p[idx[-1]]<=threshold:
                confirm_sec=MIN_SEC; state_px=p[idx[-1]]
            else:
                idx=np.where((rel>MIN_SEC)&(rel<=MAX_SEC)&(p<=threshold))[0]
                if len(idx): confirm_sec=float(rel[idx[0]]); state_px=float(p[idx[0]])
        o={'first_confirm_sec':confirm_sec,'first_confirm_found_15s':bool(np.isfinite(confirm_sec)),'first_confirm_price':state_px}
        if np.isfinite(confirm_sec): o.update(snap(zz,ms,float(r.confirm_open),float(confirm_sec)))
        rows.append(o)
    a=pd.DataFrame(rows); out=pd.concat([d.reset_index(drop=True),a],axis=1)
    out.to_csv(OUT,index=False)
    print('rows',len(out),'confirm_15s',int(out.first_confirm_found_15s.sum()),'rate',round(out.first_confirm_found_15s.mean(),4))
    print('saved',OUT)
if __name__=='__main__':main()
