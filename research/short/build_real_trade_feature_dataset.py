#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from research.short.short_economic_features import build_short_economic_features, SHORT_ECON_FEATURES
SRC=Path('/home/kasm920/binance1-pro/data/offline_cache')
TR=ROOT/'research/short/results/SHORT_REAL_TRADES_WITH_CONFIRM.csv'
OUT=ROOT/'research/short/results/SHORT_REAL_TRADES_ECON_FEATURES.csv'

def main():
    tr=pd.read_csv(TR); tr['open_ts']=pd.to_datetime(tr.open_ts,utc=True)
    out=[]
    for sym,g in tr.groupby('symbol'):
        d=pd.read_csv(SRC/f'{sym}_3m_360d.csv')
        f=build_short_economic_features(d)
        close_ms=pd.to_numeric(d.close_time,errors='coerce').to_numpy('int64')
        for _,r in g.iterrows():
            tms=int(r.open_ts.timestamp()*1000)
            # strictly completed bar only
            j=np.searchsorted(close_ms,tms,side='left')-1
            if j<0: continue
            row=r.to_dict(); row['feature_bar_index']=int(j); row['feature_close_time']=int(close_ms[j])
            for c in SHORT_ECON_FEATURES: row[c]=f.iloc[j][c]
            out.append(row)
    z=pd.DataFrame(out).sort_values('open_ts').reset_index(drop=True)
    z.to_csv(OUT,index=False)
    print('ROWS',len(z),'FEATURES',len(SHORT_ECON_FEATURES),'NAN_ROWS',int(z[SHORT_ECON_FEATURES].isna().any(axis=1).sum()))
    print('WROTE',OUT)
if __name__=='__main__': main()
