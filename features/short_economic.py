#!/usr/bin/env python3
from __future__ import annotations

from typing import List
import hashlib
import numpy as np
import pandas as pd

SHORT_ECON_FEATURES: List[str] = [
    'ret_1','ret_2','ret_3','ret_6','ret_12','ret_20','ret_60','ret_240',
    'body_pct','range_pct','close_loc','upper_wick_pct','lower_wick_pct',
    'ema5_rel','ema10_rel','ema20_rel','ema50_rel','ema100_rel',
    'ema5_slope3','ema10_slope3','ema20_slope3','ema50_slope3','ema100_slope3',
    'atr14_pct','rv10','rv20','rv60','rv240',
    'vol_z10','vol_z20','vol_z60','vol_z240',
    'down_frac10','down_frac20','down_frac60','ret_mean10','ret_mean20','ret_mean60',
    'buy_ratio','imbalance','imb_mean3','imb_mean6','imb_mean12','imb_mean30',
    'ret_accel','range_z20','hour_sin','hour_cos',
]


SHORT_ECON_MODEL_FEATURES_V1: List[str] = [
    'rv60','rv20','ret_6','ema100_slope3','ema100_rel','body_pct',
    'ema10_slope3','ret_60','ema10_rel','ret_mean60','atr14_pct',
    'down_frac20','rv10','ema50_slope3','imb_mean6','ret_3',
]

SHORT_ECON_MODEL_FEATURE_HASH_V1 = hashlib.sha256(
    "\n".join(SHORT_ECON_MODEL_FEATURES_V1).encode()
).hexdigest()


def build_short_economic_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    for c in ['open','high','low','close','volume','taker_buy_base_volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    o,h,l,c,v,tb = (d[x] for x in ['open','high','low','close','volume','taker_buy_base_volume'])
    r = c.pct_change()
    f = pd.DataFrame(index=d.index)
    for n in [1,2,3,6,12,20,60,240]:
        f[f'ret_{n}'] = c.pct_change(n)
    f['body_pct'] = (c-o)/o
    f['range_pct'] = (h-l)/c
    rng = (h-l).replace(0,np.nan)
    f['close_loc'] = (c-l)/rng
    f['upper_wick_pct'] = (h-np.maximum(o,c))/c
    f['lower_wick_pct'] = (np.minimum(o,c)-l)/c
    for span in [5,10,20,50,100]:
        e = c.ewm(span=span,adjust=False).mean()
        f[f'ema{span}_rel'] = c/e - 1.0
        f[f'ema{span}_slope3'] = e/e.shift(3)-1.0
    pc = c.shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    f['atr14_pct'] = tr.rolling(14).mean()/c
    for w in [10,20,60,240]:
        mu=v.rolling(w).mean(); sd=v.rolling(w).std().replace(0,np.nan)
        f[f'vol_z{w}']=(v-mu)/sd
        f[f'rv{w}']=r.rolling(w).std()
    for w in [10,20,60]:
        f[f'down_frac{w}']=(r<0).astype(float).rolling(w).mean()
        f[f'ret_mean{w}']=r.rolling(w).mean()
    buy=tb/v.replace(0,np.nan)
    f['buy_ratio']=buy
    f['imbalance']=2.0*buy-1.0
    for w in [3,6,12,30]:
        f[f'imb_mean{w}']=f['imbalance'].rolling(w).mean()
    f['ret_accel']=r-r.shift(1)
    f['range_z20']=(f['range_pct']-f['range_pct'].rolling(20).mean())/f['range_pct'].rolling(20).std().replace(0,np.nan)
    ts=pd.to_datetime(pd.to_numeric(d['open_time'],errors='coerce'),unit='ms',utc=True)
    hour=ts.dt.hour + ts.dt.minute/60.0
    f['hour_sin']=np.sin(2*np.pi*hour/24.0)
    f['hour_cos']=np.cos(2*np.pi*hour/24.0)
    return f[SHORT_ECON_FEATURES].replace([np.inf,-np.inf],np.nan)
