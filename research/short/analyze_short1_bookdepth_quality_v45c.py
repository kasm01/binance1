#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path('/home/kasm920/binance1')
V37=ROOT/'research/short/results/SHORT1_ENTRY_MICRO_V37.csv'
V45=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V45B/SHORT1_BOOKDEPTH_V45B.csv'
OUT=ROOT/'research/short/results/SHORT1_BOOKDEPTH_V45C_QUALITY.json'

def implied(df,prefix,band,side):
    n=pd.to_numeric(df[f'{prefix}_{band}_{side}_notional'],errors='coerce')
    d=pd.to_numeric(df[f'{prefix}_{band}_{side}_depth'],errors='coerce')
    return n/d.replace(0,np.nan)

def valid_pair(df,prefix,band,tol):
    px=pd.to_numeric(df.confirm_open,errors='coerce')
    bp=implied(df,prefix,band,'bid'); ap=implied(df,prefix,band,'ask')
    return bp.notna() & ap.notna() & (bp<px) & (ap>px) & ((px-bp)/px<tol) & ((ap-px)/px<tol)

def summary(g):
    if len(g)==0:return {'n':0}
    touch=g.touch40_5m.astype(str)
    return {'n':int(len(g)),'adverse_first_rate':float((touch=='ADVERSE_FIRST').mean()),'favorable_first_rate':float((touch=='FAVORABLE_FIRST').mean()),
            'mae5_median':float(pd.to_numeric(g.mae5,errors='coerce').median()),'mfe5_median':float(pd.to_numeric(g.mfe5,errors='coerce').median()),
            'win_rate':float(pd.to_numeric(g.final_win,errors='coerce').mean())}

a=pd.read_csv(V37); b=pd.read_csv(V45)
for c in ['confirm_ts','entry_ts']: a[c]=pd.to_datetime(a[c],utc=True)
bookcols=[c for c in b.columns if c.startswith('bd')]
d=pd.concat([a.reset_index(drop=True),b[bookcols].reset_index(drop=True)],axis=1)
d['q1_0']=valid_pair(d,'bd0','1p0',0.02); d['q1_30']=valid_pair(d,'bd30','1p0',0.02); d['q1']=d.q1_0&d.q1_30
d['q02_0']=valid_pair(d,'bd0','0p2',0.005); d['q02_30']=valid_pair(d,'bd30','0p2',0.005); d['q02']=d.q02_0&d.q02_30
# Main historical liquidity shift uses ±1% only.
d['liq_shift_1']=pd.to_numeric(d.bd_1p0_notional_imb_delta_0_30,errors='coerce')
d['bid_ratio_1']=pd.to_numeric(d.bd_1p0_bid_notional_ratio_0_30,errors='coerce')
d['ask_ratio_1']=pd.to_numeric(d.bd_1p0_ask_notional_ratio_0_30,errors='coerce')
d['score_num']=pd.to_numeric(d.score,errors='coerce')
d['actionable15']=d.first_confirm_found_15s.astype(str).str.lower().isin(['true','1'])
d['gate_ready']=d.score_num>=0.54
d['month']=d.confirm_ts.dt.to_period('M').astype(str)
rows=[]
for m,g in d.groupby('month'):
 rows.append({'month':m,'n':len(g),'q1':int(g.q1.sum()),'q02':int(g.q02.sum()),'q1_rate':float(g.q1.mean()),'q02_rate':float(g.q02.mean())})
valid=d[d.q1].copy()
# Quantiles only on valid data.
qs=valid.liq_shift_1.quantile([.1,.25,.5,.75,.9]).to_dict() if len(valid) else {}
# Tertiles and tails.
groups={}
if len(valid)>=30:
 q33,q67=valid.liq_shift_1.quantile([1/3,2/3])
 groups['liq_shift_low']=summary(valid[valid.liq_shift_1<=q33]); groups['liq_shift_mid']=summary(valid[(valid.liq_shift_1>q33)&(valid.liq_shift_1<q67)]); groups['liq_shift_high']=summary(valid[valid.liq_shift_1>=q67])
 q10,q90=valid.liq_shift_1.quantile([.1,.9]); groups['liq_shift_bottom10']=summary(valid[valid.liq_shift_1<=q10]); groups['liq_shift_top10']=summary(valid[valid.liq_shift_1>=q90])
# More production-like subsets.
prod=valid[valid.gate_ready & valid.actionable15].copy()
pgroups={}
if len(prod)>=20:
 q33,q67=prod.liq_shift_1.quantile([1/3,2/3]); pgroups['low']=summary(prod[prod.liq_shift_1<=q33]); pgroups['mid']=summary(prod[(prod.liq_shift_1>q33)&(prod.liq_shift_1<q67)]); pgroups['high']=summary(prod[prod.liq_shift_1>=q67])
# interaction with exact 20->30 price response
valid['rebound_20_30']=pd.to_numeric(valid.ret_delta_20_30,errors='coerce')>0
valid['book_bullish_shift']=valid.liq_shift_1>0
inter={}
for bs in [False,True]:
 for rb in [False,True]:
  inter[f'book_up_{bs}_rebound_{rb}']=summary(valid[(valid.book_bullish_shift==bs)&(valid.rebound_20_30==rb)])
res={'rows':len(d),'quality1_n':int(d.q1.sum()),'quality02_n':int(d.q02.sum()),'quality1_rate':float(d.q1.mean()),'quality02_rate':float(d.q02.mean()),
     'month_quality':rows,'valid_summary':summary(valid),'prod_like_n':int(len(prod)),'prod_like_summary':summary(prod),
     'liq_shift_quantiles':{str(k):float(v) for k,v in qs.items()},'groups':groups,'prod_like_groups':pgroups,'interaction':inter}
OUT.write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
print('SAVED',OUT)
