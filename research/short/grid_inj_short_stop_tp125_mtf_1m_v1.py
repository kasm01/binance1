"""Fixed historical entries, stop-first OHLC 1m, gross TP1.25 ROI, 3x, fees .14% notional.

Research sensitivities only; no model refit, no live changes; grid is exploratory
and NOT suitable for selecting a deployment threshold on the same history.
"""
import pandas as pd, numpy as np, json, bisect
from pathlib import Path
source=Path('research/results')
trades=pd.read_csv(source/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv')
market=pd.read_csv('research/short/data/real_confirm_window_1m/INJUSDT_1m.csv')
ot=market.open_time.to_numpy('int64')
ts=pd.to_datetime(trades.trade_open_ts,utc=True).astype('int64')//1000000
entry_indices=[bisect.bisect_right(ot,int(t)) for t in ts]
grid=[]
for stop in (.003,.005,.01,.015,.02,.0299):
 for max_hold in (15,30,60,120):
  each=[]
  for i in entry_indices:
   if i>=len(market) or i+max_hold>len(market):continue
   entry=float(market.open.iloc[i]);limit=entry*(1+stop);take=entry*(1-.0125/3)
   reason='TIME';px=float(market.close.iloc[i+max_hold-1])
   for k in range(i,i+max_hold):
    if float(market.high.iloc[k])>=limit:reason='STOP';px=limit;break
    if float(market.low.iloc[k])<=take:reason='TP';px=take;break
   roi=100*3*((entry-px)/entry-.0014)
   each.append((reason,roi))
  grid.append(dict(stop_adverse_price_pct=stop*100,horizon_minutes=max_hold,
   observations=len(each),mean_assumed_net_roi_pct=round(np.mean([v for _,v in each]),4),
   sum_assumed_net_roi_pct=round(sum(v for _,v in each),4),
   positive=int(sum(v>0 for _,v in each)),
   stop_count=int(sum(reason=='STOP' for reason,_ in each)),
   tp_count=int(sum(reason=='TP' for reason,_ in each))))
df=pd.DataFrame(grid)
path=source/'SHORT1_INJ_STOP_TP125_1M_OHLC_SENSITIVITY_20261114.csv'
df.to_csv(path,index=False)
print('60m variants:')
print(df[df.horizon_minutes==60].to_string(index=False))
print('WARN: exploratory grid with 60 old signal timestamps, overlapping strategies and artificial time-exits, no model intent proof.')
print('FILE',path)
