#!/usr/bin/env python3
"""Post-hoc diagnostics of FROZEN OOS V7, no threshold fitting or routing."""
import pandas as pd,numpy as np,json
from pathlib import Path
root=Path('research/results')
selected=pd.read_csv(root/'SHORT1_INJ_V7_FROZEN_OOS_SELECTED_20261114.csv')
old=pd.read_csv('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
new=pd.read_csv('research/short/data/INJUSDT_3m_20261013_20261114_FROZEN_OOS.csv')
data=pd.concat([old.tail(310),new],ignore_index=True)
for c in ['open_time','close_time','open','high','low','close','quote_asset_volume','taker_buy_quote_volume']:
 data[c]=pd.to_numeric(data[c],errors='raise')
data['ret15m']=100*(data.close/data.close.shift(5)-1)
data['ret60m']=100*(data.close/data.close.shift(20)-1)
data['ret3m']=100*(data.close/data.close.shift(1)-1)
data['ret9m']=100*(data.close/data.close.shift(3)-1)
data['ret15m_candle']=100*(data.close/data.close.shift(5)-1)
data['taker_15m']=(2*data.taker_buy_quote_volume.rolling(5).sum()/data.quote_asset_volume.rolling(5).sum()-1)
data['taker_60m']=(2*data.taker_buy_quote_volume.rolling(20).sum()/data.quote_asset_volume.rolling(20).sum()-1)
joined=selected.merge(data[['close_time','ret15m','ret60m','ret3m','ret9m','taker_15m','taker_60m']],left_on='completed_close_ms',right_on='close_time',how='left',validate='one_to_one')
assert len(joined)==len(selected) and joined.ret60m.notna().all()
joined['period']=np.where(joined.completed_close_ms<1793491200000,'OCT','NOV') # 2026-Nov-01 UTC
joined['both_up']=(joined.ret15m>0)&(joined.ret60m>0)
joined['both_flow_buy']=(joined.taker_15m>0)&(joined.taker_60m>0)
joined['both_down']=(joined.ret15m<0)&(joined.ret60m<0)
joined['flow_and_up']=joined.both_up&joined.both_flow_buy
joined['outcome']=np.where(joined.stop_first_event,'STOP',np.where(joined.tp_first_event,'TP','TIME60'))
joined['date']=pd.to_datetime(joined.completed_close_ms,unit='ms',utc=True).dt.date.astype(str)
def nonoverlap(df):
 # Ordered by exact 3m source timestamp; at most one 60m horizon per 20 3m bars.
 keep=[];last=-1
 for i,row in df.sort_values('completed_close_ms').iterrows():
  current=int(row.completed_close_ms)
  if current>last:
   keep.append(i)
   last=current+20*180000
 return df.loc[keep]
reports={}
for grp in ('all','OCT','NOV'):
 sub=joined if grp=='all' else joined[joined.period==grp]
 for tag in ('all','both_up','both_down','both_flow_buy','flow_and_up','other_than_both_up'):
  v=sub if tag=='all' else sub[(sub.both_up if tag=='both_up' else sub.both_down if tag=='both_down' else sub.both_flow_buy if tag=='both_flow_buy' else sub.flow_and_up if tag=='flow_and_up' else ~sub.both_up)]
  indep=nonoverlap(v)
  reports[f'{grp}_{tag}']=dict(
      selected=len(v),
      selected_mean_net_roi_pct=round(float(v.assumed_net_roi3x_pct.mean()),4) if len(v) else None,
      selected_stop_pct=round(float(v.stop_first_event.mean()*100),2) if len(v) else None,
      selected_tp_pct=round(float(v.tp_first_event.mean()*100),2) if len(v) else None,
      independent_n=len(indep),
      independent_net_mean_pct=round(float(indep.assumed_net_roi3x_pct.mean()),4) if len(indep) else None,
      independent_net_positive=int((indep.assumed_net_roi3x_pct>0).sum()),
      independent_stop_count=int(indep.stop_first_event.sum()),
      independent_tp_count=int(indep.tp_first_event.sum()))
daily=joined.groupby('date').agg(selected=('model_score','size'),
 net_roi_mean_pct=('assumed_net_roi3x_pct','mean'),
 stops=('stop_first_event','sum'),tps=('tp_first_event','sum')).reset_index()
daily.to_csv(root/'SHORT1_INJ_V7_FROZEN_OOS_DAILY_20261114.csv',index=False)
joined.to_csv(root/'SHORT1_INJ_V7_FROZEN_OOS_SELECTED_CONTEXT_20261114.csv',index=False)
p=root/'SHORT1_INJ_V7_FROZEN_OOS_REGIME_DIAGNOSTICS_20261114.json'
p.write_text(json.dumps(reports,indent=2))
for key in ('all_all','all_both_up','all_both_down','all_other_than_both_up','OCT_all','OCT_both_up','OCT_other_than_both_up','NOV_all','NOV_both_up','NOV_other_than_both_up'):
 print(key,reports[key])
print('REPORT',p)
print('WARNING: post-hoc stratification on final holdout; use descriptive only, no parameter promotion.')
