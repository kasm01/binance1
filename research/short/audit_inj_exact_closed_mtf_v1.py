"""Exact 1m, completed calendar-aligned 3m/5m closes as-of each V6 decision.

Do not use rolling 3/5-minute returns as if they were candle aligned 3m/5m.
Uses already downloaded 1m CSV, no network, no trading.
"""
import json,bisect
from pathlib import Path
import numpy as np,pandas as pd
DIR=Path('research/results')
events=pd.read_csv(DIR/'SHORT1_INJ_V6_LIVE_CONTRA_96H_EVENTS_20261114.csv')
data=pd.read_csv('research/short/data/INJUSDT_1m_20261110_20261114_live_contra_audit.csv')
data['utc']=pd.to_datetime(data.open_time,unit='ms',utc=True)
bar_tables={}
for n in (1,3,5):
    # left-aligned exact UTC minute bucket with right-closed LAST 1m timestamp.
    groups=data.groupby(data.utc.dt.floor(f'{n}min'),sort=True)
    x=groups.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),
                 close_ms=('close_time','max'),bar_count=('close','size')).reset_index(drop=True)
    x=x[x.bar_count==n].reset_index(drop=True)
    bar_tables[n]=(x,x.close_ms.to_numpy('int64'))
rows=[]
for _,e in events.iterrows():
    z=dict(event_ms=int(e.event_ms),event_id=e.event_id,
           source_bar_ms=int(e.source_bar_ms),
           score_pass=bool(e.score_pass),confirmation=bool(e.confirmation),
           old_shadow_ready=bool(e.old_shadow_ready),
           prev_15m_return_pct=float(e.ret15_pct),
           prev_60m_return_pct=float(e.ret60_pct),
           caution_uptrend=bool(e.caution_uptrend))
    for n in (1,3,5):
        x,cl=bar_tables[n];i=bisect.bisect_left(cl,int(e.event_ms))-1
        if i>=3:
            z[f'{n}m_complete_close_ms']=int(cl[i])
            z[f'{n}m_onebar_return_pct']=100*(float(x.close.iloc[i])/float(x.close.iloc[i-1])-1)
            z[f'{n}m_last3bars_return_pct']=100*(float(x.close.iloc[i])/float(x.close.iloc[i-3])-1)
            z[f'{n}m_last_candle_bearish']=bool(float(x.close.iloc[i])<float(x.open.iloc[i]))
        else:
            z[f'{n}m_last3bars_return_pct']=np.nan
    z['all_closed_mtf_bearish']=all(z.get(f'{n}m_onebar_return_pct',0)<0 and z.get(f'{n}m_last_candle_bearish',False) for n in (1,3,5))
    z['3m5m_last3bars_bearish']=bool(z.get('3m_last3bars_return_pct',0)<0 and z.get('5m_last3bars_return_pct',0)<0)
    rows.append(z)
df=pd.DataFrame(rows)
out=DIR/'SHORT1_INJ_V6_96H_ALIGNED_CLOSED_1M3M5M_CONTEXT_20261114.csv'
df.to_csv(out,index=False)
valid=df[df.score_pass];ready=df[df.old_shadow_ready]
s=dict(all_events=len(df),score_pass=len(valid),shadow_ready=len(ready),
   score_pass_all_mtf_bearish=int(valid.all_closed_mtf_bearish.sum()),
   ready_all_mtf_bearish=int(ready.all_closed_mtf_bearish.sum()),
   score_pass_3m5m_falling=int(valid['3m5m_last3bars_bearish'].sum()),
   ready_3m5m_falling=int(ready['3m5m_last3bars_bearish'].sum()),
   ready_uptrend=int(ready.caution_uptrend.sum()),
   ready_uptrend_all_mtf_bearish=int((ready.caution_uptrend&ready.all_closed_mtf_bearish).sum()))
res=DIR/'SHORT1_INJ_V6_96H_ALIGNED_CLOSED_MTF_SUMMARY_20261114.json'
res.write_text(json.dumps(s,indent=2))
print(json.dumps(s,indent=2))
print('READY_EXACT_CLOSES')
print(ready[['event_ms','caution_uptrend','1m_onebar_return_pct','3m_onebar_return_pct',
             '5m_onebar_return_pct','all_closed_mtf_bearish','3m5m_last3bars_bearish']].to_string(index=False))
print('REPORT',out)
