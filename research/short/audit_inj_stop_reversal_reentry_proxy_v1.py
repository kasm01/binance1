#!/usr/bin/env python3
"""After STOP, first 1m+3m+5m completed bearish context, new next-1m OPEN.

Only conditional price-path research: NO independent model ID at these Jul-Sep
timestamps, no true SHORT1 admission, no actual exchange re-entry.
"""
import bisect,json
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
out=Path('research/results')
records=pd.read_csv(out/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv')
losses=records[records.is_logged_loss].copy()
paths={
1:pd.read_csv('research/short/data/real_confirm_window_1m/INJUSDT_1m.csv'),
3:pd.read_csv('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv'),
5:pd.read_csv('research/short/data/candidate_5m_360d/INJUSDT_5m_360d.csv'),
}
ends={n:p.close_time.to_numpy('int64') for n,p in paths.items()}
m=paths[1];opens=m.open_time.to_numpy('int64')
def last_bear(n,time_ms):
    i=bisect.bisect_left(ends[n],int(time_ms))-1
    if i<1:return False
    p=paths[n]
    # 1bar negative close-to-close AND bearish candle
    return bool(p.close.iloc[i]<p.close.iloc[i-1] and p.close.iloc[i]<p.open.iloc[i])
def simulate(i):
    if i+60>len(m):return None
    ent=float(m.open.iloc[i])
    sl=ent*1.0299; tp=ent*(1-.0125/3)
    for j in range(i,i+60):
        hi=float(m.high.iloc[j]);lo=float(m.low.iloc[j])
        if hi>=sl:return ('STOP',100*3*((ent-sl)/ent-.0014),j-i+1)
        if lo<=tp:return ('TP',100*3*((ent-tp)/ent-.0014),j-i+1)
    ex=float(m.close.iloc[i+59])
    return ('TIME60',100*3*((ent-ex)/ent-.0014),60)
outrows=[]
for _,x in losses.iterrows():
    end=int(x.exit_ms)
    row=dict(original_open_ts=x.trade_open_ts,original_exit_ts=x.trade_close_ts,
             logged_stop_roi_pct=x.logged_roi_pct)
    for lookahead in (5,15,30,60,120):
        prefix=f'RE_ENTRY_{lookahead}m'
        idx=None;done_at=None
        start=bisect.bisect_right(ends[1],end)
        for j in range(start,min(len(m)-1,start+lookahead)):
            now=int(ends[1][j])+1
            if now>end+lookahead*60000:break
            if all(last_bear(frame,now) for frame in (1,3,5)):
                candidate=j+1
                if int(opens[candidate])==now:
                    idx=candidate;done_at=now;break
        row[prefix+'_found']=bool(idx is not None)
        if idx is not None:
            outcome=simulate(idx)
            row[prefix+'_wait_min']=round((done_at-end)/60000,2)
            row[prefix+'_price']=float(m.open.iloc[idx])
            row[prefix+'_price_vs_stop_exit_pct']=100*(float(m.open.iloc[idx])/float(x.exit_price)-1)
            if outcome:
                why,roi,mins=outcome
                row[prefix+'_exit_reason']=why
                row[prefix+'_net_roi3x_pct']=round(roi,5)
                row[prefix+'_duration_min']=mins
    outrows.append(row)
frame=pd.DataFrame(outrows)
frame.to_csv(out/'SHORT1_INJ_POST_STOP_MULTIFRAME_REENTRY_PROXY_20261114.csv',index=False)
summary={}
for horizon in (5,15,30,60,120):
    pref=f'RE_ENTRY_{horizon}m'
    subset=frame[frame[pref+'_found']]
    outcomes=subset[pref+'_net_roi3x_pct'].dropna()
    summary[str(horizon)]={
        'original_stop_count':len(losses),
        'reversal_confirmed':len(subset),
        'mean_delay_min':round(float(subset[pref+'_wait_min'].mean()),2) if len(subset) else None,
        'median_delta_entry_vs_old_stop_exit_pct':round(float(subset[pref+'_price_vs_stop_exit_pct'].median()),4) if len(subset) else None,
        'simulated_complete':len(outcomes),
        'simulated_net_roi_mean_pct':round(float(outcomes.mean()),4) if len(outcomes) else None,
        'net_positive':int((outcomes>0).sum()),
        'stops':int((subset[pref+'_exit_reason']=='STOP').sum()),
        'tp':int((subset[pref+'_exit_reason']=='TP').sum()),
    }
destination=out/'SHORT1_INJ_POST_STOP_REENTRY_PROXY_SUMMARY_20261114.json'
destination.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('CAUTION: no independent fresh MODEL revalidation; this is historical bearish bar conjunction, not executable SHORT ENTRY.')
print('OUTPUT',destination)
