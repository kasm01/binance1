"""Compare immediate vs delayed SHORT on SAME 1m stop-first exit assumptions.

No model/production wiring or Binance orders; overlapping logged entries
are fixed observations, not executable account portfolio history.
"""
from pathlib import Path
import json,bisect
import pandas as pd
import numpy as np
p=Path('research/results')
df=pd.read_csv(p/'SHORT1_INJ_ENTRY_GUARD_60_TRADES_COUNTERFACTUAL_20261114.csv')
bars=pd.read_csv('research/short/data/real_confirm_window_1m/INJUSDT_1m.csv')
times=bars.open_time.to_numpy('int64')
def simulate(idx):
    if idx>=len(bars) or idx+60>len(bars):return None
    entry=float(bars.open.iloc[idx])
    stop=entry*(1+.0299)
    target=entry*(1-.0125/3)
    px=float(bars.close.iloc[idx+59]);reason='TIME60'
    for j in range(idx,idx+60):
        if bars.high.iloc[j]>=stop:reason='STOP';px=stop;break
        if bars.low.iloc[j]<=target:reason='TP';px=target;break
    return {'roi':100*3*((entry-px)/entry-.0014),'reason':reason}
results=[]
for _,row in df.iterrows():
    event_ms=int(pd.Timestamp(row.open_ts).value//1_000_000)
    idx=bisect.bisect_right(times,event_ms)
    r=simulate(idx) if idx<len(times) and times[idx]-event_ms<=60000 else None
    results.append(r)
df['baseline_net_roi_pct']=[x['roi'] if x else np.nan for x in results]
df['baseline_exit']=[x['reason'] if x else None for x in results]
veto=df.baseline_net_roi_pct.where(~df.context_15m60m_up,0)
s={'count':len(df),'baseline_count':int(df.baseline_net_roi_pct.notna().sum()),
   'baseline_mean_net_roi_pct':float(df.baseline_net_roi_pct.mean()),
   'baseline_net_roi_sum_pct':float(df.baseline_net_roi_pct.sum()),
   'baseline_stops':int((df.baseline_exit=='STOP').sum()),
   'baseline_tp':int((df.baseline_exit=='TP').sum()),
   'veto_15m60m_net_roi_sum_pct':float(veto.sum()),'pending_variants':{},
   'warning':'Research proxy only, not actual fills, time-based production stop, or independent new model intent.'}
for minutes in (5,10,15):
    for flow in (False,True):
        key=f'PENDING_{minutes}min_'+('with_flow' if flow else 'no_flow')
        col=key+'_synthetic_net_roi_pct'
        matched=df[col].notna()
        diff=(df.loc[matched,col]-df.loc[matched,'baseline_net_roi_pct']).dropna()
        s['pending_variants'][key]=dict(
            matched=int(matched.sum()),skipped=int((~matched).sum()),
            matched_new_mean_net_roi_pct=float(df.loc[matched,col].mean()),
            matched_baseline_mean_net_roi_pct=float(df.loc[matched,'baseline_net_roi_pct'].mean()),
            paired_delta_mean_net_roi_pct=float(diff.mean()) if len(diff) else None,
            total_roi_sum_with_no_trade_zero_pct=float(df[col].fillna(0).sum()),
            baseline_roi_sum_pct=float(df.baseline_net_roi_pct.sum()))
out=p/'SHORT1_INJ_ENTRY_GUARDS_SAME_EXIT_BASELINE_20261114.json'
out.write_text(json.dumps(s,indent=2))
df.to_csv(p/'SHORT1_INJ_ENTRY_GUARDS_SAME_EXIT_PAIRED_20261114.csv',index=False)
print(json.dumps(s,indent=2))
