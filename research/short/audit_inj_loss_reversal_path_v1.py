"""Detailed post-STOP downside recovery timing and adverse exposure until recovery.
This is retrospective path observation, NEVER safe recommendation to delay stop.
"""
import pandas as pd,numpy as np,bisect,json
from pathlib import Path
results=Path('research/results')
t=pd.read_csv(results/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv')
m=pd.read_csv('research/short/data/real_confirm_window_1m/INJUSDT_1m.csv')
starts=m.open_time.to_numpy('int64')
records=[]
for _,x in t[t.is_logged_loss].iterrows():
    ts=int(x.exit_ms); entry=float(x.entry_price); close=float(x.exit_price)
    first=bisect.bisect_right(starts,ts)
    horizon=m.iloc[first:min(len(m),first+240)].copy()
    if len(horizon)<240: continue
    # Returns to entry via 1m LOW, and from there an approximate TP for current gross 1.25% ROI
    entry_hit=horizon.low.to_numpy(float)<=entry
    target=entry*(1-.0125/3)
    target_hit=horizon.low.to_numpy(float)<=target
    def first_min(hits):
        where=np.flatnonzero(hits)
        return int(where[0])+1 if len(where) else None
    entry_min=first_min(entry_hit);target_min=first_min(target_hit)
    def adverse_until(minute):
        subset=horizon.iloc[:minute] if minute else horizon
        return float((subset.high.max()-close)/close*100)
    # market closing HIGH after exit before recovery, no intrabar ordering; adverse max inclusive
    records.append(dict(open_ts=x.trade_open_ts,exit_ts=x.trade_close_ts,
      logged_roi_pct=x.logged_roi_pct,entry=entry,stop_close=close,
      return_entry_min=entry_min,reach_current_tp_min=target_min,
      adverse_high_before_return_entry_pct=adverse_until(entry_min),
      adverse_high_before_current_tp_pct=adverse_until(target_min),
      worst_adverse_240m_pct=adverse_until(None),
      prior_60m_up=bool(x.past_60m_price_change_pct>0),
      prior_15m_up=bool(x.past_15m_price_change_pct>0),
      prior_3m_3bar_up=bool(x['3m_return_3bars_pct']>0),
      prior_5m_3bar_up=bool(x['5m_return_3bars_pct']>0)))
d=pd.DataFrame(records)
d.to_csv(results/'SHORT1_INJ_23_LOSSES_POST_EXIT_RETURN_TIMING_20261114.csv',index=False)
summary={
 'losses':len(d),
 'return_to_entry_4h':int(d.return_entry_min.notna().sum()),
 'return_to_tp_4h':int(d.reach_current_tp_min.notna().sum()),
 'median_minutes_to_old_entry_retouch':float(d.return_entry_min.median()),
 'median_minutes_to_current_tp_retouch':float(d.reach_current_tp_min.median()),
 'median_adverse_rise_before_entry_retouch_pct':float(d.loc[d.return_entry_min.notna(),'adverse_high_before_return_entry_pct'].median()),
 'max_adverse_rise_before_entry_retouch_pct':float(d.loc[d.return_entry_min.notna(),'adverse_high_before_return_entry_pct'].max()),
 'median_4h_max_additional_adverse_rise_pct':float(d.worst_adverse_240m_pct.median()),
 'losses_no_return_4h':int(d.return_entry_min.isna().sum())}
print('SUMMARY',json.dumps(summary))
print('EXAMPLES',d[['open_ts','logged_roi_pct','return_entry_min','reach_current_tp_min','adverse_high_before_return_entry_pct','worst_adverse_240m_pct']].to_string(index=False,float_format=lambda x:f'{x:.2f}'))
(results/'SHORT1_INJ_POST_EXIT_RETURN_TIMING_SUMMARY_20261114.json').write_text(json.dumps(summary,indent=2))
