#!/usr/bin/env python3
"""Read-only retrospective SHORT entry-guard comparison on 60 logged INJ trades.

Only completed bars as-of candidate trigger; delayed entries simulated at NEXT
1m candle OPEN; stop-first ambiguity on 1m OHLC; NO new model intent is claimed.
Historical original trades and a *new synthetic lifecycle* are separate.
"""
from pathlib import Path
import pandas as pd,numpy as np,bisect,json
from collections import Counter
ROOT=Path('research/results');ROOT.mkdir(exist_ok=True,parents=True)
t=pd.read_csv(ROOT/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv')
m1=pd.read_csv('research/short/data/real_confirm_window_1m/INJUSDT_1m.csv')
m3=pd.read_csv('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv')
m5=pd.read_csv('research/short/data/candidate_5m_360d/INJUSDT_5m_360d.csv')
d={1:m1,3:m3,5:m5}
cm={n:frame.close_time.to_numpy(np.int64) for n,frame in d.items()}
open1=m1.open_time.to_numpy(np.int64)
fees_and_slip=.0014
leverage=3
tp_roi=.0125
stop_price=.0299
max_hold=60
def completed(n,ts):
    return bisect.bisect_left(cm[n],int(ts))-1
def positive_or_negative(n,i):
    frame=d[n]
    if i<1:return None
    return float(frame.close.iloc[i])-float(frame.close.iloc[i-1])
def context(ts):
    r={}
    for n in (1,3,5):
        idx=completed(n,ts)
        r[f'{n}m_closed']=int(cm[n][idx]) if idx>=0 else None
        r[f'{n}m_last_down']=positive_or_negative(n,idx)<0 if idx>=1 else None
        r[f'{n}m_last_bear']=bool(d[n].close.iloc[idx]<d[n].open.iloc[idx]) if idx>=0 else None
    j=completed(1,ts)
    r['sell_flow_3m']=None
    if j>=2:
        z=m1.iloc[j-2:j+1]
        q=float(z.quote_asset_volume.sum())
        r['sell_flow_3m']=bool(q>0 and 2*float(z.taker_buy_quote_volume.sum())/q-1 <0)
    return r
def price_replay(enter_idx,*,stop=stop_price,tp=tp_roi,hold=max_hold):
    entry=float(m1.open.iloc[enter_idx])
    max_k=min(enter_idx+hold,len(m1))
    end_k=None;reason='TIME60';px=None
    for k in range(enter_idx,max_k):
        high=float(m1.high.iloc[k]);low=float(m1.low.iloc[k])
        stop_px=entry*(1+stop)
        tp_px=entry*(1-tp/leverage)
        # If one bar has BOTH threshold crossings, STOP first, conservative.
        if high>=stop_px:
            reason='STOP';px=stop_px;end_k=k;break
        if low<=tp_px:
            reason='TP';px=tp_px;end_k=k;break
    if end_k is None:
        if max_k-enter_idx<hold:return None
        end_k=max_k-1;px=float(m1.close.iloc[end_k])
    gross_roi_pct=100*leverage*(entry-px)/entry
    net_roi_pct=gross_roi_pct-100*leverage*fees_and_slip
    return dict(synthetic_entry_price=entry,synthetic_exit_price=px,
                synthetic_exit_ms=int(m1.close_time.iloc[end_k]),
                synthetic_duration_min=int(end_k-enter_idx+1),
                synthetic_exit_reason=reason,
                synthetic_gross_roi_pct=round(gross_roi_pct,5),
                synthetic_net_roi_pct=round(net_roi_pct,5))
def revalidate(ts,*,deadline_min,require_flow):
    # Evaluate only fully completed 1m bars after entry; execute on subsequent
    # candle OPEN to avoid lookahead.
    start=bisect.bisect_left(cm[1],int(ts))
    for i in range(start,min(len(m1)-1,start+deadline_min)):
        when=int(cm[1][i])+1
        if when>ts+deadline_min*60000:break
        c=context(when)
        if c['1m_last_down'] is True and c['3m_last_down'] is True and c['5m_last_down'] is True:
            if require_flow and c['sell_flow_3m'] is not True:continue
            idx=i+1
            if int(open1[idx]) != when:continue
            return idx,c
    return None,None
records=[]
for _,a in t.iterrows():
    original_up=bool(a.past_15m_price_change_pct>0 and a.past_60m_price_change_pct>0)
    orig_pnl=float(a.assumed_net8_pnl_usdt)
    rec=dict(open_ts=a.trade_open_ts,old_exit_ts=a.trade_close_ts,
             original_loss=bool(a.is_logged_loss),
             original_estimated_net_usdt=orig_pnl,
             original_logged_roi_pct=float(a.logged_roi_pct),
             original_notional=float(a.entry_price)*float(t.loc[t.trade_open_ts==a.trade_open_ts,'entry_price'].iloc[0]*0+1) if False else float('nan'),
             context_15m60m_up=original_up,
             original_3m9m_rising=bool(a['3m_return_3bars_pct']>0),
             original_5m15m_rising=bool(a['5m_return_3bars_pct']>0))
    # Gate A: directly reject original trade if both 15m & 60m price rose
    rec['A_reject']=original_up
    rec['A_original_net_usdt_after_reject']=0 if original_up else orig_pnl
    for deadline in (5,10,15):
        for flow in (False,True):
            tag=f'PENDING_{deadline}min_'+('with_flow' if flow else 'no_flow')
            idx,c=revalidate(int(a.open_ms),deadline_min=deadline,require_flow=flow)
            rec[tag+'_revalidated']=bool(idx is not None)
            if idx is not None:
                rec[tag+'_delay_sec']=round((int(open1[idx])-int(a.open_ms))/1000,2)
                simulated=price_replay(idx)
                if simulated:
                    for k,v in simulated.items():rec[tag+'_'+k]=v
            else:
                rec[tag+'_delay_sec']=None
    # Counterfactual return after STOP: requires fresh model source which logs lack.
    rec['stop_after_60m_retouch']=bool(a.post_60m_return_to_entry) if bool(a.is_logged_loss) else None
    records.append(rec)
r=pd.DataFrame(records)
path=ROOT/'SHORT1_INJ_ENTRY_GUARD_60_TRADES_COUNTERFACTUAL_20261114.csv'
r.to_csv(path,index=False)
summary={'historical_count':len(r),'historical_losses':int(r.original_loss.sum()),
 'baseline_estimated_net_usdt':round(float(r.original_estimated_net_usdt.sum()),6),
 'note':'ORIGINAL historic net8 and synthetic 1m OHLC path have different fill/exit policies; compare only within same policy.',
 'A_15m60m_rise':{
  'blocked':int(r.A_reject.sum()),
  'blocked_losses':int((r.A_reject&r.original_loss).sum()),
  'blocked_winners':int((r.A_reject&~r.original_loss).sum()),
  'original_net_usdt_after_filter':round(float(r.A_original_net_usdt_after_reject.sum()),6),
  'original_net_usdt_blocked':round(float(r.loc[r.A_reject,'original_estimated_net_usdt'].sum()),6),
 },
 'B_delayed_confirmations':{}}
for deadline in (5,10,15):
 for flow in (False,True):
    tag=f'PENDING_{deadline}min_'+('with_flow' if flow else 'no_flow')
    selected=r[r[tag+'_revalidated']]
    pnl=selected[tag+'_synthetic_net_roi_pct'].dropna()
    summary['B_delayed_confirmations'][tag]={
       'revalidated':len(selected),
       'original_losers_revalidated':int(selected.original_loss.sum()),
       'original_winners_revalidated':int((~selected.original_loss).sum()),
       'blocked_original_losses':int((~r[tag+'_revalidated']&r.original_loss).sum()),
       'blocked_original_winners':int((~r[tag+'_revalidated']&~r.original_loss).sum()),
       'simulated_complete':int(len(pnl)),
       'synthetic_mean_net_roi_pct':round(float(pnl.mean()),4) if len(pnl) else None,
       'synthetic_net_wins':int((pnl>0).sum()),
       'synthetic_stops':int((selected[tag+'_synthetic_exit_reason']=='STOP').sum()),
       'synthetic_tps':int((selected[tag+'_synthetic_exit_reason']=='TP').sum()),
       'median_delay_sec':round(float(selected[tag+'_delay_sec'].median()),1) if len(selected) else None,
    }
out=ROOT/'SHORT1_INJ_ENTRY_GUARD_COMPARISON_SUMMARY_20261114.json'
out.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('OUTPUT',path,out)
