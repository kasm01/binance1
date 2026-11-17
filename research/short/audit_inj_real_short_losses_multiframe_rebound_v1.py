#!/usr/bin/env python3
"""Real logged SHORT1 INJ exits, past-only multi-timeframe entry context + post-exit rebound.

Historical 60 logged trades Jul-Sep 2026. Never call Binance or write Redis.
All 1m/3m/5m data read from local CSV. Do NOT confuse these historical
old-strategy hard stops with current SHORT1 -8.97% ROI protection.
"""
from pathlib import Path
from datetime import timezone
import bisect,csv,json,math
import numpy as np,pandas as pd
from collections import Counter

BASE=Path('research/short/data')
RESULTS=Path('research/results'); RESULTS.mkdir(exist_ok=True,parents=True)
t=pd.read_csv('research/short/results/SHORT_REAL_TRADES_WITH_MODEL_SCORE.csv')
t=t[t.symbol=='INJUSDT'].copy().reset_index(drop=True)
t['open_ms']=(pd.to_datetime(t.open_ts,utc=True).astype('int64')//10**6).astype('int64')
t['exit_ms']=(pd.to_datetime(t.close_ts,utc=True).astype('int64')//10**6).astype('int64')
def bars(path):
    b=pd.read_csv(path,usecols=['open_time','close_time','open','high','low','close','quote_asset_volume','taker_buy_quote_volume'])
    for k in b.columns:b[k]=pd.to_numeric(b[k])
    return b
m1=bars(BASE/'real_confirm_window_1m/INJUSDT_1m.csv')
m3=bars(Path('/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_460d.csv'))
m5=bars(BASE/'candidate_5m_360d/INJUSDT_5m_360d.csv')
src={1:m1,3:m3,5:m5}
closes={n:b.close_time.to_numpy(np.int64) for n,b in src.items()}
def completed_idx(n,when):
    return bisect.bisect_left(closes[n],int(when))-1
def price_delta(n,i,k):
    a=src[n]
    if i-k<0:return math.nan
    return 100*(float(a.close.iloc[i])/float(a.close.iloc[i-k])-1)
def entry_context(when):
    out={}
    for n in (1,3,5):
        b=src[n];i=completed_idx(n,when)
        out[f'{n}m_completed_close_ms']=int(b.close_time.iloc[i]) if i>=0 else None
        out[f'{n}m_return_1bar_pct']=price_delta(n,i,1) if i>0 else math.nan
        out[f'{n}m_return_3bars_pct']=price_delta(n,i,3) if i>2 else math.nan
        out[f'{n}m_last_bearish']=bool(b.close.iloc[i]<b.open.iloc[i]) if i>=0 else None
        out[f'{n}m_up_frac_last3']=sum(b.close.iloc[i-j]>b.open.iloc[i-j] for j in range(3))/3 if i>=2 else math.nan
    i=completed_idx(1,when)
    for window in (15,30,60):
        out[f'past_{window}m_price_change_pct']=price_delta(1,i,window) if i>=window else math.nan
        if i+1>=window:
            chunk=m1.iloc[i-window+1:i+1]
            quote=float(chunk.quote_asset_volume.sum())
            flow=float(2*chunk.taker_buy_quote_volume.sum()/quote-1) if quote>0 else math.nan
        else:flow=math.nan
        out[f'past_{window}m_taker_flow']=flow
    return out

minute_open=m1.open_time.to_numpy(np.int64)
minute_close=m1.close_time.to_numpy(np.int64)
out=[]
for _,x in t.iterrows():
    entry=int(x.open_ms);closed=int(x.exit_ms)
    entry_price=float(x.open_price);exit_price=float(x.close_price)
    row=dict(trade_open_ts=x.open_ts,trade_close_ts=x.close_ts,
      open_ms=entry,exit_ms=closed,entry_price=entry_price,exit_price=exit_price,
      exit_reason=x.reason,logged_roi_pct=100*float(x.roi),
      logged_net8_return_fraction=float(x.net8),assumed_net8_pnl_usdt=float(x.net8)*float(x.notional),duration_sec=float(x.duration_sec),
      is_logged_loss=bool(x.roi<0),old_historical_stop=True)
    row.update(entry_context(entry))
    # Within trade, 1m HIGH/LOW without claiming intrabar sequence.
    first=bisect.bisect_right(minute_close,entry)
    last=bisect.bisect_left(minute_open,closed)
    trade=m1.iloc[first:last]  # only fully enclosed minute bars
    row['complete_1m_bars_inside_trade']=len(trade)
    if len(trade):
        row['max_unrealized_short_profit_price_pct']=100*(entry_price/float(trade.low.min())-1)
        row['max_unrealized_short_loss_price_pct']=100*(entry_price/float(trade.high.max())-1)
        row['best_before_exit_short_roi_3x_pct']=3*row['max_unrealized_short_profit_price_pct']
        row['short_price_favorable_before_exit_pct']=100*(entry_price-float(trade.low.min()))/entry_price
    else:
        row['max_unrealized_short_profit_price_pct']=math.nan
        row['max_unrealized_short_loss_price_pct']=math.nan
        row['best_before_exit_short_roi_3x_pct']=math.nan
        row['short_price_favorable_before_exit_pct']=math.nan
    # Post close: begin only after full closing minute to prevent inclusion of intra-exit price.
    start=bisect.bisect_right(minute_open,closed)  # first full minute opening after timestamp
    for horizon in (15,30,60,120,240):
        subset=m1.iloc[start:min(len(m1),start+horizon)]
        # Require complete continuous window.
        complete=(len(subset)==horizon and start<len(m1)
                  and start+horizon-1<len(m1) and
                  int(subset.open_time.iloc[-1])-int(subset.open_time.iloc[0])==(horizon-1)*60000)
        row[f'post_{horizon}m_available']=complete
        if complete:
            low=float(subset.low.min());high=float(subset.high.max());last_close=float(subset.close.iloc[-1])
            row[f'post_{horizon}m_min_price']=low
            row[f'post_{horizon}m_max_price']=high
            row[f'post_{horizon}m_close_price']=last_close
            row[f'post_{horizon}m_max_drop_from_exit_pct']=100*(exit_price-low)/exit_price
            row[f'post_{horizon}m_return_to_entry']=bool(low<=entry_price)
            row[f'post_{horizon}m_reach_old_entry_plus_tp125roi']=bool(low<=entry_price*(1-.0125/3))
            row[f'post_{horizon}m_worst_rise_from_exit_pct']=100*(high-exit_price)/exit_price
        else:
            row[f'post_{horizon}m_min_price']=math.nan
            row[f'post_{horizon}m_max_price']=math.nan
            row[f'post_{horizon}m_close_price']=math.nan
            row[f'post_{horizon}m_max_drop_from_exit_pct']=math.nan
            row[f'post_{horizon}m_return_to_entry']=None
            row[f'post_{horizon}m_reach_old_entry_plus_tp125roi']=None
            row[f'post_{horizon}m_worst_rise_from_exit_pct']=math.nan
    out.append(row)
r=pd.DataFrame(out)
r.to_csv(RESULTS/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv',index=False)
loss=r[r.is_logged_loss].copy()
def nvalid(s):return int(s.notna().sum())
def bool_count(s):return int((s==True).sum())
summary={'sample':'SHORT1_INJ_2026_JUL21_SEP02_CLOSED_LOGGED_REAL_TRADES',
         'total_logged_trades':int(len(r)),'logged_loss_count':int(len(loss)),
         'logged_win_count':int((~r.is_logged_loss).sum()),
         'loss_mean_logged_roi_pct':float(loss.logged_roi_pct.mean()),
         'loss_median_logged_roi_pct':float(loss.logged_roi_pct.median()),
         'loss_mean_assumed_net8_pnl_usdt':float(loss.assumed_net8_pnl_usdt.mean()),
         'win_mean_logged_roi_pct':float(r.loc[~r.is_logged_loss,'logged_roi_pct'].mean()),
         'historical_stops_differ_from_current':True,
         'definitions':{
           'bar_trend':'last fully completed bars before actual logged entry timestamp; 3bar=3m/9m/15m in 1m/3m/5m chart respectively',
           'uptrend':'positive past 15m or 60m close change, strictly as-of entry',
           'post_rebound':'after logged stop, subsequent minute LOW <= prior SHORT entry; counterfactual only, not executable order',
           'post_tp':'subsequent LOW <= old entry*(1-1.25%ROI/3x), ignores fees/funding',
           'no_lookahead':'entry indicators only prior completed bars; follow-up evaluation separate',
           'net8':'historical assumed net return fraction; multiplied by notional to estimate USDT, not exchange-confirmed fee PnL',
         }}
for n in (1,3,5):
  for key in (f'{n}m_return_1bar_pct',f'{n}m_return_3bars_pct'):
    s=loss[key].dropna()
    summary[f'loss_{key}_positive']={'count':int((s>0).sum()),'n':len(s),'pct':float((s>0).mean()*100) if len(s) else None}
  s=loss[f'{n}m_last_bearish'].dropna()
  summary[f'loss_{n}m_last_completed_bearish']={'count':int(s.sum()),'n':int(len(s))}
for w in (15,30,60):
 s=loss[f'past_{w}m_price_change_pct'].dropna()
 summary[f'loss_prior_{w}m_uptrend']={'count':int((s>0).sum()),'n':len(s)}
 flow=loss[f'past_{w}m_taker_flow'].dropna()
 summary[f'loss_prior_{w}m_buy_flow']={'count':int((flow>0).sum()),'n':len(flow)}
both=loss[['1m_return_3bars_pct','3m_return_3bars_pct','5m_return_3bars_pct']].dropna()
summary['loss_all_three_recent_timeframes_up']={'count':int((both>0).all(axis=1).sum()),'n':len(both)}
both15=loss[['past_15m_price_change_pct','past_60m_price_change_pct']].dropna()
summary['loss_prior_15m_and_60m_up']={'count':int((both15>0).all(axis=1).sum()),'n':len(both15)}
for h in (15,30,60,120,240):
 subset=loss[loss[f'post_{h}m_available']]
 summary[f'after_stop_{h}m']={'n':int(len(subset)),
   'return_to_old_entry':bool_count(subset[f'post_{h}m_return_to_entry']),
   'reach_old_entry_plus_tp125roi':bool_count(subset[f'post_{h}m_reach_old_entry_plus_tp125roi']),
   'median_max_drop_from_exit_pct':float(subset[f'post_{h}m_max_drop_from_exit_pct'].median()) if len(subset) else None,
   'median_worst_rise_from_exit_pct':float(subset[f'post_{h}m_worst_rise_from_exit_pct'].median()) if len(subset) else None}
# compare rising-context loss rates on all trades
for key in ('1m_return_3bars_pct','3m_return_3bars_pct','5m_return_3bars_pct','past_15m_price_change_pct','past_60m_price_change_pct'):
 d=r[[key,'is_logged_loss']].dropna()
 rising=d[d[key]>0];falling=d[d[key]<=0]
 summary[f'loss_rate_when_{key}_positive']={'up_n':int(len(rising)),'up_losses':int(rising.is_logged_loss.sum()),
   'up_loss_rate_pct':float(rising.is_logged_loss.mean()*100) if len(rising) else None,
   'not_up_n':int(len(falling)),'not_up_losses':int(falling.is_logged_loss.sum()),
   'not_up_loss_rate_pct':float(falling.is_logged_loss.mean()*100) if len(falling) else None}
path=RESULTS/'SHORT1_INJ_REAL_LOSS_ENTRY_TRENDS_REBOUND_SUMMARY_20261114.json'
path.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
print('FILES',path,RESULTS/'SHORT1_INJ_REAL_TRADES_ENTRY_1M3M5M_REBOUND_20261114.csv')
