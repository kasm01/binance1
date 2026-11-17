#!/usr/bin/env python3
"""V44 20s order book SHORT sell capacity and forward markouts.

No inferential production liq_score, no synthetic whale, no order, no Redis
writes. Aggregated five-level quantities do not encode VWAP or real slippage.
"""
from collections import Counter
from pathlib import Path
import json,redis,statistics,csv,math
ROOT=Path('research/results')
r=redis.Redis(decode_responses=True,socket_timeout=3)
records=[]
# Use 2400 recent outcomes only: bounded and avoids heavy full stream transfer.
for sid,fields in r.xrevrange('telemetry:short1:inj:entry_hazard_orderbook_v44_outcomes',count=2400):
    try:
        e=json.loads(fields.get('json') or '{}')
        ob=e.get('orderbook_20s') or {}
        labels=e.get('labels_from_30s_decision') or {}
        if e.get('symbol')!='INJUSDT' or ob.get('available') is not True:continue
        src=int(ob['source_event_ms']);obs=int(ob['observed_at_ms'])
        decision=int(e['decision_ts_ms'])
        # Key distinction: outcomes available after 30s; 20s snapshot itself
        # must precede 30s price decision, not be post-label leakage.
        if not (0<src<=obs<=decision and 0<=obs-src<=2000):continue
        bid=float(ob['best_bid']);ask=float(ob['best_ask'])
        bid1=float(ob['bid_qty_l1']);bid5=float(ob['bid_qty_5'])
        ask1=float(ob['ask_qty_l1']);ask5=float(ob['ask_qty_5'])
        spread=float(ob['spread_bps'])
        if not (bid>0 and ask>=bid and bid1>=0 and bid5>=bid1 and ask1>=0 and ask5>=ask1 and spread>=0):
            continue
        row=dict(event_id=e['event_id'],model_source_event_id=e.get('source_event_id'),
            minute_ms=int(e['minute_ms']),snapshot_20s_ms=obs,
            price_mid=(bid+ask)/2,spread_bps=spread,
            bid_qty_l1=bid1,bid_qty_5=bid5,ask_qty_l1=ask1,ask_qty_5=ask5,
            bid_notional_l1_usdt=round(bid*bid1,5),
            bid_notional_l5_usdt=round(bid*bid5,5),
            depth5_imbalance=float(ob.get('depth5_imbalance') or 0),
            book_age_ms=obs-src,decision_minus_snapshot_ms=decision-obs)
        # Capacity ratio is NOT slippage, merely quantity coverage proxy.
        for amount in (30,100,500):
            row[f'bid_l1_cover_{amount}usdt']=bid*bid1>=amount
            row[f'bid_l5_cover_{amount}usdt']=bid*bid5>=amount
        for h in ('1m','3m','5m'):
            z=labels.get(h) or {}
            if z.get('available') is True:
                row[f'{h}_future_mfe_pct']=z.get('mfe_pct')
                row[f'{h}_future_mae_pct']=z.get('mae_pct')
                row[f'{h}_future_touch40bp']=z.get('touch_40bp')
        records.append(row)
    except (ValueError,TypeError,KeyError):
        continue
path=ROOT/'SHORT1_INJ_V44_20S_LIQUIDITY_CAPACITY_20261114.csv'
if not records:raise RuntimeError('no_matched_v44_outcomes')
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]),extrasaction='ignore')
    writer.writeheader();writer.writerows(records)
def med(xs):
    v=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return round(statistics.median(v),5) if v else None
summary=dict(outcome_stream_sample=2400,valid_joined_v44_events=len(records),
    spread_bps_median=med(x['spread_bps'] for x in records),
    spread_bps_p10=round(sorted(x['spread_bps'] for x in records)[int(len(records)*.1)],5),
    spread_bps_p90=round(sorted(x['spread_bps'] for x in records)[int(len(records)*.9)],5),
    bid_l1_depth_usdt_median=med(x['bid_notional_l1_usdt'] for x in records),
    bid_l5_depth_usdt_median=med(x['bid_notional_l5_usdt'] for x in records),
    orderbook_age_ms_median=med(x['book_age_ms'] for x in records),
    coverage_by_trade_notional={},
    future_market_move_by_l1_100usdt_coverage={},
    no_execution_vwap_levels=True,no_slippage_estimate=True,model_approved=False)
for amount in (30,100,500):
    summary['coverage_by_trade_notional'][str(amount)]=dict(
        observed_l1_cover=int(sum(x[f'bid_l1_cover_{amount}usdt'] for x in records)),
        observed_l1_cover_pct=round(sum(x[f'bid_l1_cover_{amount}usdt'] for x in records)*100/len(records),2),
        observed_l5_cover_pct=round(sum(x[f'bid_l5_cover_{amount}usdt'] for x in records)*100/len(records),2))
for covered in (True,False):
    sub=[x for x in records if x['bid_l1_cover_100usdt'] is covered]
    summary['future_market_move_by_l1_100usdt_coverage'][str(covered)]=dict(
        n=len(sub),
        median_future_1m_mae_pct=med(x.get('1m_future_mae_pct') for x in sub),
        median_future_3m_mae_pct=med(x.get('3m_future_mae_pct') for x in sub),
        median_future_5m_mae_pct=med(x.get('5m_future_mae_pct') for x in sub),
        favorite_first_5m=int(sum(x.get('5m_future_touch40bp')=='FAVORABLE_FIRST' for x in sub)))
summ=ROOT/'SHORT1_INJ_V44_20S_LIQUIDITY_CAPACITY_SUMMARY_20261114.json'
summ.write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('FILES',path,summ)
