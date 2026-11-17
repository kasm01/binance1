#!/usr/bin/env python3
"""SHORT1 actual Entry Confirmation at 20s from as-of V44 partial snapshots.

NO independent new MODEL event, no actual PENDING->READY, no Master,
no orders, Redis read-only. Uses actual 20-second snapshot, not future candle.
"""
import json,redis,csv,collections,math
from pathlib import Path
from core.trade_executor_short import TradeExecutorShort
from orchestration.context.short1_asof_market_join import (
    build_pending_20s_index,evaluate_pending_20s)

class CandleStub:
    def __init__(self,row):self.row=row
    def get_last_rows(self,symbol,interval):return [self.row,self.row]
class SnapshotExecutor:
    logger=None
    entry_candle_dir_interval='1m'
    short_current_reversal_min_pct=.0001
    _make_short_entry_decision=TradeExecutorShort._make_short_entry_decision
    _evaluate_short_entry_confirmation_stage=TradeExecutorShort._evaluate_short_entry_confirmation_stage
    def __init__(self,prev,current,bar):
        self.prev=prev
        self.current=current
        self.short_entry_candle_service=CandleStub(bar)
        self.bar=bar
    def _get_entry_candle_pair(self,*,symbol,interval):
        return self.prev,self.current
    def _get_entry_candle_current_row(self,*args,**kwargs):
        return self.bar

r=redis.Redis(decode_responses=True,socket_timeout=3)
v6=[json.loads(f['json']) for sid,f in r.xrevrange('telemetry:short1:inj:model_shadow_v6',count=480)]
v44=[json.loads(f['json']) for sid,f in r.xrevrange('telemetry:short1:inj:entry_hazard_orderbook_v44_shadow',count=2400)]
idx=build_pending_20s_index(v44)
rows=[]
for e in v6:
    gate=float(e['economic_probability'])>=float(e['reference_threshold'])
    if not gate:continue
    pending=evaluate_pending_20s(e,idx)
    item=dict(event_id=e['event_id'],model_ms=int(float(e['ts'])*1000),
         v6_score=float(e['economic_probability']),v6_threshold=float(e['reference_threshold']),
         original_confirmation=e['confirmation_state'],
         book_state=pending['state'],executor_entry_state='NOT_EVALUATED',
         reason='no_live_20s_snapshot',independent_fresh_model=False,master_authorized=False)
    if pending['state']=='PENDING_BOOK_EVIDENCE_AVAILABLE':
        # Re-identify V44 record by its publication millis; untrusted source
        # association to original V5 model event is checked where possible.
        minute=pending['minute_ms']
        match=[z for z in idx.get(minute,[]) if z[0]==pending['book_publish_ms']]
        if match:
            pub,source,ob,record=match[0]
            snap=record.get('snapshot_20s') or {}
            c=e.get('confirmation') or {}
            if snap.get('available') is True:
                try:
                    o=float(snap['open_price']); cl=float(snap['last_price'])
                    high=float(snap['high_price']); low=float(snap['low_price'])
                    p_open=float(c['prev_open']);p_close=float(c['prev_close'])
                    assert low<=min(o,cl)<=max(o,cl)<=high
                    assert int(c['current_open_time'])==minute
                    original_current_open=float(c['curr_open'])
                    # Same live 1m interval, no opportunistic re-timing.
                    assert abs(o-original_current_open)<.0000001
                    ex=SnapshotExecutor((p_open,p_close),(o,cl),[minute,o,high,low,cl])
                    progress=(pub-minute)/60000.0
                    score=float(e['economic_probability'])
                    # This stage may look up candle row via service; use prior
                    # runtime mock contract if the full helper is needed.
                    result=ex._evaluate_short_entry_confirmation_stage(
                        symbol='INJUSDT',
                        signal_score=score,
                        final_min_score=float(e['reference_threshold']),
                        progress=progress)
                    item.update(executor_entry_state=result.get('state'),
                        reason=result.get('reason'),progress=round(progress,4),
                        body_ratio=result.get('body_ratio'),
                        retrace_ratio=result.get('retrace_ratio'),
                        source_snapshot_asof_ms=pub)
                except (ValueError,TypeError,KeyError,AssertionError,AttributeError) as err:
                    item['reason']='insufficient_true_20s_snapshot:'+type(err).__name__
    rows.append(item)
summary=dict(score_pass=len(rows),
   fresh_20s_book=sum(x['book_state']=='PENDING_BOOK_EVIDENCE_AVAILABLE' for x in rows),
   actual_20s_entry_states=dict(collections.Counter(x['executor_entry_state'] for x in rows)),
   actual_20s_reasons=dict(collections.Counter(x['reason'] for x in rows)),
   independent_model_proven=0,
   master_authorized=0,
   risk_authorized=0,
   caution='20s entry confirmation recheck alone is NOT independent new model proof; timing is same 3m candle.')
root=Path('research/results')
(root/'SHORT1_INJ_V44_20S_ENTRY_CONFIRM_SUMMARY_20261114.json').write_text(json.dumps(summary,indent=2))
with (root/'SHORT1_INJ_V44_20S_ENTRY_CONFIRM_EVENTS_20261114.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
    w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
