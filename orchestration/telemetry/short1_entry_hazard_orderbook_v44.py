#!/usr/bin/env python3
from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import websockets
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.telemetry.short1_entry_hazard_shadow_v1 import (
    MinuteTape, TradePoint, classify_20s, classify_30s,
    _floor_minute_ms, _next_minute_ms, _safe_float, _ef, _ei, _eb,
)
from orchestration.telemetry.short1_entry_hazard_outcome_ws_v1 import Tick, label_ticks

POLICY_SCHEMA='entry_hazard_orderbook_shadow.v44'
OUTCOME_SCHEMA='entry_hazard_orderbook_outcome.v44'

def _sum_qty(levels, n:int)->float:
    try:return sum(_safe_float(x[1]) for x in (levels or [])[:n])
    except Exception:return 0.0

def _ratio(a:float,b:float)->float:
    d=a+b
    return (a-b)/d if d>1e-15 else 0.0

def _book_shift_features(delta:Dict[str,Any])->Dict[str,Any]:
    if not isinstance(delta,dict) or not delta.get('available'):
        return {'available':False}
    comps={
        'l1':_safe_float(delta.get('l1_delta'),0.0),
        'depth5':_safe_float(delta.get('depth5_delta'),0.0),
        'depth20':_safe_float(delta.get('depth20_delta'),0.0),
        'microprice':_safe_float(delta.get('microprice_delta_bps'),0.0),
    }
    bullish=sum(1 for v in comps.values() if v>0)
    bearish=sum(1 for v in comps.values() if v<0)
    return {
        'available':True,
        'bullish_shift_count':bullish,
        'bearish_shift_count':bearish,
        'net_shift_count':bullish-bearish,
        'all_bullish':bullish==4,
        'majority_bullish':bullish>=3,
        'all_bearish':bearish==4,
        **{f'{k}_delta':v for k,v in comps.items()},
    }

def _absorption_features(trade_snap:Dict[str,Any], book_snap:Dict[str,Any], book_delta:Optional[Dict[str,Any]]=None)->Dict[str,Any]:
    flow=_safe_float(trade_snap.get('flow'),0.0); ret=_safe_float(trade_snap.get('return_from_open'),0.0)
    sell_pressure=max(0.0,-flow); down_bps=max(0.0,-ret)*1e4
    efficiency=(down_bps/max(sell_pressure,0.05)) if sell_pressure>0 else None
    d=book_delta or {}
    return {
        'sell_pressure':sell_pressure,
        'down_response_bps':down_bps,
        'down_bps_per_sell_flow':efficiency,
        'depth5_support':_safe_float(book_snap.get('depth5_imbalance'),0.0),
        'depth20_support':_safe_float(book_snap.get('depth20_imbalance'),0.0),
        'l1_support':_safe_float(book_snap.get('l1_imbalance'),0.0),
        'microprice_support_bps':_safe_float(book_snap.get('microprice_dev_bps'),0.0),
        'spread_bps':_safe_float(book_snap.get('spread_bps'),0.0),
        'bid5_replenishment_ratio':d.get('bid5_ratio'),
        'ask5_replenishment_ratio':d.get('ask5_ratio'),
        'depth5_delta':d.get('depth5_delta'),
        'depth20_delta':d.get('depth20_delta'),
        'microprice_delta_bps':d.get('microprice_delta_bps'),
    }


def _sign_flips(vals)->int:
    signs=[]
    for v in vals:
        x=_safe_float(v,0.0)
        sg=1 if x>0 else (-1 if x<0 else 0)
        if sg and (not signs or sg!=signs[-1]): signs.append(sg)
    return max(0,len(signs)-1)

def _trace_window_features(rows, start_ms:int, end_ms:int)->Dict[str,Any]:
    z=[r for r in rows if start_ms <= int(r.get('ts_ms') or 0) <= end_ms]
    if not z:return {'available':False,'n':0}
    out={'available':True,'n':len(z)}
    keys=['l1','depth5','depth20','micro','spread_bps','bid5','ask5']
    for k in keys:
        vals=np.array([_safe_float(r.get(k),np.nan) for r in z],dtype=float)
        vals=vals[np.isfinite(vals)]
        if not len(vals):continue
        out[f'{k}_mean']=float(vals.mean());out[f'{k}_min']=float(vals.min());out[f'{k}_max']=float(vals.max())
        out[f'{k}_first']=float(vals[0]);out[f'{k}_last']=float(vals[-1]);out[f'{k}_delta']=float(vals[-1]-vals[0])
        if k in ('l1','depth5','depth20','micro'):
            out[f'{k}_positive_frac']=float((vals>0).mean())
            out[f'{k}_negative_frac']=float((vals<0).mean())
            out[f'{k}_sign_flips']=int(_sign_flips(vals))
    bullish=[]
    for r in z:
        comps=[_safe_float(r.get('l1')), _safe_float(r.get('depth5')), _safe_float(r.get('depth20')), _safe_float(r.get('micro'))]
        bullish.append(sum(v>0 for v in comps)/4.0)
    out['bullish_persistence']=float(np.mean(bullish))
    out['bearish_persistence']=float(1.0-out['bullish_persistence'])
    return out

class OrderBookV44:
    def __init__(self):
        self.strategy_id=str(os.getenv('STRATEGY_ID','SHORT1') or 'SHORT1').strip().upper()
        self.strategy_ns=self.strategy_id.lower()
        self.symbol=str(os.getenv('STRATEGY_SYMBOL',os.getenv('SHORT1_HAZARD_SYMBOL','INJUSDT')) or 'INJUSDT').upper()
        low=self.symbol.lower()
        self.source_stream=str(os.getenv('STRATEGY_MODEL_SHADOW_STREAM',os.getenv('SHORT1_HAZARD_SOURCE_STREAM',f'telemetry:{self.strategy_ns}:{low}:model_shadow_v5')))
        self.policy_stream=str(os.getenv('STRATEGY_HAZARD_POLICY_STREAM',os.getenv('SHORT1_HAZARD_V44_POLICY_STREAM',f'telemetry:{self.strategy_ns}:{low}:entry_hazard_orderbook_v44_shadow')))
        self.outcome_stream=str(os.getenv('STRATEGY_HAZARD_OUTCOME_STREAM',os.getenv('SHORT1_HAZARD_V44_OUTCOME_STREAM',f'telemetry:{self.strategy_ns}:{low}:entry_hazard_orderbook_v44_outcomes')))
        self.maxlen=max(1000,_ei('SHORT1_HAZARD_MAXLEN',20000))
        self.min_reversal=max(0.0,_ef('SHORT_CURRENT_REVERSAL_MIN_PCT',0.0001))
        self.flow20_max=_ef('STRATEGY_HAZARD_FLOW20_MAX',_ef('SHORT1_HAZARD_FLOW20_MAX',-0.60))
        self.flow_delta_max=_ef('STRATEGY_HAZARD_FLOW_DELTA_5_20_MAX',_ef('SHORT1_HAZARD_FLOW_DELTA_5_20_MAX',-0.30))
        self.hard_score=_ef('MASTER_MIN_TRADE_SCORE',0.540)
        self.soft_score=_ef('MASTER_SHORT_SOFT_SCORE_MIN',0.530)
        self.soft_score_enable=_eb('MASTER_SHORT_SOFT_SCORE_ENABLE',True)
        self.ws_url=str(os.getenv('SHORT1_HAZARD_V44_WS_URL',
            f'wss://fstream.binance.com/public/stream?streams={low}@trade/{low}@depth20@100ms/{low}@bookTicker'))
        self.bus=RedisBus(); self.tape:Optional[MinuteTape]=None; self.prev_candle={}; self.candidate={}
        self.last_source_id=''; self.valid_from_minute_ms=0; self.emitted20_for=0; self.emitted30_for=0; self.decision20={}
        self.ticks:Deque[Tick]=deque(); self.pending:Dict[str,Dict[str,Any]]={}
        self.depth:Dict[str,Any]={}; self.book:Dict[str,Any]={}; self.book5={}; self.book20={}; self.book30={}; self.book_trace:Deque[Dict[str,Any]]=deque()

    def _load_candidate(self, minute_ms:int):
        for sid,obj in self.bus.xrevrange_json(self.source_stream,count=3):
            if sid==self.last_source_id and self.candidate:return
            if str(obj.get('symbol') or '').upper()!=self.symbol:continue
            event_minute=_floor_minute_ms(int(_safe_float(obj.get('ts'),0.0)*1000))
            if event_minute!=minute_ms:continue
            score=_safe_float(obj.get('economic_probability'),0.0)
            gate='READY' if score>=self.hard_score else ('PENDING_CONTEXT' if self.soft_score_enable and score>=self.soft_score else 'REJECT')
            conf=obj.get('confirmation') if isinstance(obj.get('confirmation'),dict) else {}
            self.candidate={'source_stream_id':sid,'source_event_id':str(obj.get('event_id') or ''),'score':score,
                'model_gate_shadow_state':gate,'hard_threshold':self.hard_score,'soft_threshold':self.soft_score,
                'source_confirmation_state':str(obj.get('confirmation_state') or ''),'prev_open':_safe_float(conf.get('prev_open')),
                'prev_close':_safe_float(conf.get('prev_close')),'source_curr_open':_safe_float(conf.get('curr_open')),
                'source_curr_close':_safe_float(conf.get('curr_close')),'minute_ms':event_minute}
            self.last_source_id=sid;return

    def _book_snapshot(self, now_ms:int)->Dict[str,Any]:
        bids=self.depth.get('b') or []; asks=self.depth.get('a') or []
        bid=_safe_float(self.book.get('b') or (bids[0][0] if bids else 0)); ask=_safe_float(self.book.get('a') or (asks[0][0] if asks else 0))
        bq=_safe_float(self.book.get('B') or (bids[0][1] if bids else 0)); aq=_safe_float(self.book.get('A') or (asks[0][1] if asks else 0))
        mid=(bid+ask)/2 if bid>0 and ask>0 else 0.0
        bid5=_sum_qty(bids,5); ask5=_sum_qty(asks,5); bid20=_sum_qty(bids,20); ask20=_sum_qty(asks,20)
        micro=((ask*bq + bid*aq)/(bq+aq)) if mid>0 and bq+aq>0 else mid
        depth_ts=max(int(self.depth.get('E') or self.depth.get('T') or 0), int(self.book.get('E') or 0))
        return {
            'available':bool(mid>0 and bids and asks),'observed_at_ms':now_ms,'source_event_ms':depth_ts,
            'source_age_ms':max(0,now_ms-depth_ts) if depth_ts else None,
            'best_bid':bid,'best_ask':ask,'bid_qty_l1':bq,'ask_qty_l1':aq,'mid':mid,
            'spread_bps':((ask-bid)/mid*1e4) if mid>0 else 0.0,
            'microprice':micro,'microprice_dev_bps':((micro-mid)/mid*1e4) if mid>0 else 0.0,
            'l1_imbalance':_ratio(bq,aq),'bid_qty_5':bid5,'ask_qty_5':ask5,'depth5_imbalance':_ratio(bid5,ask5),
            'bid_qty_20':bid20,'ask_qty_20':ask20,'depth20_imbalance':_ratio(bid20,ask20),
        }

    def _book_deltas(self,a:Dict[str,Any],b:Dict[str,Any])->Dict[str,Any]:
        if not a.get('available') or not b.get('available'):return {'available':False}
        return {'available':True,
            'l1_delta':_safe_float(b.get('l1_imbalance'))-_safe_float(a.get('l1_imbalance')),
            'depth5_delta':_safe_float(b.get('depth5_imbalance'))-_safe_float(a.get('depth5_imbalance')),
            'depth20_delta':_safe_float(b.get('depth20_imbalance'))-_safe_float(a.get('depth20_imbalance')),
            'spread_delta_bps':_safe_float(b.get('spread_bps'))-_safe_float(a.get('spread_bps')),
            'microprice_delta_bps':_safe_float(b.get('microprice_dev_bps'))-_safe_float(a.get('microprice_dev_bps')),
            'bid5_ratio':(_safe_float(b.get('bid_qty_5'))/_safe_float(a.get('bid_qty_5'))) if _safe_float(a.get('bid_qty_5'))>0 else None,
            'ask5_ratio':(_safe_float(b.get('ask_qty_5'))/_safe_float(a.get('ask_qty_5'))) if _safe_float(a.get('ask_qty_5'))>0 else None,
        }

    def _trace_features(self,start_sec:float,end_sec:float)->Dict[str,Any]:
        if self.tape is None:return {'available':False,'n':0}
        a=int(self.tape.minute_ms+start_sec*1000); b=int(self.tape.minute_ms+end_sec*1000)
        return _trace_window_features(list(self.book_trace),a,b)

    def _base(self,stage:str,m:int):
        return {'schema_version':POLICY_SCHEMA,'event_id':str(uuid.uuid4()),'strategy_id':self.strategy_id,'ts':time.time(),'symbol':self.symbol,'minute_ms':m,
            'stage':stage,'routing':'OFF','policy_effect':'TELEMETRY_ONLY','candidate':dict(self.candidate),'prev_candle':dict(self.prev_candle),
            'research_contract':{'flow20_max':self.flow20_max,'flow_delta_5_20_max':self.flow_delta_max,'min_reversal_pct':self.min_reversal,
                                 'revalidate_sec':30,'orderbook_effect':'TELEMETRY_ONLY'}}

    def _emit_due(self,now_ms:int):
        if self.tape is None or not self.tape.valid_capture:return
        m=self.tape.minute_ms; self._load_candidate(m)
        if int(self.candidate.get('minute_ms') or 0)!=m:return
        sec=(now_ms-m)/1000.0
        prev_green=bool(_safe_float(self.prev_candle.get('close'))>_safe_float(self.prev_candle.get('open'))>0.0)
        if sec>=5 and not self.book5:self.book5=self._book_snapshot(now_ms)
        if sec>=20 and self.emitted20_for!=m:
            s5=self.tape.snapshot(5); s20=self.tape.snapshot(20); self.book20=self._book_snapshot(now_ms)
            d20=classify_20s(snap5=s5,snap20=s20,prev_green=prev_green,min_reversal_pct=self.min_reversal,flow20_max=self.flow20_max,flow_delta_max=self.flow_delta_max)
            rec=self._base('20S_HAZARD',m); rec.update({'state':d20.get('state'),'reason':d20.get('reason'),'snapshot_5s':s5,'snapshot_20s':s20,
                'decision':d20,'orderbook_5s':dict(self.book5),'orderbook_20s':dict(self.book20),'orderbook_delta_5_20':self._book_deltas(self.book5,self.book20),
                'absorption_features_20s':_absorption_features(s20,self.book20,self._book_deltas(self.book5,self.book20)),
                'orderbook_trace_0_20':self._trace_features(0,20)})
            sid=self.bus.xadd_json(self.policy_stream,rec,maxlen=self.maxlen); self.decision20=d20; self.emitted20_for=m
            print(f'[HAZARD-V44] 20s state={d20.get("state")} f20={d20.get("flow_20s")} d5={self.book20.get("depth5_imbalance")} micro={self.book20.get("microprice_dev_bps")} sid={sid or "-"}',flush=True)
        if sec>=30 and self.emitted30_for!=m and self.emitted20_for==m:
            s20=self.tape.snapshot(20); s30=self.tape.snapshot(30); self.book30=self._book_snapshot(now_ms)
            d30=classify_30s(decision20=self.decision20,snap20=s20,snap30=s30)
            rec=self._base('30S_REVALIDATION',m); rec.update({'state':d30.get('state'),'reason':d30.get('reason'),'snapshot_20s':s20,'snapshot_30s':s30,
                'decision20':dict(self.decision20),'decision':d30,'orderbook_20s':dict(self.book20),'orderbook_30s':dict(self.book30),
                'orderbook_delta_20_30':self._book_deltas(self.book20,self.book30),
                'book_shift_20_30':_book_shift_features(self._book_deltas(self.book20,self.book30)),
                'absorption_features_30s':_absorption_features(s30,self.book30,self._book_deltas(self.book20,self.book30)),
                'orderbook_trace_0_20':self._trace_features(0,20),'orderbook_trace_20_30':self._trace_features(20,30)})
            sid=self.bus.xadd_json(self.policy_stream,rec,maxlen=self.maxlen); self.emitted30_for=m
            px=_safe_float(s30.get('last_price'),0.0)
            if px>0:self.pending[rec['event_id']]={'policy_event_id':rec['event_id'],'policy_stream_id':sid,'source_event_id':self.candidate.get('source_event_id'),
                'policy_state':d30.get('state'),'model_gate_shadow_state':self.candidate.get('model_gate_shadow_state'),'score':self.candidate.get('score'),
                'decision_ts_ms':int(rec['ts']*1000),'decision_price':px,'minute_ms':m,'snapshot_20s':s20,'snapshot_30s':s30,
                'orderbook_5s':dict(self.book5),'orderbook_20s':dict(self.book20),'orderbook_30s':dict(self.book30),
                'orderbook_delta_5_20':self._book_deltas(self.book5,self.book20),'orderbook_delta_20_30':self._book_deltas(self.book20,self.book30),
                'book_shift_20_30':_book_shift_features(self._book_deltas(self.book20,self.book30)),
                'orderbook_trace_0_20':self._trace_features(0,20),'orderbook_trace_20_30':self._trace_features(20,30),
                'absorption_features_20s':_absorption_features(s20,self.book20,self._book_deltas(self.book5,self.book20)),
                'absorption_features_30s':_absorption_features(s30,self.book30,self._book_deltas(self.book20,self.book30))}
            print(f'[HAZARD-V44] 30s state={d30.get("state")} d5={self.book30.get("depth5_imbalance")} micro={self.book30.get("microprice_dev_bps")} sid={sid or "-"}',flush=True)

    def _trim(self,now_ms:int):
        cut=now_ms-7*60_000
        while self.ticks and self.ticks[0].ts_ms<cut:self.ticks.popleft()

    def _finalize(self,now_ms:int):
        arr=list(self.ticks);done=[]
        for key,p in list(self.pending.items()):
            if now_ms<p['decision_ts_ms']+5*60_000:continue
            if not arr or arr[0].ts_ms>p['decision_ts_ms']+1000:done.append(key);continue
            labels={}
            for mins in (1,3,5):
                z=[x for x in arr if p['decision_ts_ms']<=x.ts_ms<=p['decision_ts_ms']+mins*60_000]
                labels[f'{mins}m']=label_ticks(z,p['decision_price'])
            out={'schema_version':OUTCOME_SCHEMA,'event_id':str(uuid.uuid4()),'strategy_id':self.strategy_id,'ts':time.time(),'symbol':self.symbol,'routing':'OFF','policy_effect':'TELEMETRY_ONLY',**p,
                'labels_from_30s_decision':labels}
            rid=self.bus.xadd_json(self.outcome_stream,out,maxlen=self.maxlen);done.append(key)
            print(f'[HAZARD-V44] outcome state={p["policy_state"]} rid={rid or "-"} 5m={labels["5m"].get("touch_40bp")}',flush=True)
        for k in done:self.pending.pop(k,None)

    async def run_connection(self):
        async with websockets.connect(self.ws_url,ping_interval=20,ping_timeout=20,close_timeout=5,max_queue=32768) as ws:
            now_ms=int(time.time()*1000);self.valid_from_minute_ms=_next_minute_ms(now_ms);self.tape=None;self.prev_candle={};self.candidate={};self.decision20={};self.pending.clear();self.ticks.clear();self.book5={};self.book20={};self.book30={};self.book_trace.clear()
            print(f'[HAZARD-V44] connected strategy={self.strategy_id} symbol={self.symbol} valid_from={self.valid_from_minute_ms} policy={self.policy_stream} outcome={self.outcome_stream} routing=OFF',flush=True)
            while True:
                try:
                    raw=await asyncio.wait_for(ws.recv(),timeout=.25);wrap=json.loads(raw);stream=str(wrap.get('stream') or '');obj=wrap.get('data') or {}
                    if '@depth20' in stream:
                        self.depth=obj
                        if self.tape is not None and self.book:
                            ts=int(obj.get('T') or obj.get('E') or int(time.time()*1000))
                            bs=self._book_snapshot(int(time.time()*1000))
                            if bs.get('available'):
                                self.book_trace.append({'ts_ms':ts,'l1':bs.get('l1_imbalance'),'depth5':bs.get('depth5_imbalance'),'depth20':bs.get('depth20_imbalance'),'micro':bs.get('microprice_dev_bps'),'spread_bps':bs.get('spread_bps'),'bid5':bs.get('bid_qty_5'),'ask5':bs.get('ask_qty_5')})
                                cut=ts-65000
                                while self.book_trace and int(self.book_trace[0].get('ts_ms') or 0)<cut:self.book_trace.popleft()
                    elif '@bookTicker' in stream:self.book=obj
                    elif '@trade' in stream:
                        ts_ms=int(obj.get('T') or obj.get('E') or 0);price=_safe_float(obj.get('p'));qty=_safe_float(obj.get('q'))
                        if ts_ms>0 and price>0 and qty>0:
                            m=_floor_minute_ms(ts_ms)
                            if self.tape is None or self.tape.minute_ms!=m:
                                if self.tape is not None and self.tape.valid_capture and self.tape.trades:
                                    self.prev_candle={'minute_ms':self.tape.minute_ms,'open':self.tape.trades[0].price,'close':self.tape.trades[-1].price,
                                        'high':max(x.price for x in self.tape.trades),'low':min(x.price for x in self.tape.trades)}
                                self.tape=MinuteTape(minute_ms=m,valid_capture=bool(m>=self.valid_from_minute_ms));self.candidate={};self.decision20={};self.book5={};self.book20={};self.book30={};self.book_trace.clear()
                            self.tape.add(TradePoint(ts_ms=ts_ms,price=price,quote=price*qty,aggressive_buy=not bool(obj.get('m',False))))
                            self.ticks.append(Tick(ts_ms=ts_ms,price=price))
                except asyncio.TimeoutError:pass
                now_ms=int(time.time()*1000);self._trim(now_ms);self._emit_due(now_ms);self._finalize(now_ms)

    async def run(self):
        if not self.bus.ping():raise SystemExit('HAZARD_V44: Redis unavailable')
        backoff=2.0
        while True:
            try:await self.run_connection();backoff=2.0
            except asyncio.CancelledError:raise
            except Exception as e:
                print(f'[HAZARD-V44][WARN] {str(e)[:180]}',flush=True);await asyncio.sleep(backoff);backoff=min(30,backoff*1.7)

def selftest():
    m=1_800_000_000_000;tape=MinuteTape(m,True)
    for i in range(5):tape.add(TradePoint(m+i*800,100,10,True))
    for i in range(5,20):tape.add(TradePoint(m+i*800,99.9,20,False))
    d=classify_20s(snap5=tape.snapshot(5),snap20=tape.snapshot(20),prev_green=True,min_reversal_pct=.0001,flow20_max=-.60,flow_delta_max=-.30)
    assert d['state']=='PENDING'
    assert abs(_ratio(10,5)-1/3)<1e-12
    print('SELFTEST_OK')

if __name__=='__main__':
    if str(os.getenv('SHORT1_HAZARD_V44_SELFTEST','0')).lower() in ('1','true','yes','on'):selftest()
    else:asyncio.run(OrderBookV44().run())
