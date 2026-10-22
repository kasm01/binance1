#!/usr/bin/env python3
from __future__ import annotations
import asyncio, json, os, sys, time, uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import websockets
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.telemetry.short1_entry_hazard_shadow_v1 import (
    MinuteTape, TradePoint, classify_20s, classify_30s,
    _floor_minute_ms, _next_minute_ms, _safe_float, _ef, _ei, _eb,
)
from orchestration.telemetry.short1_entry_hazard_outcome_ws_v1 import Tick, label_ticks

POLICY_SCHEMA='entry_hazard_policy_combined.v43'
OUTCOME_SCHEMA='entry_hazard_policy_outcome_combined.v43'

class CombinedV43:
    def __init__(self):
        self.symbol=str(os.getenv('SHORT1_HAZARD_SYMBOL','INJUSDT') or 'INJUSDT').upper()
        self.source_stream=str(os.getenv('SHORT1_HAZARD_SOURCE_STREAM','telemetry:short1:inj:model_shadow_v5'))
        self.policy_stream=str(os.getenv('SHORT1_HAZARD_V43_POLICY_STREAM','telemetry:short1:inj:entry_hazard_policy_v43_shadow'))
        self.outcome_stream=str(os.getenv('SHORT1_HAZARD_V43_OUTCOME_STREAM','telemetry:short1:inj:entry_hazard_policy_v43_outcomes_ws'))
        self.maxlen=max(1000,_ei('SHORT1_HAZARD_MAXLEN',20000))
        self.min_reversal=max(0.0,_ef('SHORT_CURRENT_REVERSAL_MIN_PCT',0.0001))
        self.flow20_max=_ef('SHORT1_HAZARD_FLOW20_MAX',-0.60)
        self.flow_delta_max=_ef('SHORT1_HAZARD_FLOW_DELTA_5_20_MAX',-0.30)
        self.hard_score=_ef('MASTER_MIN_TRADE_SCORE',0.540)
        self.soft_score=_ef('MASTER_SHORT_SOFT_SCORE_MIN',0.530)
        self.soft_score_enable=_eb('MASTER_SHORT_SOFT_SCORE_ENABLE',True)
        self.ws_url=str(os.getenv('SHORT1_HAZARD_WS_URL',f'wss://fstream.binance.com/market/ws/{self.symbol.lower()}@aggTrade'))
        self.bus=RedisBus(); self.tape:Optional[MinuteTape]=None; self.prev_candle={}; self.candidate={}
        self.last_source_id=''; self.valid_from_minute_ms=0; self.emitted20_for=0; self.emitted30_for=0; self.decision20={}
        self.ticks:Deque[Tick]=deque(); self.pending:Dict[str,Dict[str,Any]]={}

    def _load_candidate(self, minute_ms:int):
        for sid,obj in self.bus.xrevrange_json(self.source_stream,count=3):
            if sid==self.last_source_id and self.candidate: return
            if str(obj.get('symbol') or '').upper()!=self.symbol: continue
            event_minute=_floor_minute_ms(int(_safe_float(obj.get('ts'),0.0)*1000))
            if event_minute!=minute_ms: continue
            score=_safe_float(obj.get('economic_probability'),0.0)
            gate='READY' if score>=self.hard_score else ('PENDING_CONTEXT' if self.soft_score_enable and score>=self.soft_score else 'REJECT')
            conf=obj.get('confirmation') if isinstance(obj.get('confirmation'),dict) else {}
            self.candidate={'source_stream_id':sid,'source_event_id':str(obj.get('event_id') or ''),'score':score,
                'model_gate_shadow_state':gate,'hard_threshold':self.hard_score,'soft_threshold':self.soft_score,
                'source_confirmation_state':str(obj.get('confirmation_state') or ''),'prev_open':_safe_float(conf.get('prev_open')),
                'prev_close':_safe_float(conf.get('prev_close')),'source_curr_open':_safe_float(conf.get('curr_open')),
                'source_curr_close':_safe_float(conf.get('curr_close')),'minute_ms':event_minute}
            self.last_source_id=sid; return

    def _base(self,stage:str,minute_ms:int):
        return {'schema_version':POLICY_SCHEMA,'event_id':str(uuid.uuid4()),'ts':time.time(),'symbol':self.symbol,
            'minute_ms':minute_ms,'stage':stage,'routing':'OFF','policy_effect':'TELEMETRY_ONLY','candidate':dict(self.candidate),
            'prev_candle':dict(self.prev_candle),'research_contract':{'flow20_max':self.flow20_max,'flow_delta_5_20_max':self.flow_delta_max,
            'min_reversal_pct':self.min_reversal,'revalidate_sec':30}}

    def _emit_due(self,now_ms:int):
        if self.tape is None or not self.tape.valid_capture:return
        m=self.tape.minute_ms; self._load_candidate(m)
        if int(self.candidate.get('minute_ms') or 0)!=m:return
        sec=(now_ms-m)/1000.0
        prev_green=bool(_safe_float(self.prev_candle.get('close'))>_safe_float(self.prev_candle.get('open'))>0.0)
        if sec>=20 and self.emitted20_for!=m:
            s5=self.tape.snapshot(5); s20=self.tape.snapshot(20)
            d20=classify_20s(snap5=s5,snap20=s20,prev_green=prev_green,min_reversal_pct=self.min_reversal,flow20_max=self.flow20_max,flow_delta_max=self.flow_delta_max)
            rec=self._base('20S_HAZARD',m); rec.update({'state':d20.get('state'),'reason':d20.get('reason'),'snapshot_5s':s5,'snapshot_20s':s20,'decision':d20})
            sid=self.bus.xadd_json(self.policy_stream,rec,maxlen=self.maxlen); self.decision20=d20; self.emitted20_for=m
            print(f'[HAZARD-V43] 20s state={d20.get("state")} minute={m} sid={sid or "-"}',flush=True)
        if sec>=30 and self.emitted30_for!=m and self.emitted20_for==m:
            s20=self.tape.snapshot(20); s30=self.tape.snapshot(30); d30=classify_30s(decision20=self.decision20,snap20=s20,snap30=s30)
            rec=self._base('30S_REVALIDATION',m); rec.update({'state':d30.get('state'),'reason':d30.get('reason'),'snapshot_20s':s20,'snapshot_30s':s30,'decision20':dict(self.decision20),'decision':d30})
            sid=self.bus.xadd_json(self.policy_stream,rec,maxlen=self.maxlen); self.emitted30_for=m
            px=_safe_float(s30.get('last_price'),0.0)
            if px>0:
                self.pending[rec['event_id']]={'policy_event_id':rec['event_id'],'policy_stream_id':sid,'source_event_id':self.candidate.get('source_event_id'),
                    'policy_state':d30.get('state'),'model_gate_shadow_state':self.candidate.get('model_gate_shadow_state'),'score':self.candidate.get('score'),
                    'decision_ts_ms':int(rec['ts']*1000),'decision_price':px,'minute_ms':m,'snapshot_20s':s20,'snapshot_30s':s30}
            print(f'[HAZARD-V43] 30s state={d30.get("state")} minute={m} sid={sid or "-"}',flush=True)

    def _trim(self,now_ms:int):
        cut=now_ms-7*60_000
        while self.ticks and self.ticks[0].ts_ms<cut:self.ticks.popleft()

    def _finalize(self,now_ms:int):
        arr=list(self.ticks); done=[]
        for key,p in list(self.pending.items()):
            if now_ms<p['decision_ts_ms']+5*60_000:continue
            if not arr or arr[0].ts_ms>p['decision_ts_ms']+1000:
                done.append(key); continue
            labels={}
            for mins in (1,3,5):
                z=[x for x in arr if p['decision_ts_ms']<=x.ts_ms<=p['decision_ts_ms']+mins*60_000]
                labels[f'{mins}m']=label_ticks(z,p['decision_price'])
            out={'schema_version':OUTCOME_SCHEMA,'event_id':str(uuid.uuid4()),'ts':time.time(),'symbol':self.symbol,'routing':'OFF','policy_effect':'TELEMETRY_ONLY',**p,
                'labels_from_30s_decision':labels}
            rid=self.bus.xadd_json(self.outcome_stream,out,maxlen=self.maxlen)
            print(f'[HAZARD-V43] outcome state={p["policy_state"]} rid={rid or "-"} 5m={labels["5m"].get("touch_40bp")}',flush=True); done.append(key)
        for k in done:self.pending.pop(k,None)

    async def run_connection(self):
        async with websockets.connect(self.ws_url,ping_interval=20,ping_timeout=20,close_timeout=5,max_queue=4096) as ws:
            now_ms=int(time.time()*1000); self.valid_from_minute_ms=_next_minute_ms(now_ms); self.tape=None; self.prev_candle={}; self.candidate={}; self.decision20={}; self.pending.clear(); self.ticks.clear()
            print(f'[HAZARD-V43] connected symbol={self.symbol} valid_from={self.valid_from_minute_ms} policy={self.policy_stream} outcome={self.outcome_stream} routing=OFF',flush=True)
            while True:
                try:
                    raw=await asyncio.wait_for(ws.recv(),timeout=.25); obj=json.loads(raw)
                    ts_ms=int(obj.get('T') or obj.get('E') or 0); price=_safe_float(obj.get('p')); qty=_safe_float(obj.get('q'))
                    if ts_ms<=0 or price<=0 or qty<=0:continue
                    m=_floor_minute_ms(ts_ms)
                    if self.tape is None or self.tape.minute_ms!=m:
                        if self.tape is not None and self.tape.valid_capture and self.tape.trades:
                            self.prev_candle={'minute_ms':self.tape.minute_ms,'open':self.tape.trades[0].price,'close':self.tape.trades[-1].price,
                                'high':max(x.price for x in self.tape.trades),'low':min(x.price for x in self.tape.trades)}
                        self.tape=MinuteTape(minute_ms=m,valid_capture=bool(m>=self.valid_from_minute_ms)); self.candidate={}; self.decision20={}
                    self.tape.add(TradePoint(ts_ms=ts_ms,price=price,quote=price*qty,aggressive_buy=not bool(obj.get('m',False))))
                    self.ticks.append(Tick(ts_ms=ts_ms,price=price))
                except asyncio.TimeoutError:pass
                now_ms=int(time.time()*1000); self._trim(now_ms); self._emit_due(now_ms); self._finalize(now_ms)

    async def run(self):
        if not self.bus.ping(): raise SystemExit('HAZARD_V43: Redis unavailable')
        backoff=2.0
        while True:
            try: await self.run_connection(); backoff=2.0
            except asyncio.CancelledError: raise
            except Exception as e:
                print(f'[HAZARD-V43][WARN] {str(e)[:180]}',flush=True); await asyncio.sleep(backoff); backoff=min(30,backoff*1.7)

def selftest():
    # Imported research contracts already own detailed logic; verify imports and labels compose.
    m=1_800_000_000_000; tape=MinuteTape(m,True)
    for i in range(5):tape.add(TradePoint(m+i*800,100,10,True))
    for i in range(5,20):tape.add(TradePoint(m+i*800,99.9,20,False))
    d20=classify_20s(snap5=tape.snapshot(5),snap20=tape.snapshot(20),prev_green=True,min_reversal_pct=.0001,flow20_max=-.60,flow_delta_max=-.30)
    assert d20['state']=='PENDING'
    lab=label_ticks([Tick(0,100),Tick(1000,99.5)],100); assert lab['touch_40bp']=='FAVORABLE_FIRST'
    print('SELFTEST_OK')

if __name__=='__main__':
    if str(os.getenv('SHORT1_HAZARD_V43_SELFTEST','0')).lower() in ('1','true','yes','on'):selftest()
    else:asyncio.run(CombinedV43().run())
