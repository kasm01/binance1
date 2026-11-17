"""Isolated fail-closed reference: LONG1+SHORT1 reservation lifecycle.

NOT connected to live Redis/Exchange, and not a production risk engine.
Callers MUST serialize the complete transaction under a single atomic state
owner (Redis Lua, DB transaction, or exclusive lock with lease/fencing) before
using this algorithm across processes. Only dry-run testing permitted.
"""
from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
from typing import Dict
import math

@dataclass
class Reservation:
    request_id:str
    strategy_id:str
    symbol:str
    side:str
    notional:float
    created_at_ms:int
    expires_at_ms:int
    state:str='RESERVED'
    position_id:str=''
    # Tombstones kept as part of request journal (not free for reuse).
class DryReservationReference:
    def __init__(self,*,equity:float,max_total_ratio:float=1.0,
                 max_per_side_ratio:float=.5,max_request_ratio:float=.46,
                 ttl_ms:int=21600000):
        if not math.isfinite(equity) or equity<=0:raise ValueError('invalid_equity')
        self.equity=equity
        self.cap=equity*max_total_ratio
        self.side_cap=equity*max_per_side_ratio
        self.order_cap=equity*max_request_ratio
        self.ttl_ms=ttl_ms
        self.active:Dict[str,Reservation]={}
        self.journal:Dict[str,dict]={}
        self.positions:Dict[str,Reservation]={}
        self.lock=RLock()
    def _reply(self,request_id,state,reason,**kwargs):
        return dict(request_id=request_id,state=state,reason=reason,**kwargs)
    def reserve(self,*,request_id,strategy_id,symbol,side,now_ms):
        with self.lock:
            payload=(strategy_id,symbol,side)
            cached=self.journal.get(request_id)
            if cached:
                if cached['payload'] != payload:
                    return self._reply(request_id,'REJECT','request_id_payload_conflict')
                return dict(cached['reply'],replayed=True)
            if not request_id or strategy_id not in ('LONG1','SHORT1') or side not in ('long','short'):
                result=self._reply(request_id,'REJECT','invalid_request')
            elif symbol!=('NEARUSDT' if strategy_id=='LONG1' else 'INJUSDT'):
                result=self._reply(request_id,'REJECT','strategy_symbol_mismatch')
            elif strategy_id in self.active or strategy_id in self.positions:
                result=self._reply(request_id,'REJECT','strategy_slot_busy')
            elif any(pos.symbol==symbol for pos in (*self.active.values(),*self.positions.values())):
                result=self._reply(request_id,'REJECT','symbol_slot_busy')
            else:
                reserved=sum(x.notional for x in (*self.active.values(),*self.positions.values()))
                side_used=sum(x.notional for x in (*self.active.values(),*self.positions.values()) if x.side==side)
                allowed=min(self.order_cap,self.cap-reserved,self.side_cap-side_used)
                if allowed<10:
                    result=self._reply(request_id,'REJECT','insufficient_headroom')
                else:
                    res=Reservation(request_id,strategy_id,symbol,side,allowed,now_ms,
                                    now_ms+self.ttl_ms)
                    self.active[strategy_id]=res
                    result=self._reply(request_id,'READY','reserved',
                                       reservation_token=request_id,allowed_notional=allowed)
            self.journal[request_id]=dict(payload=payload,reply=result)
            return result
    def commit(self,*,strategy_id,request_id,position_id,now_ms):
        with self.lock:
            res=self.active.get(strategy_id)
            if res is None:
                # Repeated commit is idempotent if same owner and exact position.
                old=self.positions.get(strategy_id)
                if old and old.request_id==request_id and old.position_id==position_id:
                    return self._reply(request_id,'READY','already_committed')
                return self._reply(request_id,'REJECT','reservation_not_found')
            if res.request_id!=request_id:
                return self._reply(request_id,'REJECT','reservation_owner_mismatch')
            if not position_id or now_ms>res.expires_at_ms:
                return self._reply(request_id,'REJECT','expired_or_missing_position')
            res.state='COMMITTED';res.position_id=position_id
            self.positions[strategy_id]=res
            del self.active[strategy_id]
            return self._reply(request_id,'READY','committed')
    def release(self,*,strategy_id,request_id,position_id='',flat_verified=False):
        with self.lock:
            res=self.active.get(strategy_id)
            if res is not None:
                if res.request_id!=request_id:
                    return self._reply(request_id,'REJECT','reservation_owner_mismatch')
                del self.active[strategy_id]
                return self._reply(request_id,'READY','cancelled_reservation')
            pos=self.positions.get(strategy_id)
            if pos is None:
                return self._reply(request_id,'READY','already_released')
            if pos.request_id!=request_id or pos.position_id!=position_id:
                return self._reply(request_id,'REJECT','position_owner_mismatch')
            if not flat_verified:
                return self._reply(request_id,'REJECT','exchange_flat_not_verified')
            del self.positions[strategy_id]
            return self._reply(request_id,'READY','released_confirmed_flat')
    def expire(self,now_ms):
        """Never silently expire COMMITTED. For uncommitted, fail CLOSED.

        Without exchange open order/position reconciliation, a TTL does NOT
        establish that an in-flight order failed. Keep slots blocked and
        require a separate authoritative reconcile before any release.
        """
        with self.lock:
            return [x.request_id for x in self.active.values() if now_ms>x.expires_at_ms]
