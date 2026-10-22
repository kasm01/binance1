#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import websockets

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.event_bus.redis_bus import RedisBus

SCHEMA = "short1_entry_hazard_shadow.v1"


def _ef(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return float(default)


def _ei(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except Exception:
        return int(default)


def _eb(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _floor_minute_ms(ms: int) -> int:
    return (int(ms) // 60_000) * 60_000


def _next_minute_ms(ms: int) -> int:
    return _floor_minute_ms(ms) + 60_000


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


@dataclass
class TradePoint:
    ts_ms: int
    price: float
    quote: float
    aggressive_buy: bool


@dataclass
class MinuteTape:
    minute_ms: int
    valid_capture: bool
    trades: List[TradePoint] = field(default_factory=list)

    def add(self, p: TradePoint) -> None:
        self.trades.append(p)

    def snapshot(self, sec: float) -> Dict[str, Any]:
        cutoff = self.minute_ms + int(sec * 1000)
        rows = [x for x in self.trades if x.ts_ms < cutoff]
        if not rows:
            return {"available": False, "sec": float(sec), "reason": "no_trades"}
        buy = sum(x.quote for x in rows if x.aggressive_buy)
        sell = sum(x.quote for x in rows if not x.aggressive_buy)
        den = buy + sell
        prices = [x.price for x in rows]
        open_px = float(self.trades[0].price) if self.trades else float(prices[0])
        last_px = float(rows[-1].price)
        flow = (buy - sell) / den if den > 0 else 0.0
        ret = last_px / open_px - 1.0 if open_px > 0 else 0.0
        return {
            "available": True,
            "sec": float(sec),
            "trade_count": len(rows),
            "open_price": open_px,
            "last_price": last_px,
            "high_price": max(prices),
            "low_price": min(prices),
            "quote_volume": den,
            "aggressive_buy_quote": buy,
            "aggressive_sell_quote": sell,
            "flow": flow,
            "return_from_open": ret,
            "range_pct": (max(prices) - min(prices)) / open_px if open_px > 0 else 0.0,
        }


def classify_20s(
    *,
    snap5: Dict[str, Any],
    snap20: Dict[str, Any],
    prev_green: bool,
    min_reversal_pct: float,
    flow20_max: float,
    flow_delta_max: float,
) -> Dict[str, Any]:
    if not snap5.get("available") or not snap20.get("available"):
        return {"state": "INSUFFICIENT", "reason": "missing_snapshot"}
    ret20 = _safe_float(snap20.get("return_from_open"))
    reversal = max(0.0, -ret20)
    confirmed = bool(prev_green and reversal >= float(min_reversal_pct))
    f5 = _safe_float(snap5.get("flow"))
    f20 = _safe_float(snap20.get("flow"))
    delta = f20 - f5
    risk = bool(confirmed and f20 <= float(flow20_max) and delta <= float(flow_delta_max))
    if not confirmed:
        state = "NOT_APPLICABLE"
        reason = "entry_not_confirmed_20s"
    elif risk:
        state = "PENDING"
        reason = "extreme_sell_acceleration"
    else:
        state = "NORMAL"
        reason = "no_hazard_pattern"
    return {
        "state": state,
        "reason": reason,
        "confirmed_20s": confirmed,
        "reversal_pct_20s": reversal,
        "flow_5s": f5,
        "flow_20s": f20,
        "flow_delta_5_20": delta,
    }


def classify_30s(*, decision20: Dict[str, Any], snap20: Dict[str, Any], snap30: Dict[str, Any]) -> Dict[str, Any]:
    prior_state = str(decision20.get("state") or "")
    if prior_state == "NOT_APPLICABLE":
        return {"state": "NOT_APPLICABLE", "reason": "entry_not_confirmed_20s"}
    if prior_state != "PENDING":
        return {"state": "NORMAL", "reason": "no_pending_hazard"}
    if not snap20.get("available") or not snap30.get("available"):
        return {"state": "INSUFFICIENT", "reason": "missing_revalidation_snapshot"}
    r20 = _safe_float(snap20.get("return_from_open"))
    r30 = _safe_float(snap30.get("return_from_open"))
    delta = r30 - r20
    if delta < 0.0:
        state = "READY_AFTER_REVALIDATION"
        reason = "price_continued_down_20_30s"
    else:
        state = "REJECT_CANDIDATE"
        reason = "price_rebounded_20_30s"
    return {
        "state": state,
        "reason": reason,
        "ret_20s": r20,
        "ret_30s": r30,
        "ret_delta_20_30": delta,
        "flow_30s": _safe_float(snap30.get("flow")),
    }


class HazardShadow:
    def __init__(self) -> None:
        self.symbol = str(os.getenv("SHORT1_HAZARD_SYMBOL", "INJUSDT") or "INJUSDT").upper()
        self.source_stream = str(os.getenv("SHORT1_HAZARD_SOURCE_STREAM", "telemetry:short1:inj:model_shadow_v5"))
        self.out_stream = str(os.getenv("SHORT1_HAZARD_OUT_STREAM", "telemetry:short1:inj:entry_hazard_policy_v1"))
        self.maxlen = max(1000, _ei("SHORT1_HAZARD_MAXLEN", 20000))
        self.min_reversal = max(0.0, _ef("SHORT_CURRENT_REVERSAL_MIN_PCT", 0.0001))
        self.flow20_max = _ef("SHORT1_HAZARD_FLOW20_MAX", -0.60)
        self.flow_delta_max = _ef("SHORT1_HAZARD_FLOW_DELTA_5_20_MAX", -0.30)
        self.hard_score = _ef("MASTER_MIN_TRADE_SCORE", 0.540)
        self.soft_score = _ef("MASTER_SHORT_SOFT_SCORE_MIN", 0.530)
        self.soft_score_enable = _eb("MASTER_SHORT_SOFT_SCORE_ENABLE", True)
        self.ws_url = str(os.getenv("SHORT1_HAZARD_WS_URL", f"wss://fstream.binance.com/market/ws/{self.symbol.lower()}@aggTrade"))
        self.bus = RedisBus()
        self.tape: Optional[MinuteTape] = None
        self.valid_from_minute_ms = 0
        self.candidate: Dict[str, Any] = {}
        self.last_source_id = ""
        self.emitted20_for = 0
        self.emitted30_for = 0
        self.decision20: Dict[str, Any] = {}
        self.prev_candle: Dict[str, Any] = {}

    def _load_candidate(self, minute_ms: int) -> None:
        rows = self.bus.xrevrange_json(self.source_stream, count=3)
        for sid, obj in rows:
            if sid == self.last_source_id and self.candidate:
                return
            if str(obj.get("symbol") or "").upper() != self.symbol:
                continue
            conf = obj.get("confirmation") if isinstance(obj.get("confirmation"), dict) else {}
            event_minute = _floor_minute_ms(int(_safe_float(obj.get("ts"), 0.0) * 1000))
            if event_minute != minute_ms:
                continue
            score = _safe_float(obj.get("economic_probability"), 0.0)
            if score >= float(self.hard_score):
                gate_state = "READY"
            elif self.soft_score_enable and score >= float(self.soft_score):
                gate_state = "PENDING_CONTEXT"
            else:
                gate_state = "REJECT"
            self.candidate = {
                "source_stream_id": sid,
                "source_event_id": str(obj.get("event_id") or ""),
                "score": score,
                "model_gate_shadow_state": gate_state,
                "hard_threshold": float(self.hard_score),
                "soft_threshold": float(self.soft_score),
                "source_confirmation_state": str(obj.get("confirmation_state") or ""),
                "prev_open": _safe_float(conf.get("prev_open"), 0.0),
                "prev_close": _safe_float(conf.get("prev_close"), 0.0),
                "source_curr_open": _safe_float(conf.get("curr_open"), 0.0),
                "source_curr_close": _safe_float(conf.get("curr_close"), 0.0),
                "minute_ms": event_minute,
            }
            self.last_source_id = sid
            return

    def _base_record(self, stage: str, minute_ms: int) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "event_id": str(uuid.uuid4()),
            "ts": time.time(),
            "symbol": self.symbol,
            "minute_ms": int(minute_ms),
            "stage": stage,
            "routing": "OFF",
            "policy_effect": "TELEMETRY_ONLY",
            "research_contract": {
                "flow20_max": float(self.flow20_max),
                "flow_delta_5_20_max": float(self.flow_delta_max),
                "min_reversal_pct": float(self.min_reversal),
                "revalidate_sec": 30,
            },
            "candidate": dict(self.candidate),
            "prev_candle": dict(self.prev_candle),
        }

    def _emit_if_due(self, now_ms: int) -> None:
        if self.tape is None or not self.tape.valid_capture:
            return
        minute_ms = self.tape.minute_ms
        self._load_candidate(minute_ms)
        if int(self.candidate.get("minute_ms") or 0) != minute_ms:
            return
        sec = (now_ms - minute_ms) / 1000.0
        prev_green = bool(_safe_float(self.prev_candle.get("close")) > _safe_float(self.prev_candle.get("open")) > 0.0)

        if sec >= 20.0 and self.emitted20_for != minute_ms:
            s5 = self.tape.snapshot(5.0)
            s20 = self.tape.snapshot(20.0)
            d20 = classify_20s(
                snap5=s5,
                snap20=s20,
                prev_green=prev_green,
                min_reversal_pct=self.min_reversal,
                flow20_max=self.flow20_max,
                flow_delta_max=self.flow_delta_max,
            )
            rec = self._base_record("20S_HAZARD", minute_ms)
            rec.update({"state": d20.get("state"), "reason": d20.get("reason"), "snapshot_5s": s5, "snapshot_20s": s20, "decision": d20})
            sid = self.bus.xadd_json(self.out_stream, rec, maxlen=self.maxlen)
            self.decision20 = d20
            self.emitted20_for = minute_ms
            print(f"[ENTRY-HAZARD-SHADOW] stage=20s minute={minute_ms} state={d20.get('state')} score={self.candidate.get('score')} sid={sid or '-'} routing=OFF", flush=True)

        if sec >= 30.0 and self.emitted30_for != minute_ms and self.emitted20_for == minute_ms:
            s20 = self.tape.snapshot(20.0)
            s30 = self.tape.snapshot(30.0)
            d30 = classify_30s(decision20=self.decision20, snap20=s20, snap30=s30)
            rec = self._base_record("30S_REVALIDATION", minute_ms)
            rec.update({"state": d30.get("state"), "reason": d30.get("reason"), "snapshot_20s": s20, "snapshot_30s": s30, "decision20": dict(self.decision20), "decision": d30})
            sid = self.bus.xadd_json(self.out_stream, rec, maxlen=self.maxlen)
            self.emitted30_for = minute_ms
            print(f"[ENTRY-HAZARD-SHADOW] stage=30s minute={minute_ms} state={d30.get('state')} sid={sid or '-'} routing=OFF", flush=True)

    async def run_connection(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=5, max_queue=4096) as ws:
            now_ms = int(time.time() * 1000)
            self.valid_from_minute_ms = _next_minute_ms(now_ms)
            self.tape = None
            self.candidate = {}
            self.decision20 = {}
            print(f"[ENTRY-HAZARD-SHADOW] connected symbol={self.symbol} valid_from={self.valid_from_minute_ms} stream={self.out_stream} routing=OFF", flush=True)
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
                    obj = json.loads(raw)
                    ts_ms = int(obj.get("T") or obj.get("E") or 0)
                    price = _safe_float(obj.get("p"), 0.0)
                    qty = _safe_float(obj.get("q"), 0.0)
                    if ts_ms <= 0 or price <= 0 or qty <= 0:
                        continue
                    minute_ms = _floor_minute_ms(ts_ms)
                    if self.tape is None or self.tape.minute_ms != minute_ms:
                        if self.tape is not None and self.tape.valid_capture and self.tape.trades:
                            self.prev_candle = {
                                "minute_ms": int(self.tape.minute_ms),
                                "open": float(self.tape.trades[0].price),
                                "close": float(self.tape.trades[-1].price),
                                "high": float(max(x.price for x in self.tape.trades)),
                                "low": float(min(x.price for x in self.tape.trades)),
                            }
                        self.tape = MinuteTape(minute_ms=minute_ms, valid_capture=bool(minute_ms >= self.valid_from_minute_ms))
                        self.candidate = {}
                        self.decision20 = {}
                    self.tape.add(TradePoint(ts_ms=ts_ms, price=price, quote=price * qty, aggressive_buy=not bool(obj.get("m", False))))
                except asyncio.TimeoutError:
                    pass
                self._emit_if_due(int(time.time() * 1000))

    async def run(self) -> None:
        if not self.bus.ping():
            raise SystemExit("ENTRY_HAZARD_SHADOW: Redis unavailable")
        backoff = 2.0
        while True:
            try:
                await self.run_connection()
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ENTRY-HAZARD-SHADOW][WARN] reconnect err={str(exc)[:180]}", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.7)


def _selftest() -> None:
    m = 1_800_000_000_000
    tape = MinuteTape(m, True)
    # 5s mildly positive, by 20s strong accelerating sell, 30s branches tested below.
    for i in range(5): tape.add(TradePoint(m + i*800, 100.0, 10.0, True))
    for i in range(5, 20): tape.add(TradePoint(m + i*800, 99.9, 20.0, False))
    s5=tape.snapshot(5); s20=tape.snapshot(20)
    d20=classify_20s(snap5=s5,snap20=s20,prev_green=True,min_reversal_pct=.0001,flow20_max=-.60,flow_delta_max=-.30)
    assert d20['state']=='PENDING', (s5,s20,d20)
    s30=dict(s20); s30['return_from_open']=s20['return_from_open']-0.0005; s30['flow']=-0.7; s30['available']=True
    assert classify_30s(decision20=d20,snap20=s20,snap30=s30)['state']=='READY_AFTER_REVALIDATION'
    s30['return_from_open']=s20['return_from_open']+0.0002
    assert classify_30s(decision20=d20,snap20=s20,snap30=s30)['state']=='REJECT_CANDIDATE'
    print('SELFTEST_OK')


if __name__ == "__main__":
    if str(os.getenv("SHORT1_HAZARD_SELFTEST", "0")).lower() in ("1","true","yes","on"):
        _selftest()
    else:
        asyncio.run(HazardShadow().run())
