#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import websockets
from orchestration.event_bus.redis_bus import RedisBus

SCHEMA = "entry_hazard_policy_outcome_ws.v1"


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _i(v: Any, d: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(d)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except Exception:
        return int(default)


@dataclass
class Tick:
    ts_ms: int
    price: float


@dataclass
class Pending:
    policy_stream_id: str
    policy_event_id: str
    source_event_id: str
    state: str
    model_gate_shadow_state: str
    decision_ts_ms: int
    decision_price: float
    minute_ms: int
    score: float
    snapshot_20s: Dict[str, Any]
    snapshot_30s: Dict[str, Any]


def label_ticks(ticks: List[Tick], entry_price: float) -> Dict[str, Any]:
    if not ticks or entry_price <= 0:
        return {"available": False, "reason": "no_ticks"}
    highs = [x.price for x in ticks]
    lows = highs
    mae = max(p / entry_price - 1.0 for p in highs)
    mfe = max(1.0 - p / entry_price for p in lows)
    adv_idx: Optional[int] = None
    fav_idx: Optional[int] = None
    for idx, x in enumerate(ticks):
        if adv_idx is None and x.price / entry_price - 1.0 >= 0.004:
            adv_idx = idx
        if fav_idx is None and 1.0 - x.price / entry_price >= 0.004:
            fav_idx = idx
        if adv_idx is not None and fav_idx is not None:
            break
    if adv_idx is None and fav_idx is None:
        touch = "NEITHER"
    elif adv_idx is None:
        touch = "FAVORABLE_FIRST"
    elif fav_idx is None:
        touch = "ADVERSE_FIRST"
    elif adv_idx < fav_idx:
        touch = "ADVERSE_FIRST"
    elif fav_idx < adv_idx:
        touch = "FAVORABLE_FIRST"
    else:
        touch = "BOTH_SAME_TICK"
    return {
        "available": True,
        "tick_count": len(ticks),
        "mae_pct": float(mae),
        "mfe_pct": float(mfe),
        "touch_40bp": touch,
        "last_price": float(ticks[-1].price),
    }


class OutcomeWS:
    def __init__(self) -> None:
        self.symbol = str(os.getenv("SHORT1_HAZARD_SYMBOL", "INJUSDT") or "INJUSDT").upper()
        self.policy_stream = str(os.getenv("SHORT1_HAZARD_POLICY_STREAM", "telemetry:short1:inj:entry_hazard_policy_v1"))
        self.out_stream = str(os.getenv("SHORT1_HAZARD_POLICY_OUTCOME_STREAM", "telemetry:short1:inj:entry_hazard_policy_v1_outcomes_ws"))
        self.done_set = str(os.getenv("SHORT1_HAZARD_POLICY_OUTCOME_DONE", "telemetry:short1:inj:entry_hazard_policy_v1:outcomes_ws_done"))
        self.maxlen = max(1000, _env_int("SHORT1_HAZARD_POLICY_OUTCOME_MAXLEN", 20000))
        self.ws_url = str(os.getenv("SHORT1_HAZARD_WS_URL", f"wss://fstream.binance.com/market/ws/{self.symbol.lower()}@aggTrade"))
        self.bus = RedisBus()
        self.ticks: Deque[Tick] = deque()
        self.pending: Dict[str, Pending] = {}
        self.last_policy_id = "0-0"
        self.valid_from_ms = 0
        self.last_policy_poll = 0.0

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - 7 * 60_000
        while self.ticks and self.ticks[0].ts_ms < cutoff:
            self.ticks.popleft()

    def _poll_policy(self, now_ms: int) -> None:
        if time.time() - self.last_policy_poll < 0.5:
            return
        self.last_policy_poll = time.time()
        try:
            rows = self.bus.r.xrange(self.policy_stream, min=f"({self.last_policy_id}", max="+", count=100)
        except Exception:
            return
        for sid, fields in rows:
            self.last_policy_id = sid
            try:
                obj = json.loads((fields or {}).get("json") or "{}")
            except Exception:
                continue
            if str(obj.get("stage") or "") != "30S_REVALIDATION":
                continue
            policy_event_id = str(obj.get("event_id") or "")
            if not policy_event_id or self.bus.r.sismember(self.done_set, policy_event_id):
                continue
            snap30 = obj.get("snapshot_30s") if isinstance(obj.get("snapshot_30s"), dict) else {}
            snap20 = obj.get("snapshot_20s") if isinstance(obj.get("snapshot_20s"), dict) else {}
            px = _f(snap30.get("last_price"), 0.0)
            if px <= 0.0:
                continue
            event_ts_ms = int(_f(obj.get("ts"), 0.0) * 1000)
            # We must already have been connected before the policy decision.
            if event_ts_ms < self.valid_from_ms:
                continue
            candidate = obj.get("candidate") if isinstance(obj.get("candidate"), dict) else {}
            p = Pending(
                policy_stream_id=sid,
                policy_event_id=policy_event_id,
                source_event_id=str(candidate.get("source_event_id") or ""),
                state=str(obj.get("state") or ""),
                model_gate_shadow_state=str(candidate.get("model_gate_shadow_state") or "UNKNOWN"),
                decision_ts_ms=event_ts_ms,
                decision_price=px,
                minute_ms=_i(obj.get("minute_ms"), 0),
                score=_f(candidate.get("score"), 0.0),
                snapshot_20s=snap20,
                snapshot_30s=snap30,
            )
            self.pending[policy_event_id] = p
            print(f"[ENTRY-HAZARD-OUTCOME-WS] track state={p.state} gate={p.model_gate_shadow_state} px={p.decision_price:.6f} policy={sid}", flush=True)

    def _finalize_due(self, now_ms: int) -> None:
        done: List[str] = []
        all_ticks = list(self.ticks)
        for key, p in list(self.pending.items()):
            if now_ms < p.decision_ts_ms + 5 * 60_000:
                continue
            # Fail closed if tape does not cover the decision instant through 5m.
            if not all_ticks or all_ticks[0].ts_ms > p.decision_ts_ms + 1000:
                print(f"[ENTRY-HAZARD-OUTCOME-WS][SKIP] incomplete_start policy={p.policy_stream_id}", flush=True)
                done.append(key)
                continue
            labels: Dict[str, Any] = {}
            for mins in (1, 3, 5):
                end_ms = p.decision_ts_ms + mins * 60_000
                z = [x for x in all_ticks if p.decision_ts_ms <= x.ts_ms <= end_ms]
                labels[f"{mins}m"] = label_ticks(z, p.decision_price)
            out = {
                "schema_version": SCHEMA,
                "event_id": str(uuid.uuid4()),
                "ts": time.time(),
                "symbol": self.symbol,
                "routing": "OFF",
                "policy_effect": "TELEMETRY_ONLY",
                "policy_stream_id": p.policy_stream_id,
                "policy_event_id": p.policy_event_id,
                "source_event_id": p.source_event_id,
                "policy_state": p.state,
                "model_gate_shadow_state": p.model_gate_shadow_state,
                "economic_probability": p.score,
                "decision_ts_ms": p.decision_ts_ms,
                "decision_price_30s": p.decision_price,
                "minute_ms": p.minute_ms,
                "snapshot_20s": p.snapshot_20s,
                "snapshot_30s": p.snapshot_30s,
                "labels_from_30s_decision": labels,
            }
            rid = self.bus.xadd_json(self.out_stream, out, maxlen=self.maxlen)
            if rid:
                self.bus.r.sadd(self.done_set, p.policy_event_id)
                print(f"[ENTRY-HAZARD-OUTCOME-WS] finalized state={p.state} rid={rid} 5m={labels['5m'].get('touch_40bp')} routing=OFF", flush=True)
            done.append(key)
        for key in done:
            self.pending.pop(key, None)

    async def run_connection(self) -> None:
        async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=5, max_queue=4096) as ws:
            self.valid_from_ms = int(time.time() * 1000)
            # Start at stream tail: only policy events created while this collector is live.
            tail = self.bus.r.xrevrange(self.policy_stream, count=1)
            self.last_policy_id = tail[0][0] if tail else "0-0"
            self.ticks.clear()
            self.pending.clear()
            print(f"[ENTRY-HAZARD-OUTCOME-WS] connected symbol={self.symbol} valid_from={self.valid_from_ms} policy={self.policy_stream} out={self.out_stream} routing=OFF", flush=True)
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.25)
                    obj = json.loads(raw)
                    ts_ms = _i(obj.get("T") or obj.get("E"), 0)
                    price = _f(obj.get("p"), 0.0)
                    if ts_ms > 0 and price > 0:
                        self.ticks.append(Tick(ts_ms=ts_ms, price=price))
                except asyncio.TimeoutError:
                    pass
                now_ms = int(time.time() * 1000)
                self._trim(now_ms)
                self._poll_policy(now_ms)
                self._finalize_due(now_ms)

    async def run(self) -> None:
        if not self.bus.ping():
            raise SystemExit("ENTRY_HAZARD_OUTCOME_WS: Redis unavailable")
        backoff = 2.0
        while True:
            try:
                await self.run_connection()
                backoff = 2.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[ENTRY-HAZARD-OUTCOME-WS][WARN] reconnect err={str(exc)[:180]}", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.7)


def _selftest() -> None:
    e = 100.0
    ticks = [Tick(ts_ms=i*1000, price=100.0) for i in range(10)]
    ticks += [Tick(ts_ms=10_000+i*1000, price=99.5) for i in range(10)]
    r = label_ticks(ticks, e)
    assert r["available"] and r["touch_40bp"] == "FAVORABLE_FIRST", r
    ticks2 = [Tick(ts_ms=i*1000, price=100.5) for i in range(10)]
    r2 = label_ticks(ticks2, e)
    assert r2["touch_40bp"] == "ADVERSE_FIRST", r2
    print("SELFTEST_OK")


if __name__ == "__main__":
    if str(os.getenv("SHORT1_HAZARD_OUTCOME_SELFTEST", "0")).lower() in ("1", "true", "yes", "on"):
        _selftest()
    else:
        asyncio.run(OutcomeWS().run())
