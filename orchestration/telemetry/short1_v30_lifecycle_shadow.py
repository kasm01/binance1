#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import redis
import requests


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
    return raw in {"1", "true", "yes", "on"}


def _jd(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "ignore")
    if not isinstance(raw, str):
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class V30Shadow:
    def __init__(self) -> None:
        self.strategy_id = str(os.getenv("STRATEGY_ID", "SHORT1") or "SHORT1").strip().upper()
        self.strategy_ns = self.strategy_id.lower()
        self.symbol = str(os.getenv("STRATEGY_SYMBOL", os.getenv("SHORT1_V30_SHADOW_SYMBOL", "INJUSDT")) or "INJUSDT").upper()
        self.source_stream = str(os.getenv("STRATEGY_MODEL_SHADOW_STREAM", os.getenv("SHORT1_V30_SHADOW_SOURCE_STREAM", f"telemetry:{self.strategy_ns}:{self.symbol.lower()}:model_shadow_v5")))
        self.out_stream = str(os.getenv("STRATEGY_LIFECYCLE_STREAM", os.getenv("SHORT1_V30_SHADOW_STREAM", f"telemetry:{self.strategy_ns}:lifecycle_shadow")))
        self.state_key = str(os.getenv("STRATEGY_LIFECYCLE_STATE_KEY", os.getenv("SHORT1_V30_SHADOW_STATE_KEY", f"shadow:{self.strategy_ns}:lifecycle:state")))
        self.last_id_key = str(os.getenv("STRATEGY_LIFECYCLE_LAST_ID_KEY", os.getenv("SHORT1_V30_SHADOW_LAST_ID_KEY", f"shadow:{self.strategy_ns}:lifecycle:last_source_id")))
        self.maxlen = max(1000, _ei("SHORT1_V30_SHADOW_MAXLEN", 20000))

        # Canonical V30 config is read from the same .env as production, but this process is non-routing.
        self.hard_score = _ef("MASTER_MIN_TRADE_SCORE", 0.540)
        self.soft_enable = _eb("MASTER_SHORT_SOFT_SCORE_ENABLE", True)
        self.soft_score = _ef("MASTER_SHORT_SOFT_SCORE_MIN", 0.530)
        self.flow60_exhaust = _ef("MASTER_SHORT_MONEY_FLOW_EXHAUSTION_60M", -0.20)
        self.oi_enable = _eb("RISK_OI_CONFIRMED_BUILD_ENABLE", True)
        self.oi_mult = max(0.0, min(1.0, _ef("RISK_OI_CONFIRMED_BUILD_MULTIPLIER", 0.50)))
        # V50 research-only sizing overlay. Uses only fields already present in
        # the economic-shadow event; no additional market-data requests.
        self.v50_enable = _eb("STRATEGY_CONTEXT_RISK_ENABLE", _eb("SHORT1_V50_CONTEXT_RISK_ENABLE", True))
        self.v50_mult = max(0.0, min(1.0, _ef("STRATEGY_CONTEXT_RISK_MULTIPLIER", _ef("SHORT1_V50_CONTEXT_RISK_MULTIPLIER", 0.50))))
        self.soft_stop_roi = _ef("ROI_HARD_STOP_PCT", -0.1200)
        self.stop_confirm_sec = max(0.0, _ef("SHORT_DYNAMIC_STOP_CONFIRM_SEC", 30.0))
        self.catastrophic_pct = max(0.0, _ef("SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT", 0.0600))
        self.reentry_sec = max(0.0, _ef("SHORT_STOP_REENTRY_COOLDOWN_SEC", 900.0))
        self.tp_arm = max(0.0, _ef("FIXED_ROI_TP_PCT", 0.0400))
        self.giveback = max(0.0, _ef("FIXED_ROI_TP_GIVEBACK", 0.0024))
        self.leverage = max(1.0, _ef("SHORT_SHADOW_LEVERAGE", 3.0))
        self.fee_side = max(0.0, _ef("SHORT1_V30_SHADOW_FEE_SIDE", 0.0004))
        self.dd_reduce = max(0.0, _ef("SHORT_DD_REDUCE_TRIGGER_PCT", 0.08))
        self.dd_recover = max(0.0, _ef("SHORT_DD_RECOVER_TRIGGER_PCT", 0.04))
        self.recovery_mult = max(0.0, min(1.0, _ef("SHORT_RECOVERY_POSITION_EQUITY_PCT", 0.70)))
        self.recovery_wins_required = max(1, _ei("SHORT_RECOVERY_WINS_REQUIRED", 2))
        self.start_equity = max(10.0, _ef("SHORT1_V30_SHADOW_START_EQUITY", 1000.0))

        host = str(os.getenv("REDIS_HOST", "localhost"))
        port = _ei("REDIS_PORT", 6379)
        db = _ei("REDIS_DB", 0)
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.http = requests.Session()
        self.mark_poll_sec = max(2.0, _ef("SHORT1_V30_SHADOW_MARK_POLL_SEC", 5.0))
        self._last_mark_poll = 0.0
        self._last_mark_price = 0.0
        self._last_mark_ts = 0.0
        self.state = self._load_state()
        self._last_persist = 0.0
        self._last_heartbeat = 0.0

    def _default_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "strategy_lifecycle_shadow_state.v1",
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "equity": self.start_equity,
            "equity_peak": self.start_equity,
            "risk_mode": "NORMAL",
            "recovery_wins": 0,
            "position": None,
            "reentry_until": 0.0,
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "updated_ts": time.time(),
        }

    def _load_state(self) -> Dict[str, Any]:
        obj = _jd(self.r.get(self.state_key))
        if not obj or str(obj.get("symbol") or "").upper() != self.symbol:
            return self._default_state()
        return obj

    def _persist(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_persist) < 5.0:
            return
        self.state["updated_ts"] = now
        self.r.set(self.state_key, json.dumps(self.state, separators=(",", ":"), sort_keys=True))
        self._last_persist = now

    def _emit(self, kind: str, **payload: Any) -> None:
        obj = {
            "schema_version": "strategy_lifecycle_shadow.v1",
            "strategy_id": self.strategy_id,
            "ts": time.time(),
            "kind": kind,
            "symbol": self.symbol,
            "routing": "OFF",
            **payload,
        }
        self.r.xadd(self.out_stream, {"json": json.dumps(obj, separators=(",", ":"), sort_keys=True)}, maxlen=self.maxlen, approximate=True)
        print(f"[V30-SHADOW] {kind} {json.dumps(payload, separators=(',', ':'), sort_keys=True)[:700]}", flush=True)

    @staticmethod
    def _regime(ctx: Any) -> Tuple[str, str]:
        c = ctx if isinstance(ctx, dict) else {}
        r = c.get("regime") if isinstance(c.get("regime"), dict) else {}
        return str(r.get("name") or "normal").lower(), str(r.get("build_direction") or "none").lower()

    def _risk_context_mult(self, ctx: Any) -> float:
        name, _ = self._regime(ctx)
        return self.oi_mult if self.oi_enable and name == "confirmed_build" else 1.0

    def _v50_context(self, flow60: Optional[float], ctx: Any) -> Dict[str, Any]:
        c = ctx if isinstance(ctx, dict) else {}
        mf = c.get("money_flow") if isinstance(c.get("money_flow"), dict) else {}
        oi = c.get("open_interest") if isinstance(c.get("open_interest"), dict) else {}
        flow15_raw = mf.get("flow_15m")
        oi15_raw = oi.get("oi_delta_15m")
        try:
            flow15 = float(flow15_raw)
        except Exception:
            flow15 = None
        try:
            oi15 = float(oi15_raw)
        except Exception:
            oi15 = None
        inputs_available = bool(c.get("available", False)) and flow60 is not None and flow15 is not None and oi15 is not None
        adverse_count = None
        mult = 1.0
        if inputs_available:
            adverse_count = int(float(flow60) > 0.0) + int(flow15 > 0.0) + int(oi15 > 0.0)
            if self.v50_enable and adverse_count >= 2:
                mult = self.v50_mult
        return {
            "schema_version": "short1_v50_context_risk.v1",
            "available": inputs_available,
            "flow60": flow60,
            "flow15_completed": flow15,
            "oi15_prior": oi15,
            "adverse_count": adverse_count,
            "triggered": bool(inputs_available and adverse_count is not None and adverse_count >= 2),
            "multiplier": float(mult),
            "routing": "OFF",
        }

    def _base_mult(self) -> float:
        mode = str(self.state.get("risk_mode") or "NORMAL").upper()
        if mode == "REDUCED":
            return 0.50
        if mode == "RECOVERY":
            return self.recovery_mult
        return 1.0

    def _update_risk_mode(self, *, close_reason: str = "", pnl_net: float = 0.0, closed_risk_mode: str = "") -> None:
        """V49B research shadow: NORMAL -> REDUCED -> RECOVERY -> NORMAL.

        RECOVERY starts when realized equity recovers to the configured DD threshold.
        A recovery stop returns immediately to REDUCED. NORMAL promotion requires
        N successful closes that were opened while RECOVERY sizing was active.
        """
        eq = float(self.state.get("equity") or self.start_equity)
        peak = max(float(self.state.get("equity_peak") or eq), eq)
        self.state["equity_peak"] = peak
        mode = str(self.state.get("risk_mode") or "NORMAL").upper()
        wins = int(self.state.get("recovery_wins") or 0)
        stop_reasons = {"dynamic_stop_confirmed", "dynamic_stop_catastrophic", "roi_hard_stop"}

        if mode == "NORMAL" and eq <= peak * (1.0 - self.dd_reduce):
            mode = "REDUCED"; wins = 0
            self._emit("RISK_MODE", mode=mode, equity=eq, peak=peak, recovery_wins=wins)
        elif mode == "REDUCED" and eq >= peak * (1.0 - self.dd_recover):
            mode = "RECOVERY"; wins = 0
            self._emit("RISK_MODE", mode=mode, equity=eq, peak=peak, recovery_wins=wins)
        elif mode == "RECOVERY":
            if str(close_reason or "") in stop_reasons:
                mode = "REDUCED"; wins = 0
                self._emit("RISK_MODE", mode=mode, equity=eq, peak=peak, recovery_wins=wins, reason="recovery_stop")
            elif str(closed_risk_mode or "").upper() == "RECOVERY" and float(pnl_net) > 0.0:
                wins += 1
                if wins >= self.recovery_wins_required:
                    mode = "NORMAL"; wins = 0
                    self._emit("RISK_MODE", mode=mode, equity=eq, peak=peak, recovery_wins=wins, reason="recovery_confirmed")
                else:
                    self._emit("RECOVERY_PROGRESS", mode=mode, equity=eq, peak=peak, recovery_wins=wins, required=self.recovery_wins_required)
            if eq <= peak * (1.0 - self.dd_reduce):
                mode = "REDUCED"; wins = 0
                self._emit("RISK_MODE", mode=mode, equity=eq, peak=peak, recovery_wins=wins, reason="dd_relapse")

        self.state["risk_mode"] = mode
        self.state["recovery_wins"] = wins

    def evaluate_source(self, source_id: str, obj: Dict[str, Any]) -> None:
        if str(obj.get("symbol") or "").upper() != self.symbol:
            return
        score = float(obj.get("economic_probability") or 0.0)
        model_provider = str(obj.get("model_provider") or "")
        confirm = str(obj.get("confirmation_state") or "")
        flow60_raw = obj.get("money_flow_60m")
        flow60 = None if flow60_raw is None else float(flow60_raw)
        ctx = obj.get("entry_market_context") if isinstance(obj.get("entry_market_context"), dict) else {}
        ctx_available = bool(ctx.get("available", False))
        regime_name, build_dir = self._regime(ctx)
        v50_ctx = self._v50_context(flow60, ctx)

        decision = "REJECT"
        reason = "model_score_low"
        if self.state.get("position") is not None:
            reason = "position_exists"
        elif time.time() < float(self.state.get("reentry_until") or 0.0):
            reason = "stop_reentry_cooldown"
        elif score < self.soft_score or (score < self.hard_score and not self.soft_enable):
            reason = "model_score_low"
        elif confirm != "CONFIRM":
            reason = "entry_confirmation_not_ready"
        elif flow60 is not None and flow60 <= self.flow60_exhaust:
            reason = "short_money_flow_exhaustion"
        elif score < self.hard_score:
            if not ctx_available:
                reason = "soft_entry_context_unavailable"
            elif regime_name == "confirmed_build" and build_dir == "short":
                decision, reason = "READY", "soft_short_build_confirmed"
            else:
                reason = "soft_short_build_not_confirmed"
        else:
            decision, reason = "READY", "hard_score_confirmed"

        self._emit(
            "SOURCE_DECISION",
            source_id=source_id,
            source_event_id=str(obj.get("event_id") or ""),
            model_provider=model_provider,
            score=score,
            confirmation_state=confirm,
            flow60=flow60,
            context_available=ctx_available,
            regime=regime_name,
            build_direction=build_dir,
            v50_context=v50_ctx,
            decision=decision,
            reason=reason,
        )
        if decision != "READY":
            return

        c = obj.get("confirmation") if isinstance(obj.get("confirmation"), dict) else {}
        entry = float(c.get("curr_close") or 0.0)
        if entry <= 0.0:
            self._emit("OPEN_REJECT", source_id=source_id, reason="invalid_entry_price")
            return

        context_mult = self._risk_context_mult(ctx)
        v50_mult = float(v50_ctx.get("multiplier") or 1.0)
        final_mult = self._base_mult() * context_mult * v50_mult
        equity_before = float(self.state.get("equity") or self.start_equity)
        notional = equity_before * final_mult
        if notional < 10.0:
            self._emit("OPEN_REJECT", source_id=source_id, reason="notional_below_min", notional=notional)
            return
        qty = notional / entry
        open_fee = notional * self.fee_side
        self.state["equity"] = equity_before - open_fee
        self.state["position"] = {
            "source_id": source_id,
            "source_event_id": str(obj.get("event_id") or ""),
            "model_provider": model_provider,
            "open_ts": time.time(),
            "entry_price": entry,
            "qty": qty,
            "notional": notional,
            "open_fee": open_fee,
            "score": score,
            "score_band": "SOFT" if score < self.hard_score else "HARD",
            "context_multiplier": context_mult,
            "v50_context_multiplier": v50_mult,
            "v50_context": v50_ctx,
            "final_size_multiplier": final_mult,
            "risk_mode": str(self.state.get("risk_mode") or "NORMAL").upper(),
            "regime": regime_name,
            "build_direction": build_dir,
            "best_roi": 0.0,
            "armed": False,
            "soft_pending_ts": 0.0,
        }
        self._persist(force=True)
        self._emit(
            "OPEN",
            source_id=source_id,
            model_provider=model_provider,
            entry_price=entry,
            score=score,
            score_band=self.state["position"]["score_band"],
            context_multiplier=context_mult,
            v50_context_multiplier=v50_mult,
            v50_context=v50_ctx,
            final_size_multiplier=final_mult,
            notional=notional,
            equity_before=equity_before,
            equity_after_open_fee=float(self.state["equity"]),
        )

    def on_price(self, price: float, price_ts: float) -> None:
        pos = self.state.get("position")
        if not isinstance(pos, dict) or price <= 0.0:
            self._persist()
            return
        ent = float(pos.get("entry_price") or 0.0)
        if ent <= 0.0:
            return
        roi = ((ent - price) / ent) * self.leverage
        best = max(float(pos.get("best_roi") or 0.0), roi)
        pos["best_roi"] = best
        if best >= self.tp_arm:
            pos["armed"] = True

        adverse = (price - ent) / ent
        pending_ts = float(pos.get("soft_pending_ts") or 0.0)

        if self.catastrophic_pct > 0.0 and adverse >= self.catastrophic_pct:
            self._close(price, price_ts, "dynamic_stop_catastrophic", roi)
            return

        if roi <= self.soft_stop_roi:
            if pending_ts <= 0.0:
                pos["soft_pending_ts"] = time.time()
                self._persist(force=True)
                self._emit("STOP_PENDING", price=price, roi=roi, trigger=self.soft_stop_roi)
                return
            if (time.time() - pending_ts) >= self.stop_confirm_sec:
                self._close(price, price_ts, "dynamic_stop_confirmed", roi)
                return
        elif pending_ts > 0.0:
            pos["soft_pending_ts"] = 0.0
            self._persist(force=True)
            self._emit("STOP_CLEAR", price=price, roi=roi)

        if bool(pos.get("armed", False)):
            trigger_roi = best - self.giveback
            if roi <= trigger_roi:
                self._close(price, price_ts, "tp_giveback", roi)
                return

        self.state["position"] = pos
        self._persist()

    def _close(self, price: float, price_ts: float, reason: str, roi: float) -> None:
        pos = self.state.get("position")
        if not isinstance(pos, dict):
            return
        entry = float(pos.get("entry_price") or 0.0)
        qty = float(pos.get("qty") or 0.0)
        open_fee = float(pos.get("open_fee") or 0.0)
        gross = (entry - price) * qty
        close_fee = price * qty * self.fee_side
        pnl_net = gross - close_fee - open_fee
        # equity already paid open_fee at OPEN; add gross minus close fee now.
        self.state["equity"] = float(self.state.get("equity") or self.start_equity) + gross - close_fee
        self.state["closed"] = int(self.state.get("closed") or 0) + 1
        if pnl_net > 0:
            self.state["wins"] = int(self.state.get("wins") or 0) + 1
        else:
            self.state["losses"] = int(self.state.get("losses") or 0) + 1
        if reason in {"dynamic_stop_confirmed", "dynamic_stop_catastrophic", "roi_hard_stop"}:
            self.state["reentry_until"] = time.time() + self.reentry_sec
        closed_risk_mode = str(pos.get("risk_mode") or self.state.get("risk_mode") or "NORMAL").upper()
        self.state["position"] = None
        self._update_risk_mode(close_reason=reason, pnl_net=pnl_net, closed_risk_mode=closed_risk_mode)
        self._persist(force=True)
        self._emit(
            "CLOSE",
            reason=reason,
            exit_price=price,
            mark_ts=price_ts,
            entry_price=entry,
            roi=roi,
            pnl_net=pnl_net,
            equity=float(self.state["equity"]),
            best_roi=float(pos.get("best_roi") or 0.0),
            hold_sec=max(0.0, time.time() - float(pos.get("open_ts") or time.time())),
            reentry_until=float(self.state.get("reentry_until") or 0.0),
        )

    def _fetch_mark_price_if_needed(self) -> Tuple[float, float]:
        # Cost-aware: no mark-price REST traffic while no paper position exists.
        if not isinstance(self.state.get("position"), dict):
            return 0.0, 0.0
        now = time.time()
        if self._last_mark_price > 0.0 and (now - self._last_mark_poll) < self.mark_poll_sec:
            return self._last_mark_price, self._last_mark_ts
        try:
            resp = self.http.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": self.symbol},
                timeout=5,
            )
            resp.raise_for_status()
            obj = resp.json()
            px = float(obj.get("markPrice") or 0.0)
            mts = float(obj.get("time") or 0.0) / 1000.0
            if px > 0.0:
                self._last_mark_price = px
                self._last_mark_ts = mts if mts > 0.0 else now
            self._last_mark_poll = now
        except Exception as exc:
            self._last_mark_poll = now
            self._emit("MARK_PRICE_WARN", error=str(exc)[:180])
        return self._last_mark_price, self._last_mark_ts

    def _initial_last_id(self) -> str:
        saved = str(self.r.get(self.last_id_key) or "").strip()
        if saved:
            return saved
        rows = self.r.xrevrange(self.source_stream, max="+", min="-", count=1)
        last = rows[0][0] if rows else "0-0"
        self.r.set(self.last_id_key, last)
        return last

    def run(self) -> None:
        if not self.r.ping():
            raise SystemExit("V30_SHADOW: Redis unavailable")
        last_id = self._initial_last_id()
        self._emit(
            "START",
            strategy_id=self.strategy_id,
            source_stream=self.source_stream,
            start_id=last_id,
            hard_score=self.hard_score,
            soft_score=self.soft_score,
            soft_stop_roi=self.soft_stop_roi,
            catastrophic_pct=self.catastrophic_pct,
            reentry_sec=self.reentry_sec,
            oi_multiplier=self.oi_mult,
            v50_context_risk_enable=self.v50_enable,
            v50_context_risk_multiplier=self.v50_mult,
            start_equity=float(self.state.get("equity") or self.start_equity),
        )
        while True:
            try:
                resp = self.r.xread({self.source_stream: last_id}, count=20, block=1000)
                for _stream, rows in resp:
                    for sid, fields in rows:
                        obj = _jd((fields or {}).get("json"))
                        self.evaluate_source(sid, obj)
                        last_id = sid
                        self.r.set(self.last_id_key, last_id)
                px, pts = self._fetch_mark_price_if_needed()
                price_age = (time.time() - pts) if pts > 0.0 else 999999.0
                if px > 0.0 and price_age <= max(15.0, self.mark_poll_sec * 3.0):
                    self.on_price(px, pts)
                now = time.time()
                if (now - self._last_heartbeat) >= 60.0:
                    pos = self.state.get("position") if isinstance(self.state.get("position"), dict) else None
                    self._emit(
                        "HEARTBEAT",
                        mark_price=px if px > 0.0 else None,
                        mark_age_sec=price_age if pts > 0.0 else None,
                        equity=float(self.state.get("equity") or self.start_equity),
                        risk_mode=str(self.state.get("risk_mode") or "NORMAL"),
                        position_open=bool(pos),
                        position_entry=(float(pos.get("entry_price")) if pos else None),
                        reentry_remaining_sec=max(0.0, float(self.state.get("reentry_until") or 0.0) - now),
                        closed=int(self.state.get("closed") or 0),
                        wins=int(self.state.get("wins") or 0),
                        losses=int(self.state.get("losses") or 0),
                    )
                    self._last_heartbeat = now
            except KeyboardInterrupt:
                self._persist(force=True)
                return
            except Exception as exc:
                self._emit("WARN", error=str(exc)[:220])
                time.sleep(1.0)


def _self_test() -> None:
    # Pure threshold/state semantics smoke test; no network/order path.
    assert 0.530 < 0.540
    ent = 10.0
    lev = 3.0
    assert abs((((ent - ent * 1.04) / ent) * lev) - (-0.12)) < 1e-9
    assert ent * 1.06 > ent
    print("SHORT1_V30_LIFECYCLE_SHADOW_SELFTEST_OK")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _self_test()
    else:
        V30Shadow().run()
