# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import asyncio
import inspect
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(k: str, default: int) -> int:
    try:
        return int(str(os.getenv(k, str(default))).strip())
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


class IntentBridge:
    """
    Consumes trade_intents_stream (group-based), forwards intents to TradeExecutor.

    IN:  trade_intents_stream
    OUT: exec_events_stream (log/trace)

    Restart-safe: XREADGROUP + XACK
    """

    def __init__(self) -> None:
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        self.in_stream = os.getenv("BRIDGE_IN_STREAM", os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"))
        self.out_stream = os.getenv("BRIDGE_OUT_STREAM", "exec_events_stream")

        # Consumer group
        self.group = os.getenv("BRIDGE_GROUP", "bridge_g")
        self.consumer = os.getenv("BRIDGE_CONSUMER", "bridge_1")
        self.group_start_id = os.getenv("BRIDGE_GROUP_START_ID", "$")  # "$" new only, or "0-0" from beginning
        self.drain_pending = os.getenv("BRIDGE_DRAIN_PENDING", "0").strip().lower() in ("1", "true", "yes", "on")

        # Read tuning
        self.read_block_ms = _env_int("BRIDGE_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("BRIDGE_BATCH_COUNT", 20)

        # runtime
        self.dry_run = os.getenv("DRY_RUN", "1").lower() in ("1", "true", "yes", "on")

        # gating
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")
        self.dedup_symbol = os.getenv("DEDUP_SYMBOL_OPEN", "1").strip().lower() not in ("0", "false", "no", "off")
        self.max_open = _env_int("MAX_OPEN_POSITIONS", 3)
        self.open_ttl_sec = _env_int("BRIDGE_OPEN_TTL_SEC", 0)  # 0 disabled

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        # TradeExecutor
        from core.trade_executor import TradeExecutor  # noqa
        from core.risk_manager import RiskManager  # noqa

        self.risk_manager = RiskManager()
        self.executor = TradeExecutor(
            client=None,
            risk_manager=self.risk_manager,
            position_manager=None,
            dry_run=self.dry_run,
        )

        print(
            f"[IntentBridge] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"dry_run={self.dry_run} redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[IntentBridge] XGROUP created: stream={stream} group={group} start_id={start_id}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise

    def _ack(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self.r.xack(self.in_stream, self.group, *ids)
        except Exception:
            pass

    def _publish_event(self, kind: str, data: Dict[str, Any]) -> None:
        payload = {"ts_utc": _now_utc_iso(), "kind": kind, **data}
        try:
            self.r.xadd(self.out_stream, {"json": json.dumps(payload, ensure_ascii=False)}, maxlen=5000, approximate=True)
        except Exception:
            pass

    def _normalize_side(self, side: str) -> str:
        s = side.strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    def _load_state(self) -> Dict[str, Any]:
        try:
            raw_state = self.r.get(self.state_key)
            st = json.loads(raw_state) if raw_state else {}
            return st if isinstance(st, dict) else {}
        except Exception:
            return {}

    def _save_state(self, open_map: Dict[str, Any]) -> None:
        try:
            self.r.set(self.state_key, json.dumps({"open": open_map}, ensure_ascii=False))
        except Exception:
            pass

    def _cleanup_ttl(self, open_map: Dict[str, Any]) -> Dict[str, Any]:
        if not self.open_ttl_sec or self.open_ttl_sec <= 0:
            return open_map
        try:
            now = datetime.now(timezone.utc)
            dirty = False
            for sym, info in list(open_map.items()):
                ts = ""
                if isinstance(info, dict):
                    ts = str(info.get("ts_utc") or "")
                try:
                    t0 = datetime.fromisoformat(ts) if ts else None
                except Exception:
                    t0 = None
                if (t0 is None) or ((now - t0).total_seconds() > self.open_ttl_sec):
                    open_map.pop(sym, None)
                    dirty = True
            if dirty:
                self._save_state(open_map)
        except Exception:
            pass
        return open_map

    def _gate_allow_open(self, symbol: str) -> Tuple[bool, str]:
        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}

        open_map = self._cleanup_ttl(open_map)

        if self.dedup_symbol and symbol in open_map:
            return False, f"dedup_symbol: {symbol} already open"

        if len(open_map) >= int(self.max_open):
            return False, f"max_open: {len(open_map)} >= {self.max_open}"

        return True, "ok"

    def _gate_mark_open(self, symbol: str, side: str, interval: str, intent_id: str) -> None:
        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}
        open_map[symbol] = {
            "side": side,
            "interval": interval,
            "ts_utc": _now_utc_iso(),
            "intent_id": intent_id,
        }
        self._save_state(open_map)

    def _parse_pkg(self, sid: str, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        s = fields.get("json")
        if not s:
            return None
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                d.setdefault("ts_utc", _now_utc_iso())
                d["_source_stream_id"] = sid
                return d
        except Exception:
            return None
        return None

    def _xreadgroup(self, start_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        start_id:
          ">" => new
          "0" => pending (PEL)
        """
        resp = self.r.xreadgroup(
            groupname=self.group,
            consumername=self.consumer,
            streams={self.in_stream: start_id},
            count=self.batch_count,
            block=self.read_block_ms,
        )
        out: List[Tuple[str, Dict[str, Any]]] = []
        if not resp:
            return out

        for _name, entries in resp:
            for sid, fields in entries:
                pkg = self._parse_pkg(sid, fields)
                out.append((sid, pkg or {}))
        return out

    def _forward_one(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))
        side = self._normalize_side(_safe_str(it.get("side", "long")))
        intent_id = _safe_str(it.get("intent_id", ""))

        # gate
        allow, why = self._gate_allow_open(symbol)
        if not allow:
            self._publish_event(
                "forward_skip",
                {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": why},
            )
            return

        lev = int(_safe_float(it.get("recommended_leverage", 5), 5))
        npct = float(_safe_float(it.get("recommended_notional_pct", 0.05), 0.05))
        score = float(_safe_float(it.get("score", 0.0), 0.0))

        meta = {
            "reason": "ORCH_INTENT",
            "intent_id": intent_id,
            "score": score,
            "recommended_leverage": lev,
            "recommended_notional_pct": npct,
            "source_pkg_id": source_pkg_id,
        }

        err = None

        # 1) open_position_from_signal (best)
        fn = getattr(self.executor, "open_position_from_signal", None)
        if callable(fn):
            try:
                fn(symbol=symbol, side=side, interval=interval, meta=meta)
                self._gate_mark_open(symbol, side, interval, intent_id)
                self._publish_event("forward_ok", {"intent_id": intent_id, "method": "open_position_from_signal", "symbol": symbol, "side": side})
                return
            except Exception as e:
                err = repr(e)

        # 2) open_position / execute_trade (async supported)
        for name in ("open_position", "execute_trade"):
            fn2 = getattr(self.executor, name, None)
            if not callable(fn2):
                continue
            try:
                if inspect.iscoroutinefunction(fn2):
                    asyncio.run(fn2(symbol=symbol, side=side, interval=interval, meta=meta))
                    self._gate_mark_open(symbol, side, interval, intent_id)
                    self._publish_event("forward_ok", {"intent_id": intent_id, "method": f"{name}(asyncio.run)", "symbol": symbol, "side": side})
                    return
                else:
                    fn2(symbol=symbol, side=side, interval=interval, meta=meta)
                    self._gate_mark_open(symbol, side, interval, intent_id)
                    self._publish_event("forward_ok", {"intent_id": intent_id, "method": f"{name}(sync)", "symbol": symbol, "side": side})
                    return
            except TypeError as e:
                err = err or f"{name} signature mismatch: {e}"
            except Exception as e:
                err = err or f"{name} failed: {repr(e)}"

        self._publish_event(
            "forward_skip",
            {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": err or "no callable entrypoint on executor"},
        )

    def _process_pkg(self, sid: str, pkg: Dict[str, Any]) -> None:
        items = pkg.get("items") or []
        if not isinstance(items, list) or not items:
            return
        for it in items:
            if isinstance(it, dict):
                self._forward_one(it, source_pkg_id=sid)

    def run_forever(self) -> None:
        # optional: drain pending first
        if self.drain_pending:
            print("[IntentBridge] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break
                mids = [sid for sid, _ in rows]
                for sid, pkg in rows:
                    if pkg:
                        self._process_pkg(sid, pkg)
                self._ack(mids)
                time.sleep(0.05)
            print("[IntentBridge] pending drained.")

        idle = 0
        while True:
            rows = self._xreadgroup(">")
            if not rows:
                idle += 1
                if idle % 30 == 0:
                    print("[IntentBridge] idle...")
                continue
            idle = 0

            mids = [sid for sid, _ in rows]
            for sid, pkg in rows:
                if pkg:
                    self._process_pkg(sid, pkg)

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    IntentBridge().run_forever()
