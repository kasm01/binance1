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


def _env_bool(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


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

    DRY_RUN behavior is now configurable:
      - BRIDGE_DRYRUN_BYPASS_GATE=1  -> bypass all open-position gates (dedup/max_open)
      - BRIDGE_DRYRUN_BYPASS_GATE=0  -> apply gates in dry-run (recommended for regression tests)
      - BRIDGE_DRYRUN_WRITE_STATE=1  -> keep open_positions_state in dry-run to test gates
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
        self.drain_pending = _env_bool("BRIDGE_DRAIN_PENDING", False)

        # Read tuning
        self.read_block_ms = _env_int("BRIDGE_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("BRIDGE_BATCH_COUNT", 20)

        # runtime
        self.dry_run = _env_bool("DRY_RUN", True)

        # gating + state
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")
        self.dedup_symbol = _env_bool("DEDUP_SYMBOL_OPEN", True)
        self.max_open = _env_int("MAX_OPEN_POSITIONS", 3)

        # TTL-based cleanup
        self.open_ttl_sec = _env_int("BRIDGE_OPEN_TTL_SEC", 0)  # 0 disabled
        self.cleanup_every_sec = _env_int("BRIDGE_CLEANUP_EVERY_SEC", 5)  # 0 disables periodic cleanup loop

        # DRY_RUN gating knobs (for regression tests)
        self.dryrun_bypass_gate = _env_bool("BRIDGE_DRYRUN_BYPASS_GATE", True)
        self.dryrun_write_state = _env_bool("BRIDGE_DRYRUN_WRITE_STATE", True)

        # selftest
        self.selftest = _env_bool("BRIDGE_SELFTEST", False)
        self.selftest_reset_state = _env_bool("BRIDGE_SELFTEST_RESET_STATE", True)
        self.selftest_symbols = [
            s.strip().upper()
            for s in os.getenv("BRIDGE_SELFTEST_SYMBOLS", "BTCUSDT,ETHUSDT,XRPUSDT,BNBUSDT").split(",")
            if s.strip()
        ]
        self.selftest_interval = os.getenv("BRIDGE_SELFTEST_INTERVAL", "5m")
        self.selftest_side = os.getenv("BRIDGE_SELFTEST_SIDE", "long")
        self.selftest_wait_sec = float(_safe_float(os.getenv("BRIDGE_SELFTEST_WAIT_SEC", "2.0"), 2.0))

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
            f"dry_run={self.dry_run} state_key={self.state_key} "
            f"dryrun_bypass_gate={self.dryrun_bypass_gate} dryrun_write_state={self.dryrun_write_state} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    # -----------------------
    # Redis helpers
    # -----------------------
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
            self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=5000,
                approximate=True,
            )
        except Exception:
            pass

    def _normalize_side(self, side: str) -> str:
        s = side.strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    # -----------------------
    # State helpers
    # -----------------------
    def _load_state(self) -> Dict[str, Any]:
        try:
            raw_state = self.r.get(self.state_key)
            st = json.loads(raw_state) if raw_state else {}
            return st if isinstance(st, dict) else {}
        except Exception:
            return {}

    def _save_state(self, open_map: Dict[str, Any]) -> None:
        payload = json.dumps({"open": open_map}, ensure_ascii=False)
        self.r.set(self.state_key, payload)

    def _cleanup_ttl_inplace(self, open_map: Dict[str, Any]) -> bool:
        """Returns True if modified."""
        if not self.open_ttl_sec or self.open_ttl_sec <= 0:
            return False
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
            return dirty
        except Exception:
            return False

    def _maybe_periodic_cleanup(self) -> None:
        """Periodic cleanup loop trigger: safe no-op if disabled."""
        if self.cleanup_every_sec <= 0:
            return
        try:
            state = self._load_state()
            open_map = state.get("open", {}) if isinstance(state, dict) else {}
            if not isinstance(open_map, dict):
                return
            dirty = self._cleanup_ttl_inplace(open_map)
            if dirty:
                self._save_state(open_map)
                self._publish_event("state_cleanup_ttl", {"state_key": self.state_key, "len_open": len(open_map)})
        except Exception:
            pass

    # -----------------------
    # Gate logic
    # -----------------------
    def _gate_allow_open(self, symbol: str) -> Tuple[bool, str]:
        # DRY-RUN MODE: optionally bypass gates (legacy behavior)
        if self.dry_run and self.dryrun_bypass_gate:
            return True, "dry_run_bypass"

        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}

        # TTL cleanup
        dirty = self._cleanup_ttl_inplace(open_map)
        if dirty:
            # allow cleanup writes even in dry-run if dryrun_write_state enabled
            if (not self.dry_run) or self.dryrun_write_state:
                try:
                    self._save_state(open_map)
                    self._publish_event("state_cleanup_ttl", {"state_key": self.state_key, "len_open": len(open_map)})
                except Exception:
                    pass

        if self.dedup_symbol and symbol in open_map:
            return False, f"dedup_symbol: {symbol} already open"

        if len(open_map) >= int(self.max_open):
            return False, f"max_open: {len(open_map)} >= {self.max_open}"

        return True, "ok"

    def _gate_mark_open(self, symbol: str, side: str, interval: str, intent_id: str) -> None:
        # In DRY_RUN, write state only if enabled (needed for regression tests)
        if self.dry_run and (not self.dryrun_write_state):
            return

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
        self._publish_event("state_written", {"state_key": self.state_key, "len_open": len(open_map)})

    def _gate_mark_close(self, symbol: str, reason: str, intent_id: str) -> None:
        # In DRY_RUN, write state only if enabled
        if self.dry_run and (not self.dryrun_write_state):
            return

        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}

        existed = symbol in open_map
        open_map.pop(symbol, None)

        self._save_state(open_map)
        self._publish_event(
            "state_closed",
            {"state_key": self.state_key, "symbol": symbol, "existed": existed, "reason": reason, "len_open": len(open_map), "intent_id": intent_id},
        )

    # -----------------------
    # Parsing / consuming
    # -----------------------
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

    # -----------------------
    # Intent semantics: open vs close
    # -----------------------
    def _is_close_intent(self, it: Dict[str, Any]) -> bool:
        # Flexible close detection
        action = _safe_str(it.get("action", "")).strip().lower()
        intent_type = _safe_str(it.get("intent_type", "")).strip().lower()
        close_flag = it.get("close", None)

        if action in ("close", "close_position", "exit"):
            return True
        if intent_type in ("close", "close_position", "exit"):
            return True
        if isinstance(close_flag, bool) and close_flag is True:
            return True

        # side could be "close"
        side = _safe_str(it.get("side", "")).strip().lower()
        if side in ("close", "exit"):
            return True

        return False

    # -----------------------
    # Forwarding logic
    # -----------------------
    def _forward_close(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))
        intent_id = _safe_str(it.get("intent_id", ""))

        # state update (even in DRY_RUN if enabled)
        self._gate_mark_close(symbol, reason="intent_close", intent_id=intent_id)

        if self.dry_run:
            self._publish_event(
                "close_dry",
                {"intent_id": intent_id, "symbol": symbol, "interval": interval, "why": "dry_run"},
            )
            return

        meta = {
            "reason": "ORCH_INTENT_CLOSE",
            "intent_id": intent_id,
            "source_pkg_id": source_pkg_id,
        }

        err = None

        # try executor close methods
        for name in ("close_position_from_signal", "close_position", "close_trade", "close"):
            fn = getattr(self.executor, name, None)
            if not callable(fn):
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    asyncio.run(fn(symbol=symbol, interval=interval, meta=meta))
                    self._publish_event(
                        "close_ok",
                        {"intent_id": intent_id, "method": f"{name}(asyncio.run)", "symbol": symbol, "interval": interval},
                    )
                    return
                else:
                    # best-effort signature
                    try:
                        fn(symbol=symbol, interval=interval, meta=meta)
                    except TypeError:
                        fn(symbol=symbol)
                    self._publish_event(
                        "close_ok",
                        {"intent_id": intent_id, "method": f"{name}(sync)", "symbol": symbol, "interval": interval},
                    )
                    return
            except Exception as e:
                err = err or f"{name} failed: {repr(e)}"

        self._publish_event(
            "close_skip",
            {"intent_id": intent_id, "symbol": symbol, "interval": interval, "why": err or "no close entrypoint on executor"},
        )

    def _forward_open(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))
        side = self._normalize_side(_safe_str(it.get("side", "long")))
        intent_id = _safe_str(it.get("intent_id", ""))

        allow, why = self._gate_allow_open(symbol)
        if not allow:
            self._publish_event(
                "forward_skip",
                {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": why},
            )
            return

        # DRY RUN: do not call executor, but write state if enabled (for regression tests)
        if self.dry_run:
            self._gate_mark_open(symbol, side, interval, intent_id)
            self._publish_event(
                "forward_dry",
                {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": "dry_run"},
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
                self._publish_event(
                    "forward_ok",
                    {"intent_id": intent_id, "method": "open_position_from_signal", "symbol": symbol, "side": side},
                )
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
                    self._publish_event(
                        "forward_ok",
                        {"intent_id": intent_id, "method": f"{name}(asyncio.run)", "symbol": symbol, "side": side},
                    )
                    return
                else:
                    fn2(symbol=symbol, side=side, interval=interval, meta=meta)
                    self._gate_mark_open(symbol, side, interval, intent_id)
                    self._publish_event(
                        "forward_ok",
                        {"intent_id": intent_id, "method": f"{name}(sync)", "symbol": symbol, "side": side},
                    )
                    return
            except TypeError as e:
                err = err or f"{name} signature mismatch: {e}"
            except Exception as e:
                err = err or f"{name} failed: {repr(e)}"

        self._publish_event(
            "forward_skip",
            {
                "intent_id": intent_id,
                "symbol": symbol,
                "side": side,
                "interval": interval,
                "why": err or "no callable entrypoint on executor",
            },
        )

    def _forward_one(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        if self._is_close_intent(it):
            self._forward_close(it, source_pkg_id=source_pkg_id)
        else:
            self._forward_open(it, source_pkg_id=source_pkg_id)

    def _process_pkg(self, sid: str, pkg: Dict[str, Any]) -> None:
        items = pkg.get("items") or []
        if not isinstance(items, list) or not items:
            return
        for it in items:
            if isinstance(it, dict):
                self._forward_one(it, source_pkg_id=sid)

    # -----------------------
    # Selftest
    # -----------------------
    def _selftest_run(self) -> None:
        """
        Pushes 4 different symbols as trade intents and reports summary into exec_events_stream.
        Designed as a regression smoke-test for max_open/dedup gating.
        """
        try:
            if self.selftest_reset_state:
                try:
                    self.r.delete(self.state_key)
                    self._publish_event("selftest_reset_state", {"state_key": self.state_key})
                except Exception:
                    pass

            # Compose 4 intents in 4 separate packages so events are clearer
            stamp = str(int(time.time()))
            for i, sym in enumerate(self.selftest_symbols[:4], start=1):
                pkg = {
                    "items": [
                        {
                            "symbol": sym,
                            "side": self.selftest_side,
                            "interval": self.selftest_interval,
                            "intent_id": f"selftest-{stamp}-{i}-{sym}",
                        }
                    ]
                }
                self.r.xadd(self.in_stream, {"json": json.dumps(pkg, ensure_ascii=False)})

            # Give bridge loop a moment to consume (we are in same process, but still async via stream + group)
            time.sleep(self.selftest_wait_sec)

            # Summarize: read last N exec events and count selftest results
            rows = self.r.xrevrange(self.out_stream, max="+", min="-", count=200)
            ok = 0
            dry = 0
            skip = 0
            close_ok = 0
            close_dry = 0
            close_skip = 0
            reasons: Dict[str, int] = {}

            for _id, fields in rows:
                s = fields.get("json")
                if not s:
                    continue
                try:
                    ev = json.loads(s)
                except Exception:
                    continue
                iid = _safe_str(ev.get("intent_id", ""))
                if not iid.startswith(f"selftest-{stamp}-"):
                    continue

                kind = _safe_str(ev.get("kind", ""))
                if kind == "forward_ok":
                    ok += 1
                elif kind == "forward_dry":
                    dry += 1
                elif kind == "forward_skip":
                    skip += 1
                    why = _safe_str(ev.get("why", ""))
                    reasons[why] = reasons.get(why, 0) + 1
                elif kind == "close_ok":
                    close_ok += 1
                elif kind == "close_dry":
                    close_dry += 1
                elif kind == "close_skip":
                    close_skip += 1

            self._publish_event(
                "selftest_summary",
                {
                    "stamp": stamp,
                    "symbols": self.selftest_symbols[:4],
                    "dry_run": self.dry_run,
                    "dryrun_bypass_gate": self.dryrun_bypass_gate,
                    "max_open": self.max_open,
                    "dedup_symbol": self.dedup_symbol,
                    "counts": {
                        "forward_ok": ok,
                        "forward_dry": dry,
                        "forward_skip": skip,
                        "close_ok": close_ok,
                        "close_dry": close_dry,
                        "close_skip": close_skip,
                    },
                    "skip_reasons": reasons,
                    "note": "Expected (when BRIDGE_DRYRUN_BYPASS_GATE=0 and DRY_RUN=1 and BRIDGE_DRYRUN_WRITE_STATE=1): 3 forward_dry + 1 forward_skip(max_open).",
                },
            )
        except Exception as e:
            self._publish_event("selftest_error", {"err": repr(e)})

    # -----------------------
    # Main loop
    # -----------------------
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

        # selftest (once per process)
        if self.selftest:
            self._publish_event("selftest_start", {"enabled": True})
            self._selftest_run()

        idle = 0
        last_cleanup = time.time()

        while True:
            rows = self._xreadgroup(">")
            if not rows:
                idle += 1
                # periodic TTL cleanup even if stream idle
                if self.cleanup_every_sec > 0 and (time.time() - last_cleanup) >= self.cleanup_every_sec:
                    self._maybe_periodic_cleanup()
                    last_cleanup = time.time()

                if idle % 30 == 0:
                    print("[IntentBridge] idle...")
                continue

            idle = 0
            mids = [sid for sid, _ in rows]

            for sid, pkg in rows:
                if pkg:
                    self._process_pkg(sid, pkg)

            self._ack(mids)

            # periodic cleanup after batch too
            if self.cleanup_every_sec > 0 and (time.time() - last_cleanup) >= self.cleanup_every_sec:
                self._maybe_periodic_cleanup()
                last_cleanup = time.time()

            time.sleep(0.05)


if __name__ == "__main__":
    IntentBridge().run_forever()
