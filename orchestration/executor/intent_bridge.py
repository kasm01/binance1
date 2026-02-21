# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import inspect
import json
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
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off", ""):
        return False
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


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        if not ts:
            return None
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _run_coroutine_safely(coro):
    """
    asyncio.run() mevcut loop içinde patlayabilir.
    Bridge tipik olarak loop içinde çalışmaz ama güvenli olsun diye fallback ekliyoruz.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass
class IntentBridge:
    """
    Consumes trade_intents_stream (group-based), forwards intents to TradeExecutor.

    IN:  trade_intents_stream
    OUT: exec_events_stream (log/trace)

    Restart-safe: XREADGROUP + XACK

    DRY_RUN behavior is configurable:
      - BRIDGE_DRYRUN_BYPASS_GATE=1   -> bypass all open-position gates (dedup/max_open/cooldown)
      - BRIDGE_DRYRUN_WRITE_STATE=1   -> keep open_positions_state in dry-run to test gates
      - BRIDGE_DRYRUN_CALL_EXECUTOR=1 -> DRY_RUN olsa bile executor'u çağır (executor zaten dry_run=True ise emir atmaz)

    IMPORTANT:
      - Key-level TTL for state is controlled via BRIDGE_STATE_TTL_SEC (default 600).
        If BRIDGE_STATE_TTL_SEC <= 0, key TTL is disabled.
    """

    def __init__(self) -> None:
        # Redis connection
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        # Streams
        self.in_stream = os.getenv("BRIDGE_IN_STREAM", os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"))
        self.out_stream = os.getenv("BRIDGE_OUT_STREAM", "exec_events_stream")

        # Consumer group
        self.group = os.getenv("BRIDGE_GROUP", "bridge_g")
        self.consumer = os.getenv("BRIDGE_CONSUMER", "bridge_1")
        self.group_start_id = os.getenv("BRIDGE_GROUP_START_ID", "$")
        self.drain_pending = _env_bool("BRIDGE_DRAIN_PENDING", False)

        # Read tuning
        self.read_block_ms = _env_int("BRIDGE_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("BRIDGE_BATCH_COUNT", 20)

        # Runtime flags
        self.dry_run = _env_bool("DRY_RUN", True)

        # --- LIVE SAFETY POLICY ---
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        if (not self.armed) or self.kill_switch:
            self.dry_run = True

        # Gate + state
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")
        self.dedup_symbol = _env_bool("DEDUP_SYMBOL_OPEN", True)
        self.max_open = _env_int("MAX_OPEN_POSITIONS", 3)

        # Key-level TTL
        self.state_ttl_sec = _env_int("BRIDGE_STATE_TTL_SEC", 600)  # <=0 disables key TTL

        # TTL-based cleanup (maps inside JSON)
        self.open_ttl_sec = _env_int("BRIDGE_OPEN_TTL_SEC", 0)
        self.close_cooldown_sec = _env_int("BRIDGE_CLOSE_COOLDOWN_SEC", 30)
        self.closed_ttl_sec = _env_int(
            "BRIDGE_CLOSED_TTL_SEC",
            max(0, int(self.close_cooldown_sec) * 10) if self.close_cooldown_sec > 0 else 0,
        )

        # periodic cleanup loop
        self.cleanup_every_sec = _env_int("BRIDGE_CLEANUP_EVERY_SEC", 5)

        # DRY_RUN gating knobs
        self.dryrun_bypass_gate = _env_bool("BRIDGE_DRYRUN_BYPASS_GATE", False)
        self.dryrun_write_state = _env_bool("BRIDGE_DRYRUN_WRITE_STATE", True)

        # dry-run'da executor çağırma opsiyonu
        self.dryrun_call_executor = _env_bool("BRIDGE_DRYRUN_CALL_EXECUTOR", False)

        # Success assumptions
        self.open_assume_success_on_noexc = _env_bool("BRIDGE_OPEN_ASSUME_SUCCESS_ON_NOEXC", True)
        self.close_assume_success_on_noexc = _env_bool("BRIDGE_CLOSE_ASSUME_SUCCESS_ON_NOEXC", True)

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

        # Ensure stream+group exist
        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        # TradeExecutor + RiskManager + PositionManager
        from core.trade_executor import TradeExecutor  # noqa
        from core.risk_manager import RiskManager  # noqa
        from core.position_manager import PositionManager  # noqa

        self.risk_manager = RiskManager()

        redis_url = os.getenv("POSITION_REDIS_URL", f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}")
        pm_key_prefix = os.getenv("POSITION_KEY_PREFIX", "positions")
        pm_db = _env_int("POSITION_REDIS_DB", self.redis_db)

        self.position_manager = PositionManager(
            redis_url=redis_url,
            redis_db=pm_db,
            redis_key_prefix=pm_key_prefix,
            enable_pg=_env_bool("ENABLE_PG_POS_LOG", False),
            pg_dsn=os.getenv("PG_DSN") or None,
        )

        self.executor = TradeExecutor(
            client=None,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            dry_run=self.dry_run,
        )

        print(
            f"[IntentBridge] started. in={self.in_stream} out={self.out_stream} armed={getattr(self, 'armed', None)} kill_switch={getattr(self, 'kill_switch', None)} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"dry_run={self.dry_run} state_key={self.state_key} state_ttl_sec={self.state_ttl_sec} "
            f"dryrun_bypass_gate={self.dryrun_bypass_gate} dryrun_write_state={self.dryrun_write_state} dryrun_call_executor={self.dryrun_call_executor} "
            f"max_open={self.max_open} dedup_symbol={self.dedup_symbol} "
            f"close_cooldown_sec={self.close_cooldown_sec} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db} "
            f"pos_redis={redis_url} pos_prefix={pm_key_prefix}"
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
        s = (side or "").strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    def _normalize_direction(self, side_or_dir: str) -> str:
        s = (side_or_dir or "").strip().upper()
        if s in ("LONG", "BUY"):
            return "LONG"
        if s in ("SHORT", "SELL"):
            return "SHORT"
        sl = (side_or_dir or "").strip().lower()
        if sl == "long":
            return "LONG"
        if sl == "short":
            return "SHORT"
        return ""

    def _extract_open_price(self, it: Dict[str, Any]) -> Optional[float]:
        # OPEN intent accepted fields:
        for k in ("price", "mark_price", "last_price", "entry_price", "close_price"):
            v = it.get(k, None)
            if v is None:
                continue
            p = _safe_float(v, 0.0)
            if p > 0:
                return float(p)
        return None

    def _extract_close_price(self, it: Dict[str, Any]) -> Optional[float]:
        # CLOSE intent accepted fields:
        for k in ("price", "exit_price", "close_price", "fill_price"):
            v = it.get(k, None)
            if v is None:
                continue
            p = _safe_float(v, 0.0)
            if p > 0:
                return float(p)
        return None

    def _is_close_success(self, res: Any) -> bool:
        if res is None:
            return bool(self.close_assume_success_on_noexc)
        if isinstance(res, bool):
            return res is True
        if isinstance(res, dict):
            st = _safe_str(res.get("status", "")).strip().lower()
            if st in ("success", "ok", "closed", "dry_run"):
                return True
            if _safe_str(res.get("result", "")).strip().lower() in ("success", "ok"):
                return True
            return False
        return False

    def _is_open_success(self, res: Any) -> bool:
        if res is None:
            return bool(self.open_assume_success_on_noexc)
        if isinstance(res, bool):
            return res is True
        if isinstance(res, dict):
            st = _safe_str(res.get("status", "")).strip().lower()
            if st in ("success", "ok", "opened", "open", "dry_run"):
                return True
            if st in ("skip", "fail", "error"):
                return False
        return True
    # -----------------------
    # State helpers
    # -----------------------
    def _load_state(self) -> Dict[str, Any]:
        try:
            raw_state = self.r.get(self.state_key)
            st = json.loads(raw_state) if raw_state else {}
            if not isinstance(st, dict):
                return {"open": {}, "closed": {}}
            if "open" not in st or not isinstance(st.get("open"), dict):
                st["open"] = {}
            if "closed" not in st or not isinstance(st.get("closed"), dict):
                st["closed"] = {}
            return st
        except Exception:
            return {"open": {}, "closed": {}}

    def _save_state(self, open_map: Dict[str, Any], closed_map: Optional[Dict[str, Any]] = None) -> None:
        payload_obj: Dict[str, Any] = {"open": open_map}
        if closed_map is not None:
            payload_obj["closed"] = closed_map
        else:
            st = self._load_state()
            payload_obj["closed"] = st.get("closed", {}) if isinstance(st.get("closed"), dict) else {}

        payload = json.dumps(payload_obj, ensure_ascii=False)
        ttl = int(self.state_ttl_sec or 0)
        try:
            if ttl > 0:
                self.r.set(self.state_key, payload, ex=ttl)
            else:
                self.r.set(self.state_key, payload)
        except Exception:
            pass

    def _cleanup_map_ttl_inplace(self, m: Dict[str, Any], ttl_sec: int) -> bool:
        if not ttl_sec or ttl_sec <= 0:
            return False
        try:
            now = datetime.now(timezone.utc)
            dirty = False
            for sym, info in list(m.items()):
                ts = ""
                if isinstance(info, dict):
                    ts = str(info.get("ts_utc") or "")
                t0 = _parse_iso(ts) if ts else None
                if (t0 is None) or ((now - t0).total_seconds() > ttl_sec):
                    m.pop(sym, None)
                    dirty = True
            return dirty
        except Exception:
            return False

    def _maybe_periodic_cleanup(self) -> None:
        if self.cleanup_every_sec <= 0:
            return
        try:
            state = self._load_state()
            open_map = state.get("open", {})
            closed_map = state.get("closed", {})
            if not isinstance(open_map, dict) or not isinstance(closed_map, dict):
                return

            dirty_open = self._cleanup_map_ttl_inplace(open_map, int(self.open_ttl_sec))
            dirty_closed = self._cleanup_map_ttl_inplace(closed_map, int(self.closed_ttl_sec))

            if dirty_open or dirty_closed:
                self._save_state(open_map, closed_map)
                self._publish_event(
                    "state_cleanup_ttl",
                    {
                        "state_key": self.state_key,
                        "len_open": len(open_map),
                        "len_closed": len(closed_map),
                        "open_ttl_sec": int(self.open_ttl_sec),
                        "closed_ttl_sec": int(self.closed_ttl_sec),
                        "state_ttl_sec": int(self.state_ttl_sec or 0),
                    },
                )
        except Exception:
            pass
    # -----------------------
    # Gate logic
    # -----------------------
    def _cooldown_left_sec(self, closed_map: Dict[str, Any], symbol: str) -> int:
        if not self.close_cooldown_sec or self.close_cooldown_sec <= 0:
            return 0
        info = closed_map.get(symbol)
        if not isinstance(info, dict):
            return 0
        t0 = _parse_iso(_safe_str(info.get("ts_utc", "")))
        if t0 is None:
            return 0
        now = datetime.now(timezone.utc)
        elapsed = (now - t0).total_seconds()
        left = float(self.close_cooldown_sec) - elapsed
        return int(left) if left > 0 else 0

    def _gate_allow_open(self, symbol: str) -> Tuple[bool, str]:
        if self.dry_run and self.dryrun_bypass_gate:
            return True, "dry_run_bypass"

        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        closed_map = state.get("closed", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}
        if not isinstance(closed_map, dict):
            closed_map = {}

        dirty_open = self._cleanup_map_ttl_inplace(open_map, int(self.open_ttl_sec))
        dirty_closed = self._cleanup_map_ttl_inplace(closed_map, int(self.closed_ttl_sec))
        if dirty_open or dirty_closed:
            if (not self.dry_run) or self.dryrun_write_state:
                try:
                    self._save_state(open_map, closed_map)
                    self._publish_event(
                        "state_cleanup_ttl",
                        {"state_key": self.state_key, "len_open": len(open_map), "len_closed": len(closed_map)},
                    )
                except Exception:
                    pass

        left = self._cooldown_left_sec(closed_map, symbol)
        if left > 0:
            return False, f"cooldown: {left}s left"

        if self.dedup_symbol and symbol in open_map:
            return False, f"dedup_symbol: {symbol} already open"

        if len(open_map) >= int(self.max_open):
            return False, f"max_open: {len(open_map)} >= {self.max_open}"

        return True, "ok"

    def _gate_mark_open(self, symbol: str, side: str, interval: str, intent_id: str) -> None:
        if self.dry_run and (not self.dryrun_write_state):
            return

        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        closed_map = state.get("closed", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}
        if not isinstance(closed_map, dict):
            closed_map = {}

        if symbol in closed_map:
            closed_map.pop(symbol, None)
            self._publish_event("state_closed_cleared", {"state_key": self.state_key, "symbol": symbol})

        open_map[symbol] = {"side": side, "interval": interval, "ts_utc": _now_utc_iso(), "intent_id": intent_id}
        self._save_state(open_map, closed_map)

        self._publish_event(
            "state_written",
            {
                "state_key": self.state_key,
                "len_open": len(open_map),
                "len_closed": len(closed_map),
                "state_ttl_sec": int(self.state_ttl_sec or 0),
            },
        )

    def _gate_mark_close(self, symbol: str, reason: str, intent_id: str) -> None:
        if self.dry_run and (not self.dryrun_write_state):
            return

        state = self._load_state()
        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        closed_map = state.get("closed", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}
        if not isinstance(closed_map, dict):
            closed_map = {}

        existed = symbol in open_map
        open_map.pop(symbol, None)

        if self.close_cooldown_sec and self.close_cooldown_sec > 0:
            closed_map[symbol] = {"ts_utc": _now_utc_iso(), "reason": reason, "intent_id": intent_id}

        self._save_state(open_map, closed_map)
        self._publish_event(
            "state_closed",
            {
                "state_key": self.state_key,
                "symbol": symbol,
                "existed": existed,
                "reason": reason,
                "len_open": len(open_map),
                "len_closed": len(closed_map),
                "intent_id": intent_id,
                "state_ttl_sec": int(self.state_ttl_sec or 0),
            },
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
        try:
            resp = self.r.xreadgroup(
                groupname=self.group,
                consumername=self.consumer,
                streams={self.in_stream: start_id},
                count=self.batch_count,
                block=self.read_block_ms,
            )
        except redis.exceptions.ResponseError as e:
            msg = str(e)
            if "UNBLOCKED" in msg and "no longer exists" in msg:
                print(f"[IntentBridge] WARN: stream disappeared during blocking read; recreating group. err={msg}")
                try:
                    self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)
                except Exception:
                    pass
                return []
            raise
        except Exception:
            return []

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
        action = _safe_str(it.get("action", "")).strip().lower()
        intent_type = _safe_str(it.get("intent_type", "")).strip().lower()
        close_flag = it.get("close", None)

        if action in ("close", "close_position", "exit"):
            return True
        if intent_type in ("close", "close_position", "exit"):
            return True
        if isinstance(close_flag, bool) and close_flag is True:
            return True

        side = _safe_str(it.get("side", "")).strip().lower()
        if side in ("close", "exit"):
            return True

        return False

    # -----------------------
    # Executor call helpers (TradeExecutor uyumlu)
    # -----------------------
    def _call_open_executor(self, symbol: str, side: str, interval: str, meta: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """
        Öncelik: TradeExecutor.open_position_from_signal (sync)
        Fallback: execute_decision (async) (varsa)
        """
        fn = getattr(self.executor, "open_position_from_signal", None)
        if callable(fn):
            try:
                res = fn(symbol=symbol, side=side, interval=interval, meta=meta)
                return True, res, "open_position_from_signal(sync)"
            except Exception as e:
                return False, None, f"open_position_from_signal failed: {repr(e)}"

        fn2 = getattr(self.executor, "execute_decision", None)
        if callable(fn2):
            signal = "BUY" if side == "long" else "SELL"
            try:
                if inspect.iscoroutinefunction(fn2):
                    res = _run_coroutine_safely(
                        fn2(
                            signal=signal,
                            symbol=symbol,
                            price=float(meta.get("price") or 0.0),
                            size=None,
                            interval=interval,
                            training_mode=False,
                            hybrid_mode=False,
                            probs={},
                            extra=meta,
                        )
                    )
                    return True, res, "execute_decision(async)"
                else:
                    res = fn2(
                        signal=signal,
                        symbol=symbol,
                        price=float(meta.get("price") or 0.0),
                        size=None,
                        interval=interval,
                        training_mode=False,
                        hybrid_mode=False,
                        probs={},
                        extra=meta,
                    )
                    return True, res, "execute_decision(sync)"
            except Exception as e:
                return False, None, f"execute_decision failed: {repr(e)}"

        return False, None, "no open entrypoint on executor"

    def _call_close_executor(self, symbol: str, interval: str, meta: Dict[str, Any], direction: str, price: Any) -> Tuple[bool, Any, str]:
        """
        Öncelik: TradeExecutor.close_position_from_signal (sync)
        Fallback: TradeExecutor.close_position (sync) (price required)
        """
        fn = getattr(self.executor, "close_position_from_signal", None)
        if callable(fn):
            try:
                res = fn(symbol=symbol, interval=interval, meta=meta, direction=direction, price=price, exit_price=price)
                return True, res, "close_position_from_signal(sync)"
            except Exception as e:
                return False, None, f"close_position_from_signal failed: {repr(e)}"

        fn2 = getattr(self.executor, "close_position", None)
        if callable(fn2):
            try:
                res = fn2(symbol=symbol, price=float(_safe_float(price, 0.0)), reason="INTENT_CLOSE", interval=interval)
                return True, res, "close_position(sync)"
            except Exception as e:
                return False, None, f"close_position failed: {repr(e)}"

        return False, None, "no close entrypoint on executor"
    # -----------------------
    # Forwarding logic
    # -----------------------
    def _forward_close(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))
        intent_id = _safe_str(it.get("intent_id", ""))

        raw = it.get("raw", {})
        raw = raw if isinstance(raw, dict) else {}

        if not symbol:
            self._publish_event("close_skip", {"intent_id": intent_id, "symbol": symbol, "interval": interval, "why": "missing_symbol"})
            return

        raw_side = _safe_str(it.get("side", "")).strip()
        direction = self._normalize_direction(raw_side)
        if not direction:
            st = self._load_state()
            open_map = st.get("open", {}) if isinstance(st, dict) else {}
            if isinstance(open_map, dict) and isinstance(open_map.get(symbol), dict):
                prev = _safe_str(open_map[symbol].get("side", "long")).lower()
                direction = "LONG" if prev == "long" else "SHORT"
            else:
                direction = "LONG"

        close_price = self._extract_close_price(it)

        meta = {
            "reason": "ORCH_INTENT_CLOSE",
            "intent_id": intent_id,
            "source_pkg_id": source_pkg_id,
        }

        # whale context (optional)
        whale_dir = _safe_str(it.get("whale_dir", raw.get("whale_dir", "none")))
        whale_score = float(_safe_float(it.get("whale_score", raw.get("whale_score", 0.0)), 0.0))
        meta["whale_dir"] = whale_dir
        meta["whale_score"] = whale_score

        if self.dry_run and (not self.dryrun_call_executor):
            self._gate_mark_close(symbol, reason="intent_close", intent_id=intent_id)
            self._publish_event(
                "close_dry",
                {"intent_id": intent_id, "symbol": symbol, "interval": interval, "direction": direction, "price": close_price, "why": "dry_run"},
            )
            return

        called_ok, res, method_or_err = self._call_close_executor(
            symbol=symbol,
            interval=interval,
            meta=meta,
            direction=direction,
            price=close_price,
        )

        if called_ok and self._is_close_success(res):
            self._gate_mark_close(symbol, reason="intent_close", intent_id=intent_id)
            self._publish_event(
                "close_ok" if (not self.dry_run) else "close_dry_ok",
                {"intent_id": intent_id, "method": method_or_err, "symbol": symbol, "interval": interval, "direction": direction, "price": close_price},
            )
            return

        self._publish_event(
            "close_skip",
            {"intent_id": intent_id, "symbol": symbol, "interval": interval, "direction": direction, "price": close_price, "why": method_or_err},
        )

    def _forward_open(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))
        side = self._normalize_side(_safe_str(it.get("side", "long")))
        intent_id = _safe_str(it.get("intent_id", ""))

        raw = it.get("raw", {})
        raw = raw if isinstance(raw, dict) else {}

        if not symbol:
            self._publish_event("forward_skip", {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": "missing_symbol"})
            return

        allow, why_gate = self._gate_allow_open(symbol)
        if not allow:
            self._publish_event("forward_skip", {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": why_gate})
            return

        lev = int(_safe_float(it.get("recommended_leverage", raw.get("recommended_leverage", 5)), 5))
        npct = float(_safe_float(it.get("recommended_notional_pct", raw.get("recommended_notional_pct", 0.05)), 0.05))
        score = float(_safe_float(it.get("score", raw.get("score", 0.0)), 0.0))

        # UPDATED: raw fallback (MasterExecutor bazen raw içine de koyuyor)
        trail_pct = float(_safe_float(it.get("trail_pct", raw.get("trail_pct", os.getenv("TRAIL_PCT", "0.05"))), 0.05))
        stall_ttl_sec = int(_safe_float(it.get("stall_ttl_sec", raw.get("stall_ttl_sec", os.getenv("STALL_TTL_SEC", "0"))), 0.0))

        # whale context (preferred for TradeExecutor whale bias)
        whale_dir = _safe_str(it.get("whale_dir", raw.get("whale_dir", "none")))
        whale_score = float(_safe_float(it.get("whale_score", raw.get("whale_score", 0.0)), 0.0))
        w_min = float(_safe_float(it.get("w_min", raw.get("w_min", os.getenv("W_MIN", "0.0"))), 0.0))

        open_price = self._extract_open_price(it)
        if open_price is None:
            # bazı upstream'ler raw içine price koyabilir
            rp = _safe_float(raw.get("price", 0.0), 0.0)
            if rp > 0:
                open_price = float(rp)

        meta = {
            "reason": "ORCH_INTENT",
            "intent_id": intent_id,
            "score": score,
            "recommended_leverage": lev,
            "recommended_notional_pct": npct,
            "trail_pct": trail_pct,
            "stall_ttl_sec": stall_ttl_sec,
            "whale_dir": whale_dir,
            "whale_score": whale_score,
            "w_min": w_min,
            "source_pkg_id": source_pkg_id,
        }

        # kritik: TradeExecutor.open_position_from_signal meta’dan price bekliyor
        if open_price is not None and open_price > 0:
            meta["price"] = float(open_price)

        if self.dry_run and (not self.dryrun_call_executor):
            self._gate_mark_open(symbol, side, interval, intent_id)
            self._publish_event("forward_dry", {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": "dry_run"})
            return

        called_ok, res, method_or_err = self._call_open_executor(
            symbol=symbol,
            side=side,
            interval=interval,
            meta=meta,
        )

        if called_ok and self._is_open_success(res):
            self._gate_mark_open(symbol, side, interval, intent_id)
            self._publish_event(
                "forward_ok" if (not self.dry_run) else "forward_dry_ok",
                {"intent_id": intent_id, "method": method_or_err, "symbol": symbol, "side": side, "interval": interval, "price": open_price},
            )
            return

        self._publish_event(
            "forward_skip",
            {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": method_or_err or "open failed"},
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
        try:
            if self.selftest_reset_state:
                try:
                    self.r.delete(self.state_key)
                    self._publish_event("selftest_reset_state", {"state_key": self.state_key})
                except Exception:
                    pass

            stamp = str(int(time.time()))
            expected_prefix = f"selftest-{stamp}-"
            expected_intents: List[str] = []

            for i, sym in enumerate(self.selftest_symbols[:4], start=1):
                iid = f"{expected_prefix}{i}-{sym}"
                expected_intents.append(iid)
                pkg = {
                    "items": [
                        {
                            "symbol": sym,
                            "side": self.selftest_side,
                            "interval": self.selftest_interval,
                            "intent_id": iid,
                            "price": 1.0,
                        }
                    ]
                }
                self.r.xadd(self.in_stream, {"json": json.dumps(pkg, ensure_ascii=False)})

            deadline = time.time() + float(self.selftest_wait_sec or 2.0)
            seen: set[str] = set()

            while time.time() < deadline and len(seen) < len(expected_intents):
                rows = self._xreadgroup(">")
                if not rows:
                    time.sleep(0.05)
                    continue

                mids = [sid for sid, _ in rows]
                for sid, pkg in rows:
                    if not pkg:
                        continue
                    items = pkg.get("items") or []
                    if isinstance(items, list):
                        for it in items:
                            if isinstance(it, dict):
                                iid = _safe_str(it.get("intent_id", ""))
                                if iid.startswith(expected_prefix):
                                    seen.add(iid)
                    self._process_pkg(sid, pkg)

                self._ack(mids)
                time.sleep(0.02)

            rows = self.r.xrevrange(self.out_stream, max="+", min="-", count=500)

            ok = dry = skip = 0
            close_ok = close_dry = close_skip = 0
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
                if not iid.startswith(expected_prefix):
                    continue

                kind = _safe_str(ev.get("kind", ""))
                if kind in ("forward_ok", "forward_dry_ok"):
                    ok += 1
                elif kind == "forward_dry":
                    dry += 1
                elif kind == "forward_skip":
                    skip += 1
                    why = _safe_str(ev.get("why", ""))
                    reasons[why] = reasons.get(why, 0) + 1
                elif kind in ("close_ok", "close_dry_ok"):
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
                    "dryrun_call_executor": self.dryrun_call_executor,
                    "max_open": self.max_open,
                    "dedup_symbol": self.dedup_symbol,
                    "close_cooldown_sec": self.close_cooldown_sec,
                    "state_ttl_sec": int(self.state_ttl_sec or 0),
                    "seen_intents": sorted(list(seen)),
                    "counts": {
                        "forward_ok": ok,
                        "forward_dry": dry,
                        "forward_skip": skip,
                        "close_ok": close_ok,
                        "close_dry": close_dry,
                        "close_skip": close_skip,
                    },
                    "skip_reasons": reasons,
                },
            )
        except Exception as e:
            self._publish_event("selftest_error", {"err": repr(e)})

    # -----------------------
    # Main loop
    # -----------------------
    def run_forever(self) -> None:
        try:
            self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)
        except Exception:
            pass

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

        if self.selftest:
            self._publish_event("selftest_start", {"enabled": True})
            self._selftest_run()

        idle = 0
        last_cleanup = time.time()

        while True:
            rows = self._xreadgroup(">")
            if not rows:
                idle += 1
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

            if self.cleanup_every_sec > 0 and (time.time() - last_cleanup) >= self.cleanup_every_sec:
                self._maybe_periodic_cleanup()
                last_cleanup = time.time()

            time.sleep(0.05)


if __name__ == "__main__":
    IntentBridge().run_forever()
