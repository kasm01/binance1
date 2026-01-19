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
        return int(os.getenv(k, str(default)).strip())
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
    Consumes trade_intents_stream, forwards intents to TradeExecutor.

    IN:  trade_intents_stream (MASTER_OUT_STREAM)
    OUT: optional exec_events_stream (log/trace)

    Safe-by-default: if it cannot call executor methods, it only logs.
    """

    def _gate_allow_open(self, symbol: str, side: str, interval: str, intent_id: str):
        """Return (allow: bool, why: str). Uses Redis key open_positions_state by default.
        Optional TTL cleanup via BRIDGE_OPEN_TTL_SEC (0=disabled).
        """
        try:
            max_open = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
        except Exception:
            max_open = 3

        dedup_symbol = os.getenv("DEDUP_SYMBOL_OPEN", "1").strip().lower() not in ("0","false","no","off")
        state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")

        try:
            ttl_sec = int(os.getenv("BRIDGE_OPEN_TTL_SEC", "0"))
        except Exception:
            ttl_sec = 0

        try:
            raw_state = self.r.get(state_key)
            state = json.loads(raw_state) if raw_state else {}
        except Exception:
            state = {}

        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}
        # TTL cleanup (optional)
        if ttl_sec and ttl_sec > 0:
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
                    if (t0 is None) or ((now - t0).total_seconds() > ttl_sec):
                        open_map.pop(sym, None)
                        dirty = True
                if dirty:
                    self.r.set(state_key, json.dumps({"open": open_map}))
            except Exception:
                pass


        if dedup_symbol and symbol in open_map:
            return (False, f"dedup_symbol: {symbol} already open")

        if len(open_map) >= max_open:
            return (False, f"max_open: {len(open_map)} >= {max_open}")

        return (True, "ok")


    def _gate_mark_open(self, symbol: str, side: str, interval: str, intent_id: str) -> None:
        state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")

        try:
            raw_state = self.r.get(state_key)
            state = json.loads(raw_state) if raw_state else {}
        except Exception:
            state = {}

        open_map = state.get("open", {}) if isinstance(state, dict) else {}
        if not isinstance(open_map, dict):
            open_map = {}

        open_map[symbol] = {
            "side": side,
            "interval": interval,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "intent_id": intent_id,
        }

        try:
            # state yaz
            self.r.set(state_key, json.dumps({"open": open_map}))

            # TTL (auto-expire)

        except Exception:
            pass



    def __init__(self) -> None:
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        self.in_stream = os.getenv("BRIDGE_IN_STREAM", os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"))
        self.out_stream = os.getenv("BRIDGE_OUT_STREAM", "exec_events_stream")

        self.read_block_ms = _env_int("BRIDGE_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("BRIDGE_BATCH_COUNT", 20)
        self.last_id = os.getenv("BRIDGE_START_ID", "0-0")

        self.dry_run = os.getenv("DRY_RUN", "1").lower() in ("1", "true", "yes", "on")
        self.max_parallel = _env_int("BRIDGE_MAX_PARALLEL", 1)  # şimdilik seri

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
        )

        # TradeExecutor'ı burada import ediyoruz (PYTHONPATH şart!)
        from core.trade_executor import TradeExecutor  # noqa
        from core.risk_manager import RiskManager  # noqa

        # main.py içindeki init daha kapsamlı; burada "bridge mode" minimal init yapıyoruz.
        # Eğer TradeExecutor ctor imzası farklıysa burası hata verir: o durumda stack'i buraya atarsın, düzeltirim.
        self.risk_manager = RiskManager()
        self.executor = TradeExecutor(
            client=None,
            risk_manager=self.risk_manager,
            position_manager=None,
            dry_run=self.dry_run,
        )

        print(
            f"[IntentBridge] started. in={self.in_stream} out={self.out_stream} "
            f"dry_run={self.dry_run} redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _parse(self, sid: str, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
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

    def _xread(self) -> List[Tuple[str, Dict[str, Any]]]:
        resp = self.r.xread({self.in_stream: self.last_id}, count=self.batch_count, block=self.read_block_ms)
        out: List[Tuple[str, Dict[str, Any]]] = []
        if not resp:
            return out
        for _name, entries in resp:
            for sid, fields in entries:
                pkg = self._parse(sid, fields)
                if pkg:
                    out.append((sid, pkg))
                self.last_id = sid
        return out

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

    def _forward_one(self, it: Dict[str, Any], source_pkg_id: str) -> None:
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""))

        # --- gate: max open + symbol dedupe (+ optional TTL cleanup) ---
        side = self._normalize_side(_safe_str(it.get("side", "long")))
        intent_id = _safe_str(it.get("intent_id", ""))

        try:
            allow, why = self._gate_allow_open(symbol, side, interval, intent_id)
            if not allow:
                self._publish_event(
                    "forward_skip",
                    {"intent_id": intent_id, "symbol": symbol, "side": side, "interval": interval, "why": why},
                )
                return
        except Exception:
            pass

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

        called = False
        err = None

        # 1) open_position_from_signal (varsa en iyi)
        fn = getattr(self.executor, "open_position_from_signal", None)
        if callable(fn):
            try:
                # projede genelde qty/notional hesap executor içinde; biz notional_pct meta olarak veriyoruz
                res = fn(symbol=symbol, side=side, interval=interval, meta=meta)
                called = True
                # GATE_MARK_OPEN
                try:
                    self._gate_mark_open(symbol, side, interval, intent_id)
                except Exception:
                    pass

                self._publish_event("forward_ok", {"intent_id": intent_id, "method": "open_position_from_signal", "symbol": symbol, "side": side})
                return
            except Exception as e:
                err = repr(e)

        # 2) open_position / execute_trade (async ise asyncio.run ile çalıştır)
        for name in ("open_position", "execute_trade"):
            fn2 = getattr(self.executor, name, None)
            if not callable(fn2):
                continue
            try:
                if inspect.iscoroutinefunction(fn2):
                    # Çoğu projede open_position(*args, **kwargs) / execute_trade(*args, **kwargs) kabul ediyor
                    asyncio.run(fn2(symbol=symbol, side=side, interval=interval, meta=meta))
                    called = True
                    # GATE_MARK_OPEN
                    try:
                        self._gate_mark_open(symbol, side, interval, intent_id)
                    except Exception:
                        pass

                    self._publish_event("forward_ok", {"intent_id": intent_id, "method": f"{name}(asyncio.run)", "symbol": symbol, "side": side})
                    return
                else:
                    # sync ise direkt dene
                    fn2(symbol=symbol, side=side, interval=interval, meta=meta)
                    called = True
                    # GATE_MARK_OPEN
                    try:
                        self._gate_mark_open(symbol, side, interval, intent_id)
                    except Exception:
                        pass

                    self._publish_event("forward_ok", {"intent_id": intent_id, "method": f"{name}(sync)", "symbol": symbol, "side": side})
                    return
            except TypeError as e:
                # imza uymadıysa tekrar denemek yerine logla
                err = err or f"{name} signature mismatch: {e}"
            except Exception as e:
                err = err or f"{name} failed: {repr(e)}"

        # 3) hiçbirini çağırmadık → sadece logla hiçbirini çağırmadık → sadece logla
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

    def run_forever(self) -> None:
        idle = 0
        while True:
            pkgs = self._xread()
            if not pkgs:
                idle += 1
                if idle % 30 == 0:
                    print(f"[IntentBridge] idle... last_id={self.last_id}")
                continue
            idle = 0

            for sid, pkg in pkgs:
                items = pkg.get("items") or []
                if not isinstance(items, list) or not items:
                    continue
                for it in items:
                    if isinstance(it, dict):
                        self._forward_one(it, source_pkg_id=sid)


if __name__ == "__main__":
    IntentBridge().run_forever()
