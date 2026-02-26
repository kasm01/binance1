# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_bool(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return default
    t = str(v).strip().lower()
    if t in ("1", "true", "yes", "y", "on"):
        return True
    if t in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _env_str(k: str, default: str = "") -> str:
    v = os.getenv(k)
    if v is None:
        return default
    return str(v).strip()


def _env_int(k: str, default: int) -> int:
    try:
        return int(str(os.getenv(k, str(default))).strip())
    except Exception:
        return default


def _env_float(k: str, default: float) -> float:
    try:
        return float(str(os.getenv(k, str(default))).strip())
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


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return []


@dataclass
class ExecEvent:
    """
    What IntentBridge publishes to exec_events_stream
    """
    ts_utc: str
    kind: str                      # "open_intent" | "close_intent" | "reject_intent" | "noop"
    intent_id: str
    symbol: str
    interval: str
    side: str                      # "long" | "short" | "close"
    price: float
    score: float
    trail_pct: float = 0.0
    stall_ttl_sec: int = 0
    reason: str = ""
    raw: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "kind": self.kind,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "interval": self.interval,
            "side": self.side,
            "price": float(self.price),
            "score": float(self.score),
            "trail_pct": float(self.trail_pct or 0.0),
            "stall_ttl_sec": int(self.stall_ttl_sec or 0),
            "reason": self.reason,
            "raw": self.raw or {},
        }


class IntentBridge:
    """
    IN:  trade_intents_stream (MasterExecutor publishes {"ts_utc","count","items":[...]} )
    OUT: exec_events_stream   (downstream executor consumes)
    STATE: open_positions_state (hash/json; best-effort)

    Safety:
      - DRY_RUN=1: still emits exec_events_stream, but tags as dry_run in raw.
      - DRY_RUN=0: requires ARMED=1 and LIVE_KILL_SWITCH=0 and ARM_TOKEN>=16.

    Consumer group is pinned via env:
      BRIDGE_GROUP (default bridge_g)
      BRIDGE_CONSUMER (default bridge_1)
      BRIDGE_GROUP_START_ID (default "$")
    """

    def __init__(self) -> None:
        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

        # Streams
        self.in_stream = os.getenv("BRIDGE_IN_STREAM", os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"))
        self.out_stream = os.getenv("BRIDGE_OUT_STREAM", os.getenv("EXEC_EVENTS_STREAM", "exec_events_stream"))

        # Consumer group
        self.group = os.getenv("BRIDGE_GROUP", "bridge_g")
        self.consumer = os.getenv("BRIDGE_CONSUMER", "bridge_1")
        self.group_start_id = os.getenv("BRIDGE_GROUP_START_ID", "$")
        self.drain_pending = _env_bool("BRIDGE_DRAIN_PENDING", False)

        # Read tuning
        self.read_block_ms = _env_int("BRIDGE_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("BRIDGE_BATCH_COUNT", 50)

        # Output trim
        self.out_maxlen = _env_int("BRIDGE_OUT_MAXLEN", 5000)

        # State
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")

        # Policy knobs
        self.require_price = _env_bool("BRIDGE_REQUIRE_PRICE", True)

        # Live safety policy
        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")

        self.publish_allowed = bool(
            self.dry_run_env or (self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16))
        )

        if (not self.dry_run_env) and (not self.publish_allowed):
            print(
                f"[IntentBridge][SAFE] DRY_RUN=0 but publish blocked: "
                f"ARMED={self.armed} KILL={self.kill_switch} ARM_TOKEN_len={len(self.arm_token)}"
            )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        print(
            f"[IntentBridge] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"require_price={self.require_price} publish_allowed={self.publish_allowed} dry_run={self.dry_run_env} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[IntentBridge] XGROUP created: stream={stream} group={group} start_id={start_id}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise
        except Exception:
            return

    def _ack(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self.r.xack(self.in_stream, self.group, *ids)
        except Exception:
            pass

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
                self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)
                return []
            return []
        except Exception:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        if not resp:
            return out

        for _stream_name, entries in resp:
            for sid, fields in entries:
                s = fields.get("json")
                if not s:
                    out.append((sid, {}))
                    continue
                try:
                    pkg = json.loads(s)
                    out.append((sid, pkg if isinstance(pkg, dict) else {}))
                except Exception:
                    out.append((sid, {}))
        return out

    def _extract_items(self, pkg: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = pkg.get("items") if isinstance(pkg, dict) else None
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        return []

    def _norm_side(self, side: Any) -> str:
        s = str(side or "").strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        if s in ("close", "flat", "exit"):
            return "close"
        return s or "long"

    def _intent_price(self, it: Dict[str, Any]) -> float:
        # price may appear in multiple locations; prefer top-level
        p = _safe_float(it.get("price", 0.0), 0.0)
        if p > 0:
            return float(p)
        raw1 = it.get("raw")
        if isinstance(raw1, dict):
            p2 = _safe_float(raw1.get("price", 0.0), 0.0)
            if p2 > 0:
                return float(p2)
            raw2 = raw1.get("raw")
            if isinstance(raw2, dict):
                p3 = _safe_float(raw2.get("price", 0.0), 0.0)
                if p3 > 0:
                    return float(p3)
        return 0.0

    def _state_get_open(self) -> Dict[str, Any]:
        try:
            s = self.r.get(self.state_key)
            if not s:
                return {}
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _state_set_open(self, obj: Dict[str, Any]) -> None:
        try:
            self.r.set(self.state_key, json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass
    def _publish_exec_event(self, ev: ExecEvent) -> Optional[str]:
        if not self.publish_allowed:
            return None

        payload = ev.to_dict()
        # tag dry_run for downstream observability
        try:
            payload["raw"] = payload.get("raw") or {}
            payload["raw"]["dry_run"] = bool(self.dry_run_env)
        except Exception:
            pass

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=self.out_maxlen,
                approximate=True,
            )
            return sid
        except Exception:
            return None

    def _handle_one_intent(self, it: Dict[str, Any], state: Dict[str, Any]) -> Optional[ExecEvent]:
        intent_id = _safe_str(it.get("intent_id", ""), "")
        symbol = _safe_str(it.get("symbol", "")).upper()
        interval = _safe_str(it.get("interval", ""), "5m")
        side = self._norm_side(it.get("side", "long"))
        score = float(_safe_float(it.get("score", 0.0), 0.0))

        trail_pct = float(_safe_float(it.get("trail_pct", 0.0), 0.0))
        stall_ttl_sec = int(_safe_float(it.get("stall_ttl_sec", 0), 0))

        price = float(self._intent_price(it))

        if not intent_id:
            # fallback: accept missing id but generate one
            intent_id = _safe_str(it.get("id", ""), "") or _safe_str(it.get("uuid", ""), "")
            if not intent_id:
                intent_id = _safe_str(f"bridge-{int(time.time()*1000)}", "")

        if self.require_price and side != "close" and price <= 0.0:
            return ExecEvent(
                ts_utc=_now_utc_iso(),
                kind="reject_intent",
                intent_id=intent_id,
                symbol=symbol,
                interval=interval,
                side=side,
                price=float(price),
                score=float(score),
                trail_pct=trail_pct,
                stall_ttl_sec=stall_ttl_sec,
                reason="missing_price",
                raw={"intent": it},
            )

        # very simple open/close state:
        # state is a dict keyed by symbol -> {"side": "...", "ts_utc": "...", "intent_id": "..."}
        opened = state.get(symbol) if isinstance(state.get(symbol), dict) else None

        if side == "close":
            if not opened:
                return ExecEvent(
                    ts_utc=_now_utc_iso(),
                    kind="noop",
                    intent_id=intent_id,
                    symbol=symbol,
                    interval=interval,
                    side="close",
                    price=float(price),
                    score=float(score),
                    reason="close_no_open",
                    raw={"intent": it},
                )
            # close existing
            try:
                state.pop(symbol, None)
            except Exception:
                pass
            return ExecEvent(
                ts_utc=_now_utc_iso(),
                kind="close_intent",
                intent_id=intent_id,
                symbol=symbol,
                interval=interval,
                side="close",
                price=float(price),
                score=float(score),
                reason="close",
                raw={"intent": it, "prev_open": opened or {}},
            )

        # open intent:
        # if already open on symbol => noop (or could replace; for now noop)
        if opened:
            return ExecEvent(
                ts_utc=_now_utc_iso(),
                kind="noop",
                intent_id=intent_id,
                symbol=symbol,
                interval=interval,
                side=side,
                price=float(price),
                score=float(score),
                trail_pct=trail_pct,
                stall_ttl_sec=stall_ttl_sec,
                reason="already_open",
                raw={"intent": it, "open": opened},
            )

        # mark open
        state[symbol] = {"side": side, "ts_utc": _now_utc_iso(), "intent_id": intent_id, "interval": interval}
        return ExecEvent(
            ts_utc=_now_utc_iso(),
            kind="open_intent",
            intent_id=intent_id,
            symbol=symbol,
            interval=interval,
            side=side,
            price=float(price),
            score=float(score),
            trail_pct=trail_pct,
            stall_ttl_sec=stall_ttl_sec,
            reason="open",
            raw={"intent": it},
        )

    def _handle_pkg(self, sid: str, pkg: Dict[str, Any]) -> int:
        items = self._extract_items(pkg)
        if not items:
            return 0

        state = self._state_get_open()

        n_pub = 0
        for it in items:
            ev = self._handle_one_intent(it, state)
            if ev is None:
                continue
            out_id = self._publish_exec_event(ev)
            if out_id:
                n_pub += 1

        # persist state (best-effort)
        self._state_set_open(state)
        return n_pub
    def run_forever(self) -> None:
        if self.drain_pending:
            print("[IntentBridge] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break

                mids = [sid for sid, _ in rows]
                for sid, pkg in rows:
                    n = self._handle_pkg(sid, pkg)
                    if n:
                        print(f"[IntentBridge] (PEL) published {n} -> {self.out_stream} source={sid}")
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
                n = self._handle_pkg(sid, pkg)
                if n:
                    print(f"[IntentBridge] published {n} -> {self.out_stream} source={sid}")

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    IntentBridge().run_forever()
