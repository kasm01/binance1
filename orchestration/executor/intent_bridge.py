# -*- coding: utf-8 -*-
from __future__ import annotations

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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
        return s
    except Exception:
        return default


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


class IntentBridge:
    """
    IN : trade_intents_stream  (MasterExecutor publishes {"ts_utc","count","items":[TradeIntent,...]})
    OUT: exec_events_stream    (Executor consumes)

    Responsibilities:
      - Maintain open_positions_state with TTL
      - Enforce max_open
      - Dedup per symbol (optional)
      - Apply close cooldown (avoid immediate reopen spam)
      - Publish intents downstream (safe-gated)
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

        # State
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")
        self.state_ttl_sec = _env_int("BRIDGE_STATE_TTL_SEC", 900)
        self.open_ttl_sec = _env_int("BRIDGE_OPEN_TTL_SEC", 900)  # per-symbol "expires_at"
        self.cleanup_every_sec = max(1, _env_int("BRIDGE_CLEANUP_EVERY_SEC", 5))

        # Limits / dedup
        self.max_open = _env_int("BRIDGE_MAX_OPEN", _env_int("MAX_OPEN_POSITIONS", 3))
        self.dedup_symbol_open = _env_bool("DEDUP_SYMBOL_OPEN", True)

        # Close cooldown
        self.close_cooldown_sec = _env_int("BRIDGE_CLOSE_COOLDOWN_SEC", 30)
        self.close_cd_key = f"{self.state_key}:close_cd"

        # Require valid price for publish (default true)
        self.require_price = _env_bool("BRIDGE_REQUIRE_PRICE", True)

        # Dry-run behavior knobs
        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.dryrun_call_executor = _env_bool("BRIDGE_DRYRUN_CALL_EXECUTOR", True)
        self.dryrun_write_state = _env_bool("BRIDGE_DRYRUN_WRITE_STATE", False)

        # LIVE safety (publish gate)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")

        self.publish_allowed = bool(
            (self.dry_run_env and self.dryrun_call_executor)
            or ((not self.dry_run_env) and self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16))
        )

        self._last_cleanup_ts = 0.0
        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        print(
            f"[IntentBridge] init ok. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} start_id={self.group_start_id} "
            f"drain_pending={self.drain_pending} max_open={self.max_open} dedup_symbol_open={self.dedup_symbol_open} "
            f"state_key={self.state_key} ttl={self.state_ttl_sec}s prune_every={self.cleanup_every_sec}s "
            f"require_price={self.require_price} publish_allowed={self.publish_allowed} "
            f"dry_run={self.dry_run_env} dryrun_write_state={self.dryrun_write_state} "
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

    def _parse_pkg(self, stream_id: str, fields: Dict[str, str]) -> Optional[Dict[str, Any]]:
        s = fields.get("json")
        if not s:
            return None
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                obj.setdefault("ts_utc", _now_utc_iso())
                obj["_source_stream_id"] = stream_id
                return obj
        except Exception:
            return None
        return None

    def _get_state(self) -> Dict[str, Any]:
        try:
            s = self.r.get(self.state_key)
            if not s:
                return {}
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _set_state(self, st: Dict[str, Any]) -> None:
        # In DRY_RUN we may want to not touch state (configurable)
        if self.dry_run_env and (not self.dryrun_write_state):
            return
        try:
            payload = json.dumps(st or {}, ensure_ascii=False)
            self.r.set(self.state_key, payload, ex=max(1, int(self.state_ttl_sec)))
        except Exception:
            pass

    def _open_count(self, st: Dict[str, Any]) -> int:
        try:
            return len([1 for _k, v in (st or {}).items() if isinstance(v, dict)])
        except Exception:
            return 0

    def _is_open(self, st: Dict[str, Any], symbol: str) -> bool:
        try:
            return isinstance((st or {}).get(symbol), dict)
        except Exception:
            return False

    def _action_of(self, item: Dict[str, Any]) -> str:
        # fail-open detection: OPEN if long/short; CLOSE if explicit close
        side = _safe_str(item.get("side", "")).strip().lower()
        action = _safe_str(item.get("action", "")).strip().lower()
        itype = _safe_str(item.get("intent_type", "")).strip().lower()

        if side == "close" or action == "close" or itype == "close":
            return "close"
        if side in ("long", "short", "buy", "sell"):
            return "open"
        # default: open (safe for backward compatibility)
        return "open"

    def _norm_side(self, item: Dict[str, Any]) -> str:
        s = _safe_str(item.get("side", "long")).strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        if s == "close":
            return "close"
        return s or "long"

    def _price_of(self, item: Dict[str, Any]) -> float:
        p = _safe_float(item.get("price", 0.0), 0.0)
        return float(p) if p > 0 else 0.0

    def _close_cd_get(self, symbol: str) -> float:
        try:
            v = self.r.hget(self.close_cd_key, symbol)
            return float(v) if v else 0.0
        except Exception:
            return 0.0

    def _close_cd_set(self, symbol: str, ts: float) -> None:
        try:
            self.r.hset(self.close_cd_key, symbol, str(float(ts)))
            # keep cooldown hash small; expire > cooldown
            self.r.expire(self.close_cd_key, max(10, int(self.close_cooldown_sec * 2)))
        except Exception:
            pass

    def _prune_state(self, st: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Remove expired symbols based on per-symbol expires_at.
        Returns: (new_state, removed_count)
        """
        now = time.time()
        removed = 0
        out: Dict[str, Any] = {}
        for sym, v in (st or {}).items():
            if not isinstance(v, dict):
                continue
            exp = _safe_float(v.get("expires_at", 0.0), 0.0)
            if exp > 0.0 and exp < now:
                removed += 1
                continue
            out[sym] = v
        return out, removed

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if (now - self._last_cleanup_ts) < float(self.cleanup_every_sec):
            return
        self._last_cleanup_ts = now

        st = self._get_state()
        st2, removed = self._prune_state(st)
        if removed > 0:
            self._set_state(st2)

    def _apply_state_update(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update open state based on item.
        Returns: (state_changed, reason)
        """
        sym = _safe_str(item.get("symbol", "")).upper().strip()
        interval = _safe_str(item.get("interval", "5m")).strip()
        intent_id = _safe_str(item.get("intent_id", "")).strip()
        action = self._action_of(item)
        side = self._norm_side(item)

        if not sym:
            return False, "no_symbol"

        st = self._get_state()
        st, _ = self._prune_state(st)

        if action == "close":
            if self._is_open(st, sym):
                st.pop(sym, None)
                self._close_cd_set(sym, time.time())
                self._set_state(st)
                return True, "closed"
            # still set cooldown to reduce spam, but don't rewrite big state
            self._close_cd_set(sym, time.time())
            return False, "close_not_open"

        # OPEN path
        # close cooldown gate
        last_close = self._close_cd_get(sym)
        if last_close > 0 and (time.time() - last_close) < float(self.close_cooldown_sec):
            return False, "cooldown"

        if self.dedup_symbol_open and self._is_open(st, sym):
            return False, "dedup_open"

        if self._open_count(st) >= int(self.max_open):
            return False, "max_open"

        st[sym] = {
            "side": side if side in ("long", "short") else "long",
            "ts_utc": _now_utc_iso(),
            "intent_id": intent_id or "",
            "interval": interval,
            "expires_at": float(time.time() + max(1, int(self.open_ttl_sec))),
        }
        self._set_state(st)
        return True, "opened"
    def _publish_exec(self, source_stream_id: str, item: Dict[str, Any]) -> Optional[str]:
        if not self.publish_allowed:
            return None

        sym = _safe_str(item.get("symbol", "")).upper().strip()
        interval = _safe_str(item.get("interval", "5m")).strip()
        side = self._norm_side(item)
        intent_id = _safe_str(item.get("intent_id", "")).strip()
        score = _safe_float(item.get("score", 0.0), 0.0)
        price = self._price_of(item)

        if self.require_price and price <= 0.0 and (side != "close"):
            return None

        payload = {
            "ts_utc": _now_utc_iso(),
            "source_intents_id": source_stream_id,
            "count": 1,
            "items": [
                {
                    "symbol": sym,
                    "interval": interval,
                    "side": side,
                    "intent_id": intent_id,
                    "score": float(score),
                    "price": float(price),
                    # pass-through knobs if exist
                    "trail_pct": _safe_float(item.get("trail_pct", 0.0), 0.0),
                    "stall_ttl_sec": _env_int("STALL_TTL_SEC", _env_int("STALL_TTL_SEC", 0)),
                    "raw": dict(item),
                }
            ],
        }

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=5000,
                approximate=True,
            )
            return sid
        except Exception:
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
                pkg = self._parse_pkg(sid, fields) or {}
                out.append((sid, pkg if isinstance(pkg, dict) else {}))
        return out

    def run_forever(self) -> None:
        if self.drain_pending:
            print("[IntentBridge] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break
                mids = [sid for sid, _ in rows]
                for sid, pkg in rows:
                    items = pkg.get("items") or []
                    if not isinstance(items, list):
                        continue
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        self._maybe_cleanup()
                        self._apply_state_update(it)
                        out_id = self._publish_exec(source_stream_id=sid, item=it)
                        if out_id:
                            print(f"[IntentBridge] (PEL) published 1 -> {self.out_stream} source={sid}")
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
                self._maybe_cleanup()
                continue
            idle = 0

            mids = [sid for sid, _ in rows]
            for sid, pkg in rows:
                items = pkg.get("items") or []
                if not isinstance(items, list):
                    continue

                for it in items:
                    if not isinstance(it, dict):
                        continue

                    self._maybe_cleanup()
                    _changed, _reason = self._apply_state_update(it)
                    out_id = self._publish_exec(source_stream_id=sid, item=it)
                    if out_id:
                        print(f"[IntentBridge] published 1 -> {self.out_stream} source={sid}")

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    IntentBridge().run_forever()
