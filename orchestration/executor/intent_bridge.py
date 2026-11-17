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
        return str(x)
    except Exception:
        return default


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}



def _json_dumps_safe(x: Any, *, limit: int = 12000) -> str:
    """
    raw pass-through'u şişirmemek için truncate'lı json.
    """
    try:
        s = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    if len(s) > int(limit):
        return s[: int(limit)] + "...(trunc)"
    return s


def sleep_backoff(idle_count: int, *, base: float = 0.02, cap: float = 0.50) -> None:
    """
    Idle durumunda CPU yakmamak için küçük backoff.
    """
    try:
        t = min(float(cap), float(base) * max(1, int(idle_count)))
        time.sleep(float(t))
    except Exception:
        time.sleep(0.05)


class IntentBridge:
    """
    IN : trade_intents_stream  (MasterExecutor publishes {"ts_utc","count","items":[TradeIntent,...]})
    OUT: exec_events_stream    (Executor consumes)

    Responsibilities:
      - Validate transport-level payload requirements
      - Publish intents downstream (safe-gated)
      - Never decide position/cooldown/max-open execution risk
    """

    def __init__(self) -> None:
        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        self.socket_timeout = _env_int("REDIS_SOCKET_TIMEOUT", 5)
        self.socket_connect_timeout = _env_int("REDIS_SOCKET_CONNECT_TIMEOUT", 5)

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
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


        # Require valid price for publish (default true)

        # Stream publish tuning
        self.out_maxlen = _env_int("BRIDGE_OUT_MAXLEN", 5000)
        self.out_approx = _env_bool("BRIDGE_OUT_APPROX", True)

        # Raw pass-through controls
        self.pass_raw = _env_bool("BRIDGE_PASS_RAW", True)
        self.raw_max_chars = _env_int("BRIDGE_RAW_MAX_CHARS", 12000)

        # Dry-run behavior knobs
        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.dryrun_call_executor = _env_bool("BRIDGE_DRYRUN_CALL_EXECUTOR", True)
        self.dryrun_write_state = _env_bool("BRIDGE_DRYRUN_WRITE_STATE", False)

        # SHORT1 entry-confirmation micro exhaustion gate.
        # Reuses V44 public websocket telemetry; zero extra exchange API calls.
        # Default is OFF and dry-run-only. Missing telemetry fails open but is recorded.
        self.short_micro_gate_enable = _env_bool("BRIDGE_SHORT_MICRO_GATE_ENABLE", False)
        self.short_micro_gate_dryrun_only = _env_bool("BRIDGE_SHORT_MICRO_GATE_DRYRUN_ONLY", True)
        self.short_micro_gate_fail_open = _env_bool("BRIDGE_SHORT_MICRO_GATE_FAIL_OPEN", True)
        self.short_micro_policy_stream = _env_str(
            "BRIDGE_SHORT_MICRO_POLICY_STREAM",
            "telemetry:short1:inj:entry_hazard_orderbook_v44_shadow",
        )
        self.short_micro_telemetry_stream = _env_str(
            "BRIDGE_SHORT_MICRO_TELEMETRY_STREAM",
            "telemetry:short1:micro_entry_gate:v1",
        )
        self.short_micro_flow5_max = _safe_float(os.getenv("BRIDGE_SHORT_MICRO_FLOW5_MAX", "-0.40"), -0.40)
        self.short_micro_delta_min = _safe_float(os.getenv("BRIDGE_SHORT_MICRO_DELTA_5_20_MIN", "0.20"), 0.20)
        self.short_micro_wait_sec = max(0.0, _safe_float(os.getenv("BRIDGE_SHORT_MICRO_WAIT_SEC", "22"), 22.0))
        self.short_micro_maxlen = max(1000, _env_int("BRIDGE_SHORT_MICRO_TELEMETRY_MAXLEN", 20000))

        # LIVE safety (publish gate)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")

        self.publish_allowed = bool(
            (self.dry_run_env and self.dryrun_call_executor)
            or ((not self.dry_run_env) and self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16))
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        print(
            f"[IntentBridge] init ok. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} start_id={self.group_start_id} "
            f"drain_pending={self.drain_pending} mode=transport_only "
            f"publish_allowed={self.publish_allowed} "
            f"dry_run={self.dry_run_env} dryrun_write_state={self.dryrun_write_state} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        """
        Stream + XGROUP create (idempotent).
        """
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

    def _candidate_minute_ms(self, item: Dict[str, Any]) -> int:
        raw = _safe_dict(item.get("raw"))
        ts = _safe_str(raw.get("ts_utc", ""), "").strip() or _safe_str(item.get("ts_utc", ""), "").strip()
        if not ts:
            return 0
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000) // 60000 * 60000
        except Exception:
            return 0

    def _emit_short_micro_gate(self, payload: Dict[str, Any]) -> None:
        try:
            self.r.xadd(
                self.short_micro_telemetry_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=int(self.short_micro_maxlen),
                approximate=True,
            )
        except Exception:
            pass

    def _short_micro_entry_gate(self, item: Dict[str, Any]) -> Tuple[bool, str]:
        if not self.short_micro_gate_enable:
            return True, "micro_gate_disabled"
        if self.short_micro_gate_dryrun_only and not self.dry_run_env:
            return True, "micro_gate_dryrun_only"
        if self._norm_side(item) != "short":
            return True, "micro_gate_not_short"

        minute_ms = self._candidate_minute_ms(item)
        base = {
            "schema_version": "short1_micro_entry_gate.v1",
            "ts_utc": _now_utc_iso(),
            "intent_id": _safe_str(item.get("intent_id", "")),
            "symbol": _safe_str(item.get("symbol", "")).upper(),
            "candidate_minute_ms": int(minute_ms),
            "flow5_max": float(self.short_micro_flow5_max),
            "delta_5_20_min": float(self.short_micro_delta_min),
            "dry_run": bool(self.dry_run_env),
        }
        if minute_ms <= 0:
            base.update({"decision": "MISSING", "reason": "candidate_minute_missing"})
            self._emit_short_micro_gate(base)
            return bool(self.short_micro_gate_fail_open), "micro_candidate_minute_missing"

        # V44 emits the 20S_HAZARD record around second 20. Wait only when the
        # intent arrives earlier in the same minute; no exchange calls are made.
        target = minute_ms / 1000.0 + 20.20
        now = time.time()
        if now < target:
            time.sleep(min(self.short_micro_wait_sec, max(0.0, target - now)))

        lo = f"{minute_ms + 19000}-0"
        hi = f"{minute_ms + 29999}-999999"
        matched = None
        for _ in range(3):
            try:
                rows = self.r.xrevrange(self.short_micro_policy_stream, max=hi, min=lo, count=12)
            except Exception:
                rows = []
            for sid, fields in rows:
                try:
                    obj = json.loads(fields.get("json") or "{}")
                except Exception:
                    continue
                if str(obj.get("stage") or "") != "20S_HAZARD":
                    continue
                if int(_safe_float(obj.get("minute_ms"), 0.0)) != int(minute_ms):
                    continue
                if _safe_str(obj.get("symbol", "")).upper() != _safe_str(item.get("symbol", "")).upper():
                    continue
                matched = (sid, obj)
                break
            if matched is not None:
                break
            time.sleep(0.35)

        if matched is None:
            base.update({"decision": "MISSING", "reason": "v44_20s_missing"})
            self._emit_short_micro_gate(base)
            return bool(self.short_micro_gate_fail_open), "micro_v44_missing"

        sid, obj = matched
        s5 = _safe_dict(obj.get("snapshot_5s"))
        dec = _safe_dict(obj.get("decision"))
        f5 = _safe_float(s5.get("flow"), _safe_float(dec.get("flow_5s"), 0.0))
        f20 = _safe_float(dec.get("flow_20s"), 0.0)
        delta = _safe_float(dec.get("flow_delta_5_20"), f20 - f5)
        reject = bool(f5 <= float(self.short_micro_flow5_max) and delta >= float(self.short_micro_delta_min))
        base.update({
            "v44_stream_id": str(sid),
            "v44_event_id": _safe_str(obj.get("event_id", "")),
            "flow_5s": float(f5),
            "flow_20s": float(f20),
            "flow_delta_5_20": float(delta),
            "decision": "REJECT" if reject else "READY",
            "reason": "sell_exhaustion_rebound" if reject else "micro_pass",
        })
        self._emit_short_micro_gate(base)
        return (not reject), str(base["reason"])

    def _publish_exec(
        self,
        source_stream_id: str,
        item: Dict[str, Any],
        reason_hint: str = "",
    ) -> Optional[str]:
        """
        OUT stream'e tek item publish.

        Kurallar:
          - open event için valid price gerekir
          - close event'te price 0 olabilir
          - publish log reason'ı payload / hint ile belirlenir
        """
        if not self.publish_allowed:
            return None

        sym = _safe_str(item.get("symbol", "")).upper().strip()
        interval = _safe_str(item.get("interval", "5m")).strip()
        side = self._norm_side(item)
        intent_id = _safe_str(item.get("intent_id", "")).strip()
        if sym == "INJUSDT" and side == "short":
            from orchestration.context.short1_strict_market_contract import check
            ok, why = check(item)
            if not ok:
                # Defense in depth: do not forward a noncanonical research
                # or default-filled SHORT1 candidate to executor events.
                return None

        score = _safe_float(item.get("score", 0.0), 0.0)
        price = self._price_of(item)

        if not sym:
            return None

        payload_item: Dict[str, Any] = {
            "symbol": sym,
            "interval": interval,
            "side": side,
            "intent_id": intent_id,
            "source_signal_event_id": str(item.get("source_signal_event_id") or ""),
            "source_completed_close_ms": int(item.get("source_completed_close_ms") or 0),
            "score": float(score),
            "model_score": float(_safe_float(item.get("model_score", score), score)),
            "ranking_score": float(_safe_float(item.get("ranking_score", score), score)),
            "model_provider": str(item.get("model_provider") or "hybrid_coin_specific"),
            "price": float(price),
            "execution_source": "orchestrated",
            "market_context_passed": bool(item.get("market_context_passed", False)),
            "market_context": dict(item.get("market_context") or {}) if isinstance(item.get("market_context"), dict) else {},
            "model_score_gate_state": str(item.get("model_score_gate_state") or "REJECT"),
            "model_score_gate_passed": bool(item.get("model_score_gate_passed", False)),
            "model_score_gate_threshold": float(_safe_float(item.get("model_score_gate_threshold", 0.0), 0.0)),
            "model_score_gate_soft_threshold": float(_safe_float(item.get("model_score_gate_soft_threshold", 0.0), 0.0)),
            "soft_context_required": bool(item.get("soft_context_required", False)),
        }

        if self.pass_raw:
            payload_item["raw"] = _json_dumps_safe(_safe_dict(item), limit=self.raw_max_chars)

        payload = {
            "ts_utc": _now_utc_iso(),
            "source_intents_id": source_stream_id,
            "count": 1,
            "items": [payload_item],
        }

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=int(self.out_maxlen),
                approximate=bool(self.out_approx),
            )
        except Exception:
            return None

        publish_reason = str(reason_hint or "").strip().lower()
        if not publish_reason:
            if side == "close":
                publish_reason = "closed"
            elif side in ("long", "short"):
                publish_reason = "opened"
            else:
                publish_reason = "published"

        print(
            f"[IntentBridge] published 1 -> {self.out_stream} "
            f"source={source_stream_id} reason={publish_reason}"
        )
        return sid

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
    def _process_rows(self, rows: List[Tuple[str, Dict[str, Any]]], *, is_pel: bool = False) -> None:
        """Transport intents only; execution risk belongs to TradeExecutorShort."""
        mids = [sid for sid, _ in rows]

        for sid, pkg in rows:
            items = pkg.get("items")
            if not isinstance(items, list):
                continue

            for it in items:
                if not isinstance(it, dict):
                    continue

                tag = "(PEL) " if is_pel else ""
                micro_ok, micro_reason = self._short_micro_entry_gate(it)
                if not micro_ok:
                    print(
                        f"[IntentBridge][MICRO][REJECT] source={sid} "
                        f"symbol={_safe_str(it.get('symbol', '')).upper()} reason={micro_reason}",
                        flush=True,
                    )
                    continue
                out_id = self._publish_exec(source_stream_id=sid, item=it, reason_hint="transported")
                if out_id:
                    print(
                        f"[IntentBridge] {tag}transported -> {self.out_stream} "
                        f"source={sid} symbol={_safe_str(it.get('symbol', '')).upper()}"
                    )
                else:
                    print(
                        f"[IntentBridge] {tag}publish failed source={sid} "
                        f"symbol={_safe_str(it.get('symbol', '')).upper()}"
                    )

        self._ack(mids)
    def run_forever(self) -> None:
        if self.drain_pending:
            print("[IntentBridge] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break
                self._process_rows(rows, is_pel=True)
                time.sleep(0.05)
            print("[IntentBridge] pending drained.")

        idle = 0
        while True:
            rows = self._xreadgroup(">")
            if not rows:
                idle += 1
                if idle % 30 == 0:
                    print("[IntentBridge] idle...")
                sleep_backoff(idle)
                continue

            idle = 0
            self._process_rows(rows, is_pel=False)
            time.sleep(0.02)


if __name__ == "__main__":
    IntentBridge().run_forever()
