# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
import uuid
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
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
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
        s = str(x)
        return s if s else default
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        fx = float(x)
    except Exception:
        fx = lo
    return max(lo, min(hi, fx))


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


def _meta_get(raw: Dict[str, Any], key: str, default: Any = None) -> Any:
    try:
        meta = raw.get("meta") or {}
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
    except Exception:
        pass
    return default

def _json_load_if_str(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def _deep_dict_chain(x: Any, max_depth: int = 4) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cur = _json_load_if_str(x)

    for _ in range(max_depth):
        if not isinstance(cur, dict):
            break
        out.append(cur)
        nxt = cur.get("raw")
        nxt = _json_load_if_str(nxt)
        if not isinstance(nxt, dict):
            break
        cur = nxt

    return out


def _first_non_empty(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip() == "":
                continue
            return v
        return v
    return default
def _deep_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


@dataclass
class TradeIntent:
    intent_id: str
    ts_utc: str
    symbol: str
    interval: str
    side: str
    price: float
    score: float
    recommended_leverage: int
    recommended_notional_pct: float
    reasons: List[str]
    risk_tags: List[str]
    raw: Dict[str, Any]
    trail_pct: float = 0.0
    stall_ttl_sec: int = 0


class MasterExecutor:
    """
    IN:  top5_stream
    OUT: trade_intents_stream

    Final decision layer:
      - score gate
      - whale alignment / contra / veto
      - max open position gate
      - same-symbol dedup gate
      - leverage / notional sizing refinement
    """

    def __init__(self) -> None:
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

        self.in_stream = os.getenv(
            "MASTER_IN_STREAM",
            os.getenv("TOP5_STREAM", "top5_stream"),
        )
        self.out_stream = os.getenv(
            "MASTER_OUT_STREAM",
            os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"),
        )

        self.group = os.getenv("MASTER_GROUP", "master_exec_g")
        self.consumer = os.getenv("MASTER_CONSUMER", "master_1")
        self.group_start_id = os.getenv("MASTER_GROUP_START_ID", "$")
        self.drain_pending = _env_bool("MASTER_DRAIN_PENDING", False)

        self.read_block_ms = _env_int("MASTER_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("MASTER_BATCH_COUNT", 50)

        self.topn = _env_int("MASTER_TOPN", _env_int("BRIDGE_MAX_OPEN", 3))
        self.max_pos = _env_int(
            "MASTER_MAX_POS",
            _env_int("MAX_OPEN_POSITIONS", _env_int("MAX_OPEN_TRADES", 3)),
        )

        self.dedup_symbol_open = _env_bool("DEDUP_SYMBOL_OPEN", True)
        self.state_key = os.getenv("BRIDGE_STATE_KEY", "open_positions_state")

        self.publish_cooldown_sec = _env_int("MASTER_PUBLISH_COOLDOWN_SEC", 2)
        self._last_publish_ts = 0.0
        self._last_published_source_id: Optional[str] = None

        self.min_trade_score = _env_float(
            "MASTER_MIN_TRADE_SCORE",
            _env_float("MIN_SCORE", 0.10),
        )
        self.w_min = _env_float("MASTER_W_MIN", _env_float("W_MIN", 0.0))
        self.require_price = _env_bool("MASTER_REQUIRE_PRICE", True)

        self.trail_pct = _env_float("TRAIL_PCT", 0.05)
        self.stall_ttl_sec = _env_int("STALL_TTL_SEC", 0)

        self.w_w = _env_float("MASTER_W_SCORE_WHALE", 0.60)
        self.w_mtf = _env_float("MASTER_W_SCORE_MTF", 0.25)
        self.w_micro = _env_float("MASTER_W_SCORE_MICRO", 0.15)

        self.lev_min = _env_int("LEV_MIN", 3)
        self.lev_max = _env_int("LEV_MAX", 30)
        self.notional_min_pct = _env_float("NOTIONAL_MIN_PCT", 0.02)
        self.notional_max_pct = _env_float("NOTIONAL_MAX_PCT", 0.25)

        self.whale_boost_thr = _env_float("MASTER_WHALE_BOOST_THR", 0.20)
        self.whale_lev_boost = _env_float("MASTER_WHALE_LEV_BOOST", 1.35)
        self.whale_npct_boost = _env_float("MASTER_WHALE_NPCT_BOOST", 1.20)
        self.whale_lev_floor = _env_int("MASTER_WHALE_LEV_FLOOR", 8)
        self.whale_npct_floor = _env_float("MASTER_WHALE_NPCT_FLOOR", 0.04)

        self.high_vol_tag = os.getenv("HIGH_VOL_TAG", "high_vol")
        self.high_vol_npct_mult = _env_float("HIGH_VOL_NPCT_MULT", 0.80)
        self.high_vol_lev_mult = _env_float("HIGH_VOL_LEV_MULT", 0.85)

        self.drop_if_whale_contra = _env_bool("MASTER_DROP_WHALE_CONTRA", True)

        self.whale_use_final_decision = _env_bool("MASTER_WHALE_USE_FINAL_DECISION", True)
        self.whale_hard_block_actions = set(
            _as_list(_env_str("MASTER_WHALE_HARD_BLOCK_ACTIONS", "block_open,hard_block"))
        )
        self.whale_soft_block_actions = set(
            _as_list(_env_str("MASTER_WHALE_SOFT_BLOCK_ACTIONS", "avoid_open"))
        )
        self.whale_boost_actions = set(
            _as_list(_env_str("MASTER_WHALE_BOOST_ACTIONS", "confirm,strong_confirm,boost_open"))
        )
        self.whale_reduce_actions = set(
            _as_list(_env_str("MASTER_WHALE_REDUCE_ACTIONS", "reduce_size,tighten_risk"))
        )
        self.whale_force_exit_actions = set(
            _as_list(_env_str("MASTER_WHALE_FORCE_EXIT_ACTIONS", "force_exit"))
        )

        self.whale_soft_block_score_cap = _env_float("MASTER_WHALE_SOFT_BLOCK_SCORE_CAP", 0.55)
        self.whale_reduce_npct_mult = _env_float("MASTER_WHALE_REDUCE_NPCT_MULT", 0.70)
        self.whale_reduce_lev_mult = _env_float("MASTER_WHALE_REDUCE_LEV_MULT", 0.80)
        self.whale_confirm_score_bonus = _env_float("MASTER_WHALE_CONFIRM_SCORE_BONUS", 0.08)
        self.whale_strong_confirm_score_bonus = _env_float(
            "MASTER_WHALE_STRONG_CONFIRM_SCORE_BONUS",
            0.15,
        )

        self.heavy_enable = _env_bool("MASTER_HEAVY_ENABLE", False)
        self.heavy_topk = _env_int("MASTER_HEAVY_TOPK", 5)
        self.heavy_discard_below = _env_float("MASTER_HEAVY_DISCARD_BELOW", 0.70)
        self._warned_heavy_stub = False

        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")
        self.publish_allowed = bool(
            self.dry_run_env or (
                self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16)
            )
        )

        self._warned_missing_price: set[str] = set()
        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        print(
            f"[MasterExecutor] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"topn={self.topn} max_pos={self.max_pos} dedup_symbol_open={self.dedup_symbol_open} "
            f"min_trade_score={self.min_trade_score:.3f} w_min={self.w_min:.3f} "
            f"whale_final_decision={self.whale_use_final_decision} "
            f"trail_pct={self.trail_pct:.4f} stall_ttl_sec={self.stall_ttl_sec} "
            f"publish_allowed={self.publish_allowed} dry_run={self.dry_run_env} "
            f"state_key={self.state_key} redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(
                f"[MasterExecutor] XGROUP created: "
                f"stream={stream} group={group} start_id={start_id}"
            )
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
                    if isinstance(pkg, dict):
                        pkg.setdefault("ts_utc", _now_utc_iso())
                        pkg["_source_stream_id"] = sid
                        out.append((sid, pkg))
                    else:
                        out.append((sid, {}))
                except Exception:
                    out.append((sid, {}))
        return out

    # ---------- OPEN STATE ----------
    def _get_open_state(self) -> Dict[str, Any]:
        try:
            s = self.r.get(self.state_key)
            if not s:
                return {}
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _open_count(self, st: Dict[str, Any]) -> int:
        try:
            return len([1 for _k, v in (st or {}).items() if isinstance(v, dict)])
        except Exception:
            return 0

    def _is_symbol_open(self, st: Dict[str, Any], symbol: str) -> bool:
        try:
            return isinstance((st or {}).get(symbol), dict)
        except Exception:
            return False

    # ---------- SCORING / WHALE ----------
    def _normalize_side(self, side: Any) -> str:
        s = str(side or "").strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    def _candidate_score(self, c: Dict[str, Any]) -> float:
        return _safe_float(
            c.get("_score_total_final", c.get("_score_selected", c.get("score_total", 0.0))),
            0.0,
        )

    def _get_raw_evt(self, c: Dict[str, Any]) -> Dict[str, Any]:
        raw_evt = c.get("raw")
        raw_evt = _json_load_if_str(raw_evt)
        if isinstance(raw_evt, dict):
            return raw_evt
        return {}

    def _extract_whale_context(self, raw_evt: Dict[str, Any]) -> Dict[str, Any]:
        chain = _deep_dict_chain(raw_evt, max_depth=4)
        metas: List[Dict[str, Any]] = []

        for d in chain:
            meta = d.get("meta")
            if isinstance(meta, dict):
                metas.append(meta)

        whale_decision: Dict[str, Any] = {}
        for d in chain:
            wd = d.get("whale_final_decision")
            if isinstance(wd, dict):
                whale_decision = wd
                break
        if not whale_decision:
            for m in metas:
                wd = m.get("whale_final_decision")
                if isinstance(wd, dict):
                    whale_decision = wd
                    break

        whale_dir = _first_non_empty(
            raw_evt.get("whale_dir"),
            *[d.get("whale_dir") for d in chain],
            *[m.get("whale_dir") for m in metas],
            whale_decision.get("direction"),
            default="none",
        )

        whale_score = _first_non_empty(
            raw_evt.get("whale_score"),
            *[d.get("whale_score") for d in chain],
            *[m.get("whale_score") for m in metas],
            whale_decision.get("score"),
            whale_decision.get("confidence"),
            default=0.0,
        )

        whale_action = _first_non_empty(
            raw_evt.get("whale_action"),
            raw_evt.get("whale_decision"),
            raw_evt.get("whale_policy"),
            *[d.get("whale_action") for d in chain],
            *[d.get("whale_decision") for d in chain],
            *[m.get("whale_action") for m in metas],
            whale_decision.get("action"),
            default="",
        )

        try:
            whale_dir = str(whale_dir or "none").strip().lower()
        except Exception:
            whale_dir = "none"

        try:
            whale_score = float(whale_score or 0.0)
        except Exception:
            whale_score = 0.0

        try:
            whale_action = str(whale_action or "").strip().lower()
        except Exception:
            whale_action = ""

        return {
            "whale_dir": whale_dir,
            "whale_score": float(_clamp(whale_score, 0.0, 1.0)),
            "whale_action": whale_action,
            "whale_final_decision": whale_decision if isinstance(whale_decision, dict) else {},
            "raw_chain": chain,
            "meta_chain": metas,
        }

    def _get_whale_decision(self, raw_evt: Dict[str, Any]) -> Dict[str, Any]:
        return self._extract_whale_context(raw_evt).get("whale_final_decision", {}) or {}

    def _get_whale_action(self, raw_evt: Dict[str, Any]) -> str:
        ctx = self._extract_whale_context(raw_evt)
        return _safe_str(ctx.get("whale_action", ""), "").strip().lower()

    def _get_whale_confidence(self, raw_evt: Dict[str, Any]) -> float:
        ctx = self._extract_whale_context(raw_evt)
        return float(_clamp(_safe_float(ctx.get("whale_score", 0.0), 0.0), 0.0, 1.0))

    def _get_whale_direction(self, raw_evt: Dict[str, Any]) -> str:
        ctx = self._extract_whale_context(raw_evt)
        return _safe_str(ctx.get("whale_dir", "none"), "none").strip().lower()
    def _is_whale_contra(self, side: str, raw_evt: Dict[str, Any]) -> bool:
        whale_dir = self._get_whale_direction(raw_evt)
        whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
        whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")
        return (whale_is_buy and side == "short") or (whale_is_sell and side == "long")

    def _compute_final_score(self, c: Dict[str, Any]) -> float:
        raw_evt = self._get_raw_evt(c)
        ctx = self._extract_whale_context(raw_evt)

        whale = _safe_float(ctx.get("whale_score", c.get("whale_score", 0.0)), 0.0)

        micro = _safe_float(
            _first_non_empty(
                raw_evt.get("micro_score"),
                *[_meta_get(d, "micro_score", None) for d in ctx.get("raw_chain", []) if isinstance(d, dict)],
                c.get("micro_score", 0.0),
                0.0,
            ),
            0.0,
        )

        mtf = _safe_float(
            _first_non_empty(
                _meta_get(raw_evt, "fast_model_score", None),
                _meta_get(raw_evt, "p_used", None),
                raw_evt.get("fast_model_score"),
                raw_evt.get("p_used"),
                c.get("score_total", 0.0),
                0.0,
            ),
            0.0,
        )

        whale = _clamp(whale, 0.0, 1.0)
        micro = _clamp(micro, 0.0, 1.0)
        mtf = _clamp(mtf, 0.0, 1.0)

        score = (self.w_w * whale) + (self.w_mtf * mtf) + (self.w_micro * micro)

        if self.whale_use_final_decision:
            action = _safe_str(ctx.get("whale_action", ""), "").strip().lower()
            wconf = float(_clamp(_safe_float(ctx.get("whale_score", 0.0), 0.0), 0.0, 1.0))

            if action in self.whale_boost_actions:
                bonus = self.whale_confirm_score_bonus
                if action in ("strong_confirm", "boost_open"):
                    bonus = self.whale_strong_confirm_score_bonus
                score += float(bonus) * float(wconf)

            if action in self.whale_soft_block_actions:
                score = min(score, self.whale_soft_block_score_cap)

            if action in self.whale_hard_block_actions:
                score = 0.0

        score = _clamp(score, 0.0, 1.0)

        if self.heavy_enable and not self._warned_heavy_stub:
            print("[MasterExecutor][HEAVY] enabled but using stub heavy stage (no external model call).")
            self._warned_heavy_stub = True

        return float(score)
    def _warn_missing_price_once(self, symbol: str, interval: str, side: str) -> None:
        try:
            key = f"{symbol}|{interval}|{side}"
            if key in self._warned_missing_price:
                return
            self._warned_missing_price.add(key)
            print(f"[MasterExecutor][WARN] missing/invalid price -> dropping | {key}")
        except Exception:
            pass

    def _extract_price(self, c: Dict[str, Any], raw_evt: Dict[str, Any]) -> float:
        price = _safe_float(c.get("price", 0.0), 0.0)
        if price > 0:
            return float(price)

        ctx = self._extract_whale_context(raw_evt)

        candidates = [
            raw_evt.get("price"),
            _deep_get(raw_evt, ["raw", "price"]),
            _deep_get(raw_evt, ["raw", "raw", "price"]),
            _deep_get(raw_evt, ["raw", "raw", "raw", "price"]),
        ]

        for d in ctx.get("raw_chain", []):
            if isinstance(d, dict):
                candidates.append(d.get("price"))

        for v in candidates:
            pv = _safe_float(v, 0.0)
            if pv > 0:
                return float(pv)

        return 0.0
    def _heavy_score_one(self, c: Dict[str, Any]) -> Tuple[float, List[str]]:
        base = float(_clamp(self._candidate_score(c), 0.0, 1.0))
        return base, ["heavy_passthrough"]

    def _apply_heavy_stage(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.heavy_enable:
            return items

        k = max(0, int(self.heavy_topk))
        thr = float(self.heavy_discard_below)
        if k <= 0:
            return items

        out: List[Dict[str, Any]] = []
        for i, c in enumerate(items):
            if not isinstance(c, dict):
                continue

            c2 = dict(c)
            fast = float(_clamp(self._candidate_score(c2), 0.0, 1.0))

            if i < k:
                t0 = time.time()
                hs, hreas = self._heavy_score_one(c2)
                ms = int(round((time.time() - t0) * 1000.0))
                hs = float(_clamp(float(hs), 0.0, 1.0))
                sym = _safe_str(c2.get("symbol", "")).upper()

                print(f"[MasterExecutor][HEAVY][LAT] {sym} {ms}ms heavy={hs:.3f} fast={fast:.3f}")

                c2["_score_heavy"] = float(hs)
                final = float(max(fast, hs))
                c2["_score_total_final"] = final

                reasons = [str(x) for x in _as_list(c2.get("reason_codes"))]
                if not reasons:
                    reasons = [str(x) for x in _as_list(c2.get("reasons"))]
                c2["_reasons_final"] = reasons + list(hreas or [])

                if final < thr:
                    continue
            else:
                c2["_score_total_final"] = fast
                reasons = [str(x) for x in _as_list(c2.get("reason_codes"))]
                if not reasons:
                    reasons = [str(x) for x in _as_list(c2.get("reasons"))]
                c2["_reasons_final"] = reasons

            out.append(c2)

        out.sort(key=lambda x: float(x.get("_score_total_final", 0.0)), reverse=True)
        return out

    def _reject_by_whale_policy(
        self,
        symbol: str,
        interval: str,
        side: str,
        raw_evt: Dict[str, Any],
        score: float,
    ) -> bool:
        action = self._get_whale_action(raw_evt)
        whale_score = self._get_whale_confidence(raw_evt)
        whale_contra = self._is_whale_contra(side, raw_evt)

        if float(self.w_min) > 0.0 and float(whale_score) < float(self.w_min):
            print(
                f"[MasterExecutor][WHALE][DROP] {symbol} {interval} {side} "
                f"reason=w_min whale={whale_score:.3f} need={self.w_min:.3f}"
            )
            return True

        if self.whale_use_final_decision and action in self.whale_hard_block_actions:
            print(
                f"[MasterExecutor][WHALE][DROP] {symbol} {interval} {side} "
                f"reason=hard_block action={action} whale={whale_score:.3f}"
            )
            return True

        if self.drop_if_whale_contra and whale_contra and whale_score >= float(self.whale_boost_thr):
            print(
                f"[MasterExecutor][WHALE][DROP] {symbol} {interval} {side} "
                f"reason=contra whale_dir={self._get_whale_direction(raw_evt)} "
                f"whale={whale_score:.3f} score={score:.3f}"
            )
            return True

        return False
    def _make_intent(self, c: Dict[str, Any]) -> Optional[TradeIntent]:
        raw_evt = self._get_raw_evt(c)

        price = self._extract_price(c, raw_evt)

        score = _safe_float(c.get("_score_total_final", None), -1.0)
        if score < 0:
            score = self._compute_final_score(c)

        score = float(_clamp(score, 0.0, 1.0))

        side = self._normalize_side(_safe_str(c.get("side", raw_evt.get("side_candidate", ""))))
        symbol0 = _safe_str(c.get("symbol", "")).upper()
        interval0 = _safe_str(c.get("interval", ""), "5m")

        if self.require_price and price <= 0.0:
            self._warn_missing_price_once(symbol0, interval0, side)
            return None

        if score < float(self.min_trade_score):
            return None

        if self._reject_by_whale_policy(symbol0, interval0, side, raw_evt, score):
            return None

        reasons = [str(x) for x in _as_list(c.get("reason_codes"))]
        if not reasons:
            reasons = [str(x) for x in _as_list(c.get("reasons"))]

        risk_tags = [str(x) for x in _as_list(c.get("risk_tags"))]

        whale_score = self._get_whale_confidence(raw_evt)
        whale_action = self._get_whale_action(raw_evt)
        whale_aligned = whale_action in self.whale_boost_actions

        base_lev = int(_safe_float(c.get("recommended_leverage", 5), 5))
        base_npct = float(_safe_float(c.get("recommended_notional_pct", 0.05), 0.05))

        conf = _safe_float(raw_evt.get("confidence", c.get("confidence", 0.5)), 0.5)
        atr_pct = _safe_float(raw_evt.get("atr_pct", c.get("atr_pct", 0.01)), 0.01)
        spread_pct = _safe_float(raw_evt.get("spread_pct", c.get("spread_pct", 0.0003)), 0.0003)

        conf_adj = _clamp(0.7 + conf, 0.7, 1.7)
        atr_adj = _clamp(0.015 / max(atr_pct, 1e-6), 0.4, 1.2)
        spr_adj = _clamp(0.0004 / max(spread_pct, 1e-9), 0.4, 1.2)

        whale_mult_lev = 1.0
        whale_mult_npct = 1.0

        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            whale_mult_lev = float(self.whale_lev_boost)
            whale_mult_npct = float(self.whale_npct_boost)
            reasons.append("master_whale_boost")

        lev = int(round(base_lev * conf_adj * atr_adj * spr_adj * whale_mult_lev))
        lev = int(_clamp(lev, self.lev_min, self.lev_max))

        npct = base_npct * conf_adj * _clamp(atr_adj, 0.6, 1.1) * _clamp(spr_adj, 0.6, 1.1) * whale_mult_npct
        npct = float(_clamp(npct, self.notional_min_pct, self.notional_max_pct))

        if whale_action in self.whale_reduce_actions:
            lev = int(_clamp(lev * self.whale_reduce_lev_mult, self.lev_min, self.lev_max))
            npct = float(_clamp(npct * self.whale_reduce_npct_mult, self.notional_min_pct, self.notional_max_pct))
            reasons.append("master_whale_reduce")

        if self.high_vol_tag in (risk_tags or []):
            lev = int(_clamp(round(lev * self.high_vol_lev_mult), self.lev_min, self.lev_max))
            npct = float(_clamp(npct * self.high_vol_npct_mult, self.notional_min_pct, self.notional_max_pct))
            reasons.append("master_high_vol_clamp")

        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            lev = max(lev, int(self.whale_lev_floor))
            npct = max(npct, float(self.whale_npct_floor))

        c2 = dict(c)
        c2["_score_total_final"] = float(score)
        c2["trail_pct"] = float(self.trail_pct)
        c2["stall_ttl_sec"] = int(self.stall_ttl_sec)
        c2["w_min"] = float(self.w_min)
        c2["price"] = float(price)

        try:
            raw0 = c2.get("raw")
            raw0 = _json_load_if_str(raw0)

            if isinstance(raw0, dict):
                raw0["price"] = float(price)

                raw1 = raw0.get("raw")
                raw1 = _json_load_if_str(raw1)
                if isinstance(raw1, dict):
                    raw1["price"] = float(price)
                    raw0["raw"] = raw1

                    raw2 = raw1.get("raw")
                    raw2 = _json_load_if_str(raw2)
                    if isinstance(raw2, dict):
                        raw2["price"] = float(price)
                        raw1["raw"] = raw2

                c2["raw"] = raw0
        except Exception:
            pass
        return TradeIntent(
            intent_id=str(uuid.uuid4()),
            ts_utc=_now_utc_iso(),
            symbol=symbol0,
            interval=interval0,
            side=side,
            price=float(price),
            score=float(score),
            recommended_leverage=int(lev),
            recommended_notional_pct=float(npct),
            reasons=reasons,
            risk_tags=risk_tags,
            raw=dict(c2),
            trail_pct=float(self.trail_pct),
            stall_ttl_sec=int(self.stall_ttl_sec),
        )

    def _select_top_unique(self, items: List[Dict[str, Any]]) -> List[TradeIntent]:
        scored: List[Dict[str, Any]] = []

        for c in items:
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            if c2.get("_score_total_final") is None:
                c2["_score_total_final"] = self._compute_final_score(c2)
            scored.append(c2)

        scored.sort(key=lambda x: float(x.get("_score_total_final", 0.0)), reverse=True)

        st = self._get_open_state()
        open_count = self._open_count(st)
        slots = max(0, int(self.max_pos) - int(open_count))

        if slots <= 0:
            return []

        best_by_symbol: Dict[str, Dict[str, Any]] = {}

        for c in scored:
            sym = _safe_str(c.get("symbol", "")).upper()
            if not sym:
                continue

            if self.dedup_symbol_open and self._is_symbol_open(st, sym):
                continue

            sc = _safe_float(c.get("_score_total_final", 0.0), 0.0)
            prev = best_by_symbol.get(sym)

            if (prev is None) or (sc > _safe_float(prev.get("_score_total_final", 0.0), 0.0)):
                best_by_symbol[sym] = c

        intents: List[TradeIntent] = []

        for c in best_by_symbol.values():
            it = self._make_intent(c)
            if it:
                intents.append(it)

        intents.sort(key=lambda x: float(x.score), reverse=True)
        intents = intents[: max(0, int(self.topn))]
        intents = intents[:slots]

        return intents

    def _intent_price_for_publish(self, it: TradeIntent) -> float:
        p = float(_safe_float(it.price, 0.0))
        if p > 0:
            return p

        try:
            p2 = float(_safe_float((it.raw or {}).get("price", 0.0), 0.0))
            return p2 if p2 > 0 else 0.0
        except Exception:
            return 0.0

    def _publish_intents(self, source_stream_id: str, intents: List[TradeIntent]) -> Optional[str]:
        if not self.publish_allowed:
            return None

        if self._last_published_source_id == source_stream_id:
            return None

        now = time.time()

        if self.publish_cooldown_sec > 0 and (now - self._last_publish_ts) < self.publish_cooldown_sec:
            return None

        items_out: List[Dict[str, Any]] = []

        for it in intents:
            p = float(self._intent_price_for_publish(it))

            if self.require_price and p <= 0.0:
                self._warn_missing_price_once(it.symbol, it.interval, it.side)
                continue

            raw_copy = dict(it.raw) if isinstance(it.raw, dict) else {}

            try:
                raw_copy["price"] = float(p)
                if isinstance(raw_copy.get("raw"), dict):
                    raw_copy["raw"]["price"] = float(p)
            except Exception:
                pass

            items_out.append(
                {
                    "intent_id": it.intent_id,
                    "ts_utc": it.ts_utc,
                    "symbol": it.symbol,
                    "interval": it.interval,
                    "side": it.side,
                    "price": float(p),
                    "score": float(it.score),
                    "recommended_leverage": int(it.recommended_leverage),
                    "recommended_notional_pct": float(it.recommended_notional_pct),
                    "trail_pct": float(it.trail_pct or 0.0),
                    "stall_ttl_sec": int(it.stall_ttl_sec or 0),
                    "reasons": it.reasons,
                    "risk_tags": it.risk_tags,
                    "raw": raw_copy,
                }
            )

        if not items_out:
            return None

        payload = {
            "ts_utc": _now_utc_iso(),
            "source_top5_id": source_stream_id,
            "count": len(items_out),
            "items": items_out,
        }

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=5000,
                approximate=True,
            )
        except Exception:
            return None

        self._last_publish_ts = time.time()
        self._last_published_source_id = source_stream_id

        return sid
    def run_forever(self) -> None:
        if self.drain_pending:
            print("[MasterExecutor] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break

                mids = [sid for sid, _ in rows]

                for sid, pkg in rows:
                    items = pkg.get("items") or []
                    if not isinstance(items, list) or not items:
                        continue

                    items2 = self._apply_heavy_stage(items)
                    intents = self._select_top_unique(items2)
                    out_id = self._publish_intents(source_stream_id=sid, intents=intents) if intents else None

                    if out_id:
                        summary = ", ".join(
                            [
                                f"{it.symbol}:{it.side}@L{it.recommended_leverage} "
                                f"npct={it.recommended_notional_pct:.3f}"
                                for it in intents
                            ]
                        )
                        print(
                            f"[MasterExecutor] (PEL) published intents={len(intents)} "
                            f"id={out_id} | {summary}"
                        )

                self._ack(mids)
                time.sleep(0.05)

            print("[MasterExecutor] pending drained.")

        idle = 0
        while True:
            rows = self._xreadgroup(">")
            if not rows:
                idle += 1
                if idle % 30 == 0:
                    print("[MasterExecutor] idle...")
                continue

            idle = 0
            mids = [sid for sid, _ in rows]

            for sid, pkg in rows:
                items = pkg.get("items") or []
                if not isinstance(items, list) or not items:
                    continue

                items2 = self._apply_heavy_stage(items)
                intents = self._select_top_unique(items2)
                out_id = self._publish_intents(source_stream_id=sid, intents=intents) if intents else None

                if out_id:
                    summary = ", ".join(
                        [
                            f"{it.symbol}:{it.side}@L{it.recommended_leverage} "
                            f"npct={it.recommended_notional_pct:.3f}"
                            for it in intents
                        ]
                    )
                    print(
                        f"[MasterExecutor] published intents={len(intents)} "
                        f"-> {self.out_stream} id={out_id} | {summary}"
                    )

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    MasterExecutor().run_forever()
