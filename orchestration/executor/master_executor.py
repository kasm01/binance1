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

from orchestration.context.market_context_contract import classify_confirmed_position_build


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


@dataclass
class TradeIntent:
    intent_id: str
    ts_utc: str
    symbol: str
    interval: str
    side: str
    price: float
    score: float              # canonical MODEL score
    ranking_score: float      # context/ranking score; never a model hard gate
    reasons: List[str]
    risk_tags: List[str]
    raw: Dict[str, Any]


class MasterExecutor:
    """
    IN:  top5_stream
    OUT: trade_intents_stream

    Final decision layer:
      - score gate
      - whale alignment / contra / veto
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

        self.topn = _env_int("MASTER_TOPN", 3)

        self._last_published_source_id: Optional[str] = None

        self.min_trade_score = _env_float("MASTER_MIN_TRADE_SCORE", 0.501)
        # SHORT soft-score band is a provisional MODEL SCORE state, not a pass.
        # When disabled, production behavior is identical to the hard threshold.
        self.short_soft_score_enable = _env_bool("MASTER_SHORT_SOFT_SCORE_ENABLE", False)
        self.short_soft_score_min = _clamp(
            _env_float("MASTER_SHORT_SOFT_SCORE_MIN", 0.530),
            0.0,
            float(self.min_trade_score),
        )
        self.w_min = _env_float("MASTER_W_MIN", 0.0)
        self.short_money_flow_enable = _env_bool("MASTER_SHORT_MONEY_FLOW_ENABLE", True)
        self.short_money_flow_exhaustion_60m = _env_float("MASTER_SHORT_MONEY_FLOW_EXHAUSTION_60M", -0.20)
        # MARKET CONTEXT owns only OI regime classification. Risk sizing policy
        # belongs to the central Risk Master contract. Disabled by default.
        self.oi_context_enable = _env_bool(
            "MASTER_OI_CONTEXT_ENABLE",
            False,
        )
        self.oi_context_max_age_sec = max(60, _env_int("MASTER_OI_CONTEXT_MAX_AGE_SEC", 900))

        self.whale_boost_thr = _env_float("MASTER_WHALE_BOOST_THR", 0.20)
        self.drop_if_whale_contra = _env_bool("MASTER_DROP_WHALE_CONTRA", True)

        self.whale_use_final_decision = _env_bool("MASTER_WHALE_USE_FINAL_DECISION", True)
        self.whale_hard_block_actions = set(
            _as_list(_env_str("MASTER_WHALE_HARD_BLOCK_ACTIONS", "block_open,hard_block"))
        )

        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")
        self.publish_allowed = bool(
            self.dry_run_env or (
                self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16)
            )
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        print(
            f"[MasterExecutor] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"topn={self.topn} "
            f"min_trade_score={self.min_trade_score:.3f} w_min={self.w_min:.3f} "
            f"whale_final_decision={self.whale_use_final_decision} "
            f"publish_allowed={self.publish_allowed} dry_run={self.dry_run_env} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
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



    # ---------- SCORING / WHALE ----------
    def _normalize_side(self, side: Any) -> str:
        s = str(side or "").strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    def _load_oi_context(self, symbol: str) -> Dict[str, Any]:
        """Read the centrally collected OI snapshot. Missing/stale data is neutral."""
        sym = str(symbol or "").upper().strip()
        out: Dict[str, Any] = {
            "available": False, "fresh": False, "source": "",
            "source_ts_ms": 0, "age_sec": None,
            "sum_open_interest": 0.0, "oi_delta_5m": 0.0, "oi_delta_15m": 0.0,
            "previous_source_ts_ms": 0, "previous_sum_open_interest": 0.0,
            "previous_oi_delta_5m": 0.0, "previous_oi_delta_15m": 0.0,
        }
        if not sym:
            return out
        try:
            raw = self.r.get(f"context:oi:{sym}")
            if not raw:
                return out
            obj = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(obj, dict):
                return out
            source_ts_ms = int(_safe_float(obj.get("source_ts_ms", 0), 0.0))
            age_sec = max(0.0, (time.time() * 1000.0 - float(source_ts_ms)) / 1000.0) if source_ts_ms > 0 else None
            fresh = bool(age_sec is not None and age_sec <= float(self.oi_context_max_age_sec))
            out.update({
                "available": True, "fresh": fresh, "source": str(obj.get("source") or ""),
                "source_ts_ms": source_ts_ms, "age_sec": age_sec,
                "sum_open_interest": _safe_float(obj.get("sum_open_interest", 0.0), 0.0),
                "oi_delta_5m": _safe_float(obj.get("oi_delta_5m", 0.0), 0.0),
                "oi_delta_15m": _safe_float(obj.get("oi_delta_15m", 0.0), 0.0),
                "previous_source_ts_ms": int(_safe_float(obj.get("previous_source_ts_ms", 0), 0.0)),
                "previous_sum_open_interest": _safe_float(obj.get("previous_sum_open_interest", 0.0), 0.0),
                "previous_oi_delta_5m": _safe_float(obj.get("previous_oi_delta_5m", 0.0), 0.0),
                "previous_oi_delta_15m": _safe_float(obj.get("previous_oi_delta_15m", 0.0), 0.0),
            })
        except Exception:
            return out
        return out

    def _evaluate_market_context_stage(
        self, *, candidate: Dict[str, Any], symbol: str, interval: str, side: str, ranking_score: float,
    ) -> Dict[str, Any]:
        """Single MARKET CONTEXT authority with canonical market_context.v1 output."""
        whale_score = float(_clamp(_safe_float(candidate.get("whale_score", 0.0), 0.0), 0.0, 1.0))
        whale_dir = _safe_str(candidate.get("whale_dir", "none"), "none").strip().lower()
        whale_action = _safe_str(candidate.get("whale_action", ""), "").strip().lower()
        spread_pct = max(0.0, _safe_float(candidate.get("spread_pct", 0.0), 0.0))
        atr_pct = max(0.0, _safe_float(candidate.get("atr_pct", 0.0), 0.0))
        liq_score = float(_clamp(_safe_float(candidate.get("liq_score", 0.0), 0.0), 0.0, 1.0))
        money_flow_15m = max(-1.0, min(1.0, _safe_float(candidate.get("money_flow_15m", 0.0), 0.0)))
        money_flow_60m = max(-1.0, min(1.0, _safe_float(candidate.get("money_flow_60m", 0.0), 0.0)))
        price_return_15m = _safe_float(candidate.get("price_return_15m", 0.0), 0.0)
        oi = self._load_oi_context(symbol)

        whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
        whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")
        whale_contra = (whale_is_buy and side == "short") or (whale_is_sell and side == "long")
        money_flow_exhaustion = bool(side == "short" and self.short_money_flow_enable and money_flow_60m <= float(self.short_money_flow_exhaustion_60m))
        oi_regime = classify_confirmed_position_build(
            oi_delta_15m=(oi.get("oi_delta_15m", 0.0) if self.oi_context_enable and oi.get("fresh", False) else 0.0),
            price_return_15m=price_return_15m,
            money_flow_15m=money_flow_15m,
        )
        confirmed_build = bool(oi_regime["confirmed_build"])
        build_direction = str(oi_regime["build_direction"])
        context = {
            "schema_version": "market_context.v1", "symbol": str(symbol or "").upper(),
            "interval": str(interval or ""), "side": str(side or "").lower(),
            "whale_dir": whale_dir, "whale_score": whale_score, "whale_action": whale_action,
            "spread_pct": spread_pct, "atr_pct": atr_pct, "liq_score": liq_score,
            "money_flow_15m": float(money_flow_15m), "money_flow_60m": float(money_flow_60m),
            "price_return_15m": float(price_return_15m),
            "money_flow_state": "short_exhaustion" if money_flow_exhaustion else "normal",
            "oi_state": "confirmed_build" if confirmed_build else "normal",
            "price_response": {"return_15m": float(price_return_15m), "direction": "up" if price_return_15m > 0 else "down" if price_return_15m < 0 else "flat"},
            "money_flow": {"flow_15m": float(money_flow_15m), "flow_60m": float(money_flow_60m), "state": "short_exhaustion" if money_flow_exhaustion else "normal"},
            "open_interest": dict(oi),
            "regime": {"name": "confirmed_build" if confirmed_build else "normal", "build_direction": build_direction},
        }
        if money_flow_exhaustion:
            return {"state": "REJECT", "reason": "short_money_flow_exhaustion", "context": context}
        if float(self.w_min) > 0.0 and whale_dir != "none" and whale_score < float(self.w_min):
            return {"state": "REJECT", "reason": "whale_min", "context": context}
        if self.whale_use_final_decision and whale_action in self.whale_hard_block_actions:
            return {"state": "REJECT", "reason": "whale_hard_block", "context": context}
        if self.drop_if_whale_contra and whale_contra and whale_score >= float(self.whale_boost_thr):
            return {"state": "REJECT", "reason": "whale_contra", "context": context}
        return {"state": "READY", "reason": "market_context_pass", "context": context, "ranking_score": float(ranking_score), "symbol": symbol, "interval": interval, "side": side}

    def _evaluate_model_score_gate_stage(self, *, model_score: float, side: str = "") -> Dict[str, Any]:
        """Single MODEL SCORE GATE authority.

        SHORT may expose a research-validated provisional band as PENDING_CONTEXT.
        PENDING_CONTEXT is never equivalent to a hard pass; downstream must confirm
        the exact entry-time market context before an order can be eligible.
        """
        score = float(_clamp(model_score, 0.0, 1.0))
        threshold = float(self.min_trade_score)
        side_norm = self._normalize_side(side)

        if score >= threshold:
            state = "READY"
            reason = "model_score_pass"
        elif (
            side_norm == "short"
            and bool(self.short_soft_score_enable)
            and score >= float(self.short_soft_score_min)
        ):
            state = "PENDING_CONTEXT"
            reason = "short_soft_score_pending_context"
        else:
            state = "REJECT"
            reason = "model_score_low"

        return {
            "state": state,
            "reason": reason,
            "model_score": score,
            "threshold": threshold,
            "soft_threshold": float(self.short_soft_score_min),
        }

    def _make_intent(self, c: Dict[str, Any]) -> Optional[TradeIntent]:
        price = max(0.0, _safe_float(c.get("price", 0.0), 0.0))
        ranking_score = float(_clamp(_safe_float(c.get("ranking_score", c.get("model_score", 0.0)), 0.0), 0.0, 1.0))
        model_score = float(_clamp(_safe_float(c.get("model_score", 0.0), 0.0), 0.0, 1.0))

        side = self._normalize_side(_safe_str(c.get("side", "")))
        symbol0 = _safe_str(c.get("symbol", "")).upper()
        interval0 = _safe_str(c.get("interval", ""), "5m")

        market_context = self._evaluate_market_context_stage(
            candidate=c, symbol=symbol0, interval=interval0, side=side, ranking_score=ranking_score,
        )
        if market_context["state"] != "READY":
            return None

        score_gate = self._evaluate_model_score_gate_stage(model_score=model_score, side=side)
        if score_gate["state"] not in ("READY", "PENDING_CONTEXT"):
            return None

        reasons = [str(x) for x in _as_list(c.get("reason_codes"))]
        if not reasons:
            reasons = [str(x) for x in _as_list(c.get("reasons"))]

        risk_tags = [str(x) for x in _as_list(c.get("risk_tags"))]

        c2 = dict(c)
        c2["ranking_score"] = float(ranking_score)
        c2["model_score"] = float(model_score)
        c2["market_context"] = dict(market_context.get("context") or {})
        c2["model_score_gate_state"] = str(score_gate.get("state") or "REJECT")
        c2["model_score_gate_passed"] = bool(score_gate.get("state") == "READY")
        c2["model_score_gate_threshold"] = float(score_gate.get("threshold") or self.min_trade_score)
        c2["model_score_gate_soft_threshold"] = float(score_gate.get("soft_threshold") or self.short_soft_score_min)
        c2["soft_context_required"] = bool(score_gate.get("state") == "PENDING_CONTEXT")
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
            score=float(model_score),
            ranking_score=float(ranking_score),
            reasons=reasons,
            risk_tags=risk_tags,
            raw=dict(c2),
        )

    def _select_top_unique(self, items: List[Dict[str, Any]]) -> List[TradeIntent]:
        scored: List[Dict[str, Any]] = []

        for c in items:
            if not isinstance(c, dict):
                continue
            scored.append(dict(c))

        scored.sort(key=lambda x: float(x.get("ranking_score", x.get("model_score", 0.0)) or 0.0), reverse=True)

        best_by_symbol: Dict[str, Dict[str, Any]] = {}

        for c in scored:
            sym = _safe_str(c.get("symbol", "")).upper()
            if not sym:
                continue

            sc = _safe_float(c.get("ranking_score", c.get("model_score", 0.0)), 0.0)
            prev = best_by_symbol.get(sym)

            if (prev is None) or (sc > _safe_float(prev.get("ranking_score", prev.get("model_score", 0.0)), 0.0)):
                best_by_symbol[sym] = c

        intents: List[TradeIntent] = []

        for c in best_by_symbol.values():
            it = self._make_intent(c)
            if it:
                intents.append(it)

        intents.sort(key=lambda x: float(x.ranking_score), reverse=True)
        intents = intents[: max(0, int(self.topn))]

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

        items_out: List[Dict[str, Any]] = []

        for it in intents:
            p = float(self._intent_price_for_publish(it))

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
                    "model_score": float(it.score),
                    "ranking_score": float(it.ranking_score),
                    "model_provider": str(raw_copy.get("model_provider") or ((raw_copy.get("raw") or {}).get("model_provider") if isinstance(raw_copy.get("raw"), dict) else "") or "hybrid_coin_specific"),
                    "market_context_passed": True,
                    "market_context": dict(raw_copy.get("market_context") or {}),
                    "model_score_gate_state": str(raw_copy.get("model_score_gate_state") or "READY"),
                    "model_score_gate_passed": bool(raw_copy.get("model_score_gate_passed", True)),
                    "model_score_gate_threshold": float(raw_copy.get("model_score_gate_threshold") or self.min_trade_score),
                    "model_score_gate_soft_threshold": float(raw_copy.get("model_score_gate_soft_threshold") or self.short_soft_score_min),
                    "soft_context_required": bool(raw_copy.get("soft_context_required", False)),
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

                    intents = self._select_top_unique(items)
                    out_id = self._publish_intents(source_stream_id=sid, intents=intents) if intents else None

                    if out_id:
                        summary = ", ".join(
                            [
                                f"{it.symbol}:{it.side} model={it.score:.3f} rank={it.ranking_score:.3f}"
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

                intents = self._select_top_unique(items)
                out_id = self._publish_intents(source_stream_id=sid, intents=intents) if intents else None

                if out_id:
                    summary = ", ".join(
                        [
                            f"{it.symbol}:{it.side} model={it.score:.3f} rank={it.ranking_score:.3f}"
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
