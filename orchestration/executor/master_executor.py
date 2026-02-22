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


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        s = str(x)
    except Exception:
        return default
    return s


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
    """
    Worker -> Aggregator raw evt'de bazı alanlar meta altında.
    raw.meta.whale_dir, raw.meta.p_used, raw.meta.fast_model_score gibi.
    """
    try:
        meta = raw.get("meta") or {}
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
    except Exception:
        pass
    return default


@dataclass
class TradeIntent:
    intent_id: str
    ts_utc: str
    symbol: str
    interval: str
    side: str
    score: float
    recommended_leverage: int
    recommended_notional_pct: float
    reasons: List[str]
    risk_tags: List[str]
    raw: Dict[str, Any]

    # downstream (executor/position manager) can use these
    trail_pct: float = 0.0
    stall_ttl_sec: int = 0


class MasterExecutor:
    """
    IN:  top5_stream  (TopSelector publishes {"ts_utc","topk","items":[CandidateTrade,...]})
    OUT: trade_intents_stream

    Notes:
      - DRY_RUN=1 -> intents ARE published (safe; execution is controlled downstream).
      - DRY_RUN=0 -> requires ARMED=1, LIVE_KILL_SWITCH=0, ARM_TOKEN>=16.
    """

    def __init__(self) -> None:
        # Redis
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_db = _env_int("REDIS_DB", 0)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None

        # Streams
        self.in_stream = os.getenv("MASTER_IN_STREAM", os.getenv("TOP5_STREAM", "top5_stream"))
        self.out_stream = os.getenv("MASTER_OUT_STREAM", os.getenv("TRADE_INTENTS_STREAM", "trade_intents_stream"))

        # Consumer group
        self.group = os.getenv("MASTER_GROUP", "master_exec_g")
        self.consumer = os.getenv("MASTER_CONSUMER", "master_1")
        self.group_start_id = os.getenv("MASTER_GROUP_START_ID", "$")
        self.drain_pending = _env_bool("MASTER_DRAIN_PENDING", False)

        # Read tuning
        self.read_block_ms = _env_int("MASTER_READ_BLOCK_MS", 2000)
        self.batch_count = _env_int("MASTER_BATCH_COUNT", 50)

        # Selection / limits
        self.topn = _env_int("MASTER_TOPN", 3)
        self.max_pos = _env_int("MASTER_MAX_POS", 3)

        # Publish cooldown (prevents spam)
        self.publish_cooldown_sec = _env_int("MASTER_PUBLISH_COOLDOWN_SEC", 2)
        self._last_publish_ts = 0.0
        self._last_published_source_id: Optional[str] = None

        # Global score gate ("0 trade normal")
        self.min_trade_score = _env_float(
            "MASTER_MIN_TRADE_SCORE",
            _env_float("MIN_SCORE", 0.10),
        )

        # Whale min gate (second stage safety)
        # If set (>0), intents require whale_score >= threshold
        self.w_min = _env_float("MASTER_W_MIN", _env_float("W_MIN", 0.0))

        # trailing / stall knobs forwarded to downstream executor
        self.trail_pct = _env_float("TRAIL_PCT", 0.05)      # default 5%
        self.stall_ttl_sec = _env_int("STALL_TTL_SEC", 0)   # e.g. 7200 for 2h

        # Whale-first scoring controls (final score)
        self.w_w = _env_float("MASTER_W_SCORE_WHALE", 0.60)
        self.w_mtf = _env_float("MASTER_W_SCORE_MTF", 0.25)
        self.w_micro = _env_float("MASTER_W_SCORE_MICRO", 0.15)

        # Heavy-on-top5 toggles (optional stage)
        self.heavy_enable = _env_bool("MASTER_HEAVY_ENABLE", False)
        self.heavy_topk = _env_int("MASTER_HEAVY_TOPK", 5)
        self.heavy_discard_below = _env_float("MASTER_HEAVY_DISCARD_BELOW", 0.70)

        if self.heavy_enable:
            self.min_trade_score = max(float(self.min_trade_score), float(self.heavy_discard_below))

        # Risk sizing clamp
        self.lev_min = _env_int("LEV_MIN", 3)
        self.lev_max = _env_int("LEV_MAX", 30)
        self.notional_min_pct = _env_float("NOTIONAL_MIN_PCT", 0.02)
        self.notional_max_pct = _env_float("NOTIONAL_MAX_PCT", 0.25)

        # Whale aggression controls
        self.whale_boost_thr = _env_float("MASTER_WHALE_BOOST_THR", 0.20)
        self.whale_lev_boost = _env_float("MASTER_WHALE_LEV_BOOST", 1.35)
        self.whale_npct_boost = _env_float("MASTER_WHALE_NPCT_BOOST", 1.20)
        self.whale_lev_floor = _env_int("MASTER_WHALE_LEV_FLOOR", 8)
        self.whale_npct_floor = _env_float("MASTER_WHALE_NPCT_FLOOR", 0.04)

        # High-volatility risk clamp
        self.high_vol_tag = os.getenv("HIGH_VOL_TAG", "high_vol")
        self.high_vol_npct_mult = _env_float("HIGH_VOL_NPCT_MULT", 0.80)
        self.high_vol_lev_mult = _env_float("HIGH_VOL_LEV_MULT", 0.85)

        # Contra whale safety
        self.drop_if_whale_contra = _env_bool("MASTER_DROP_WHALE_CONTRA", True)

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        # --- LIVE SAFETY POLICY ---
        self.dry_run_env = _env_bool("DRY_RUN", True)
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")

        self.publish_allowed = bool(
            self.dry_run_env or (self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16))
        )

        if (not self.dry_run_env) and (not self.publish_allowed):
            print(
                f"[MasterExecutor][SAFE] DRY_RUN=0 but publish blocked: "
                f"ARMED={self.armed} KILL={self.kill_switch} ARM_TOKEN_len={len(self.arm_token)}"
            )

        self._warned_heavy_stub = False

        print(
            f"[MasterExecutor] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"topn={self.topn} max_pos={self.max_pos} min_trade_score={self.min_trade_score:.3f} "
            f"w_min={self.w_min:.3f} trail_pct={self.trail_pct:.4f} stall_ttl_sec={self.stall_ttl_sec} "
            f"heavy_enable={self.heavy_enable} heavy_topk={self.heavy_topk} "
            f"publish_allowed={self.publish_allowed} dry_run={self.dry_run_env} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[MasterExecutor] XGROUP created: stream={stream} group={group} start_id={start_id}")
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
            pkg = json.loads(s)
            if isinstance(pkg, dict):
                pkg.setdefault("ts_utc", _now_utc_iso())
                pkg["_source_stream_id"] = stream_id
                return pkg
        except Exception:
            return None
        return None

    def _normalize_side(self, side: str) -> str:
        s = (side or "").strip().lower()
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

    def _heavy_score_one(self, c: Dict[str, Any]) -> Tuple[float, List[str]]:
        base = float(self._candidate_score(c))
        return base, ["heavy_passthrough"]

    def _is_whale_contra(self, side: str, raw_evt: Dict[str, Any]) -> bool:
        whale_dir = _safe_str(raw_evt.get("whale_dir", _meta_get(raw_evt, "whale_dir", "none"))).lower()
        whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
        whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")
        if whale_is_buy and side == "short":
            return True
        if whale_is_sell and side == "long":
            return True
        return False

    def _compute_final_score(self, c: Dict[str, Any]) -> float:
        raw_evt = c.get("raw") if isinstance(c.get("raw"), dict) else {}
        if raw_evt is None:
            raw_evt = {}

        whale = _safe_float(raw_evt.get("whale_score", c.get("whale_score", 0.0)), 0.0)

        micro = _safe_float(
            _meta_get(raw_evt, "micro_score", raw_evt.get("micro_score", c.get("micro_score", 0.0))),
            0.0,
        )

        mtf = _safe_float(
            _meta_get(
                raw_evt,
                "fast_model_score",
                _meta_get(raw_evt, "p_used", c.get("score_total", 0.0)),
            ),
            0.0,
        )

        whale = _clamp(whale, 0.0, 1.0)
        micro = _clamp(micro, 0.0, 1.0)
        mtf = _clamp(mtf, 0.0, 1.0)

        score = (self.w_w * whale) + (self.w_mtf * mtf) + (self.w_micro * micro)
        score = _clamp(score, 0.0, 1.0)

        if self.heavy_enable and not self._warned_heavy_stub:
            print("[MasterExecutor][HEAVY] enabled but using stub final-score (no real heavy model call yet).")
            self._warned_heavy_stub = True

        return float(score)
    def _make_intent(self, c: Dict[str, Any]) -> Optional[TradeIntent]:
        raw_evt = c.get("raw") if isinstance(c.get("raw"), dict) else {}
        if raw_evt is None:
            raw_evt = {}

        # --- PRICE extraction (critical) ---
        # CandidateTrade top-level price OR raw.price OR raw.raw.price
        price = _safe_float(c.get("price", 0.0), 0.0)
        if price <= 0:
            price = _safe_float(raw_evt.get("price", 0.0), 0.0)
        if price <= 0 and isinstance(raw_evt.get("raw"), dict):
            price = _safe_float(raw_evt["raw"].get("price", 0.0), 0.0)
        if price < 0:
            price = 0.0

        # score: prefer precomputed, otherwise compute
        if c.get("_score_total_final") is None:
            score = self._compute_final_score(c)
        else:
            score = _safe_float(c.get("_score_total_final", 0.0), 0.0)

        score = float(_clamp(float(score), 0.0, 1.0))
        if score < float(self.min_trade_score):
            return None

        base_lev = int(_safe_float(c.get("recommended_leverage", 5), 5))
        base_npct = float(_safe_float(c.get("recommended_notional_pct", 0.05), 0.05))

        conf = _safe_float(raw_evt.get("confidence", c.get("confidence", 0.5)), 0.5)
        atr_pct = _safe_float(raw_evt.get("atr_pct", c.get("atr_pct", 0.01)), 0.01)
        spread_pct = _safe_float(raw_evt.get("spread_pct", c.get("spread_pct", 0.0003)), 0.0003)

        conf_adj = _clamp(0.7 + conf, 0.7, 1.7)
        atr_adj = _clamp(0.015 / max(atr_pct, 1e-6), 0.4, 1.2)
        spr_adj = _clamp(0.0004 / max(spread_pct, 1e-9), 0.4, 1.2)

        side = self._normalize_side(_safe_str(c.get("side", raw_evt.get("side_candidate", ""))))

        reasons = [str(x) for x in _as_list(c.get("reason_codes"))] or [str(x) for x in _as_list(c.get("reasons"))]
        risk_tags = [str(x) for x in _as_list(c.get("risk_tags"))]

        whale_score = _safe_float(raw_evt.get("whale_score", c.get("whale_score", 0.0)), 0.0)
        whale_dir = _safe_str(raw_evt.get("whale_dir", _meta_get(raw_evt, "whale_dir", "none")))
        whale_contra = self._is_whale_contra(side, {**raw_evt, "whale_dir": whale_dir})

        # Whale minimum gate (second stage)
        if float(self.w_min) > 0.0 and float(whale_score) < float(self.w_min):
            return None

        whale_aligned = ("whale_align_long" in reasons) or ("whale_align_short" in reasons)

        if self.drop_if_whale_contra and whale_contra and whale_score >= float(self.whale_boost_thr):
            return None

        whale_mult_lev = 1.0
        whale_mult_npct = 1.0
        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            whale_mult_lev = float(self.whale_lev_boost)
            whale_mult_npct = float(self.whale_npct_boost)
            reasons = reasons + ["master_whale_boost"]

        lev = int(round(float(base_lev) * conf_adj * atr_adj * spr_adj * whale_mult_lev))
        lev = int(_clamp(float(lev), float(self.lev_min), float(self.lev_max)))

        npct = (
            float(base_npct)
            * conf_adj
            * _clamp(atr_adj, 0.6, 1.1)
            * _clamp(spr_adj, 0.6, 1.1)
            * whale_mult_npct
        )
        npct = float(_clamp(float(npct), float(self.notional_min_pct), float(self.notional_max_pct)))

        if self.high_vol_tag in (risk_tags or []):
            lev = int(
                _clamp(
                    float(int(round(float(lev) * float(self.high_vol_lev_mult)))),
                    float(self.lev_min),
                    float(self.lev_max),
                )
            )
            npct = float(
                _clamp(
                    float(npct) * float(self.high_vol_npct_mult),
                    float(self.notional_min_pct),
                    float(self.notional_max_pct),
                )
            )
            reasons = reasons + ["master_high_vol_clamp"]

        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            lev = max(lev, int(self.whale_lev_floor))
            npct = max(npct, float(self.whale_npct_floor))

        c2 = dict(c)
        c2["_score_total_final"] = float(score)

        # add execution-management knobs to raw (downstream can use)
        c2["trail_pct"] = float(self.trail_pct)
        c2["stall_ttl_sec"] = int(self.stall_ttl_sec)
        c2["w_min"] = float(self.w_min)

        # --- ensure price propagated into raw ---
        c2["price"] = float(price)
        try:
            if isinstance(c2.get("raw"), dict):
                c2["raw"]["price"] = float(price)
        except Exception:
            pass

        return TradeIntent(
            intent_id=str(uuid.uuid4()),
            ts_utc=_now_utc_iso(),
            symbol=_safe_str(c2.get("symbol", "")).upper(),
            interval=_safe_str(c2.get("interval", "")),
            side=side,
            score=float(score),
            recommended_leverage=int(lev),
            recommended_notional_pct=float(npct),
            reasons=reasons,
            risk_tags=risk_tags,
            raw=dict(c2),
            trail_pct=float(self.trail_pct),
            stall_ttl_sec=int(self.stall_ttl_sec),
        )

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

                reasons = [str(x) for x in _as_list(c2.get("reason_codes"))] or [str(x) for x in _as_list(c2.get("reasons"))]
                c2["_reasons_final"] = reasons + list(hreas or [])

                if final < thr:
                    continue
            else:
                c2["_score_total_final"] = fast
                reasons = [str(x) for x in _as_list(c2.get("reason_codes"))] or [str(x) for x in _as_list(c2.get("reasons"))]
                c2["_reasons_final"] = reasons

            out.append(c2)

        out.sort(key=lambda x: float(x.get("_score_total_final", 0.0)), reverse=True)
        return out

    def _select_top_unique(self, items: List[Dict[str, Any]]) -> List[TradeIntent]:
        t0 = time.time()

        scored: List[Dict[str, Any]] = []
        for c in items:
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            if c2.get("_score_total_final") is None:
                c2["_score_total_final"] = self._compute_final_score(c2)
            scored.append(c2)

        scored.sort(key=lambda x: float(x.get("_score_total_final", 0.0)), reverse=True)

        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for c in scored:
            sym = _safe_str(c.get("symbol", "")).upper()
            if not sym:
                continue
            sc = _safe_float(c.get("_score_total_final", 0.0), 0.0)
            prev = best_by_symbol.get(sym)
            if (prev is None) or (sc > _safe_float(prev.get("_score_total_final", 0.0), 0.0)):
                best_by_symbol[sym] = c

        intents: List[TradeIntent] = []
        for c in best_by_symbol.values():
            it = self._make_intent(c)
            if it is not None:
                intents.append(it)

        intents.sort(key=lambda x: float(x.score), reverse=True)
        intents = intents[: max(0, int(self.topn))]

        dt_ms = int(round((time.time() - t0) * 1000.0))
        if intents:
            print(f"[LAT][master] select+intent dt={dt_ms}ms in_items={len(items)} out_intents={len(intents)}")

        return intents

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
    def _publish_intents(self, source_stream_id: str, intents: List[TradeIntent]) -> Optional[str]:
        if not self.publish_allowed:
            return None

        if self._last_published_source_id == source_stream_id:
            return None

        now = time.time()
        if self.publish_cooldown_sec > 0 and (now - self._last_publish_ts) < self.publish_cooldown_sec:
            return None

        payload = {
            "ts_utc": _now_utc_iso(),
            "source_top5_id": source_stream_id,
            "count": len(intents),
            "items": [
                {
                    "intent_id": it.intent_id,
                    "ts_utc": it.ts_utc,
                    "symbol": it.symbol,
                    "interval": it.interval,
                    "side": it.side,
                    "price": float(_safe_float(it.raw.get("price", 0.0), 0.0)),  # <<< CRITICAL
                    "score": it.score,
                    "recommended_leverage": it.recommended_leverage,
                    "recommended_notional_pct": it.recommended_notional_pct,
                    "trail_pct": float(getattr(it, "trail_pct", 0.0) or 0.0),
                    "stall_ttl_sec": int(getattr(it, "stall_ttl_sec", 0) or 0),
                    "reasons": it.reasons,
                    "risk_tags": it.risk_tags,
                    "raw": it.raw,
                }
                for it in intents
            ],
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
                    if not intents:
                        continue

                    out_id = self._publish_intents(source_stream_id=sid, intents=intents)
                    if out_id:
                        summary = ", ".join(
                            [
                                f"{it.symbol}:{it.side}@L{it.recommended_leverage} npct={it.recommended_notional_pct:.3f}"
                                for it in intents
                            ]
                        )
                        print(f"[MasterExecutor] (PEL) published intents={len(intents)} id={out_id} | {summary}")

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
                if not intents:
                    continue

                out_id = self._publish_intents(source_stream_id=sid, intents=intents)
                if out_id:
                    summary = ", ".join(
                        [
                            f"{it.symbol}:{it.side}@L{it.recommended_leverage} npct={it.recommended_notional_pct:.3f}"
                            for it in intents
                        ]
                    )
                    print(f"[MasterExecutor] published intents={len(intents)} -> {self.out_stream} id={out_id} | {summary}")

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    MasterExecutor().run_forever()
