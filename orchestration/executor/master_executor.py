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
    if t in ("1","true","yes","y","on"):
        return True
    if t in ("0","false","no","n","off",""):
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
        # "a,b,c" gibi gelirse ayır
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return []


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


class MasterExecutor:
    """
    Reads TOP5_STREAM packages, selects topN unique symbols, publishes trade intents.

    IN:  top5_stream (group-based consumption)
    OUT: trade_intents_stream
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
        self.drain_pending = os.getenv("MASTER_DRAIN_PENDING", "0").strip().lower() in ("1", "true", "yes", "on")

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

        # Global score gate (CRITICAL)
        self.min_trade_score = _env_float("MASTER_MIN_TRADE_SCORE", 0.10)

        # Risk sizing clamp
        self.lev_min = _env_int("LEV_MIN", 3)
        self.lev_max = _env_int("LEV_MAX", 30)
        self.notional_min_pct = _env_float("NOTIONAL_MIN_PCT", 0.02)
        self.notional_max_pct = _env_float("NOTIONAL_MAX_PCT", 0.25)

        # Whale aggression controls
        self.whale_boost_thr = _env_float("MASTER_WHALE_BOOST_THR", 0.20)
        self.whale_lev_boost = _env_float("MASTER_WHALE_LEV_BOOST", 1.35)      # multiplier
        self.whale_npct_boost = _env_float("MASTER_WHALE_NPCT_BOOST", 1.20)    # multiplier
        # Whale aggressive floor
        self.whale_lev_floor = _env_int("MASTER_WHALE_LEV_FLOOR", 8)           # whale align -> lev at least this
        self.whale_npct_floor = _env_float("MASTER_WHALE_NPCT_FLOOR", 0.04)    # whale align -> npct at least this

        # High-volatility risk clamp (defaults)
        self.high_vol_tag = os.getenv("HIGH_VOL_TAG", "high_vol")
        self.high_vol_npct_mult = _env_float("HIGH_VOL_NPCT_MULT", 0.80)
        self.high_vol_lev_mult = _env_float("HIGH_VOL_LEV_MULT", 0.85)

        # Contra whale safety
        self.drop_if_whale_contra = os.getenv("MASTER_DROP_WHALE_CONTRA", "1").strip().lower() in ("1", "true", "yes", "on")

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

            print(
            f"[MasterExecutor] started. in={self.in_stream} out={self.out_stream} "

                f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
                f"topn={self.topn} max_pos={self.max_pos} min_trade_score={self.min_trade_score:.3f} "
                f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
            )
        # --- LIVE SAFETY POLICY ---
        # If DRY_RUN=0 (live), require ARMED=1, ARM_TOKEN non-empty, and LIVE_KILL_SWITCH=0
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")
        self.dry_run_env = _env_bool("DRY_RUN", True)

        self.live_allowed = (not self.dry_run_env) and self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16)
        if not self.dry_run_env and not self.live_allowed:
            print(f"[MasterExecutor][SAFE] live blocked: DRY_RUN=0 but ARMED={self.armed} KILL={self.kill_switch} ARM_TOKEN_len={len(self.arm_token)} -> will NOT publish intents")

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[MasterExecutor] XGROUP created: stream={stream} group={group} start_id={start_id}")
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
        s = side.strip().lower()
        if s in ("buy", "long"):
            return "long"
        if s in ("sell", "short"):
            return "short"
        return s or "long"

    def _candidate_score(self, c: Dict[str, Any]) -> float:
        raw = c.get("raw") or {}
        return _safe_float(c.get("_score_selected", c.get("score_total", raw.get("_score_total", 0.0))), 0.0)

    def _is_whale_contra(self, side: str, raw: Dict[str, Any]) -> bool:
        whale_dir = _safe_str(raw.get("whale_dir", "none")).lower()
        whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
        whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")
        if whale_is_buy and side == "short":
            return True
        if whale_is_sell and side == "long":
            return True
        return False

    def _make_intent(self, c: Dict[str, Any]) -> Optional[TradeIntent]:
        raw = c.get("raw") or {}
        score = self._candidate_score(c)

        # hard drop by score
        if score < float(self.min_trade_score):
            return None

        base_lev = int(_safe_float(c.get("recommended_leverage", 5), 5))
        base_npct = float(_safe_float(c.get("recommended_notional_pct", 0.05), 0.05))

        conf = _safe_float(raw.get("confidence", 0.5), 0.5)
        atr_pct = _safe_float(raw.get("atr_pct", 0.01), 0.01)
        spread_pct = _safe_float(raw.get("spread_pct", 0.0003), 0.0003)

        # base adjusters
        conf_adj = _clamp(0.7 + conf, 0.7, 1.7)
        atr_adj = _clamp(0.015 / max(atr_pct, 1e-6), 0.4, 1.2)
        spr_adj = _clamp(0.0004 / max(spread_pct, 1e-9), 0.4, 1.2)

        side = self._normalize_side(_safe_str(c.get("side", raw.get("side", ""))))

        reasons = [str(x) for x in _as_list(c.get("reasons"))]
        risk_tags = [str(x) for x in _as_list(c.get("risk_tags"))]

        # aligned check: reasons contains whale_align_*
        whale_aligned = ("whale_align_long" in reasons) or ("whale_align_short" in reasons)

        # whale boost / contra drop
        whale_score = _safe_float(raw.get("whale_score", 0.0), 0.0)
        whale_contra = self._is_whale_contra(side, raw)

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

        npct = float(base_npct) * conf_adj * _clamp(atr_adj, 0.6, 1.1) * _clamp(spr_adj, 0.6, 1.1) * whale_mult_npct
        npct = float(_clamp(float(npct), float(self.notional_min_pct), float(self.notional_max_pct)))

        # high_vol -> notional+lev kıs (crash-proof)
        if self.high_vol_tag in (risk_tags or []):
            hnp = float(getattr(self, "high_vol_npct_mult", 0.80))
            hlv = float(getattr(self, "high_vol_lev_mult", 0.85))
            lev = int(_clamp(float(int(round(float(lev) * hlv))), float(self.lev_min), float(self.lev_max)))
            npct = float(_clamp(float(npct) * hnp, float(self.notional_min_pct), float(self.notional_max_pct)))
            reasons = reasons + ["master_high_vol_clamp"]

        # whale aligned ise: agresif floor uygula (lev/npct asla çok düşmesin)
        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            lev = max(lev, int(self.whale_lev_floor))
            npct = max(npct, float(self.whale_npct_floor))

        return TradeIntent(
            intent_id=str(uuid.uuid4()),
            ts_utc=_now_utc_iso(),
            symbol=_safe_str(c.get("symbol", "")).upper(),
            interval=_safe_str(c.get("interval", "")),
            side=side,
            score=float(score),
            recommended_leverage=int(lev),
            recommended_notional_pct=float(npct),
            reasons=reasons,
            risk_tags=risk_tags,
            raw=dict(c),
        )

    def _select_top_unique(self, items: List[Dict[str, Any]]) -> List[TradeIntent]:
        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for c in items:
            sym = _safe_str(c.get("symbol", "")).upper()
            if not sym:
                continue
            sc = self._candidate_score(c)
            prev = best_by_symbol.get(sym)
            if (prev is None) or (sc > self._candidate_score(prev)):
                best_by_symbol[sym] = c

        intents: List[TradeIntent] = []
        for c in best_by_symbol.values():
            it = self._make_intent(c)
            if it is not None:
                intents.append(it)

        intents.sort(key=lambda x: float(x.score), reverse=True)
        return intents[: max(0, int(self.topn))]

    def _xreadgroup(self, start_id: str) -> List[Tuple[str, Dict[str, Any]]]:
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

        for _stream_name, entries in resp:
            for sid, fields in entries:
                pkg = self._parse_pkg(sid, fields)
                if pkg:
                    out.append((sid, pkg))
                else:
                    out.append((sid, {}))
        return out

    def _publish_intents(self, source_stream_id: str, intents: List[TradeIntent]) -> Optional[str]:
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
                    "score": it.score,
                    "recommended_leverage": it.recommended_leverage,
                    "recommended_notional_pct": it.recommended_notional_pct,
                    "reasons": it.reasons,
                    "risk_tags": it.risk_tags,
                    "raw": it.raw,
                }
                for it in intents
            ],
        }

        sid = self.r.xadd(
            self.out_stream,
            {"json": json.dumps(payload, ensure_ascii=False)},
            maxlen=5000,
            approximate=True,
        )

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
                    intents = self._select_top_unique(items)
                    if not intents:
                        continue
                    out_id = self._publish_intents(source_stream_id=sid, intents=intents)
                    if out_id:
                        summary = ", ".join(
                            [f"{it.symbol}:{it.side}@L{it.recommended_leverage} npct={it.recommended_notional_pct:.3f}" for it in intents]
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

                intents = self._select_top_unique(items)
                if not intents:
                    continue

                out_id = self._publish_intents(source_stream_id=sid, intents=intents)
                if out_id:
                    summary = ", ".join(
                        [f"{it.symbol}:{it.side}@L{it.recommended_leverage} npct={it.recommended_notional_pct:.3f}" for it in intents]
                    )
                            # gate_publish
        if (not getattr(self, "live_allowed", True)):
            # live blocked -> skip publishing
            continue
print(f"[MasterExecutor] published intents={len(intents)} -> {self.out_stream} id={out_id} | {summary}")

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    MasterExecutor().run_forever()

