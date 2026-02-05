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

        # Whale-first scoring controls (final score)
        self.w_w = _env_float("MASTER_W_SCORE_WHALE", 0.60)
        self.w_mtf = _env_float("MASTER_W_SCORE_MTF", 0.25)
        self.w_micro = _env_float("MASTER_W_SCORE_MICRO", 0.15)

        # Heavy-on-top5 toggles (stage-2)
        self.heavy_enable = _env_bool("MASTER_HEAVY_ENABLE", False)
        self.heavy_topk = _env_int("MASTER_HEAVY_TOPK", 5)
        self.heavy_discard_below = _env_float("MASTER_HEAVY_DISCARD_BELOW", 0.70)

        # Eğer heavy aktifse, score gate’i daha sıkı yap
        if self.heavy_enable:
            self.min_trade_score = max(float(self.min_trade_score), float(self.heavy_discard_below))

        # Risk sizing clamp
        self.lev_min = _env_int("LEV_MIN", 3)
        self.lev_max = _env_int("LEV_MAX", 30)
        self.notional_min_pct = _env_float("NOTIONAL_MIN_PCT", 0.02)
        self.notional_max_pct = _env_float("NOTIONAL_MAX_PCT", 0.25)

        # Whale aggression controls
        self.whale_boost_thr = _env_float("MASTER_WHALE_BOOST_THR", 0.20)
        self.whale_lev_boost = _env_float("MASTER_WHALE_LEV_BOOST", 1.35)      # multiplier
        self.whale_npct_boost = _env_float("MASTER_WHALE_NPCT_BOOST", 1.20)    # multiplier
        self.whale_lev_floor = _env_int("MASTER_WHALE_LEV_FLOOR", 8)
        self.whale_npct_floor = _env_float("MASTER_WHALE_NPCT_FLOOR", 0.04)

        # High-volatility risk clamp
        self.high_vol_tag = os.getenv("HIGH_VOL_TAG", "high_vol")
        self.high_vol_npct_mult = _env_float("HIGH_VOL_NPCT_MULT", 0.80)
        self.high_vol_lev_mult = _env_float("HIGH_VOL_LEV_MULT", 0.85)

        # Contra whale safety
        self.drop_if_whale_contra = os.getenv("MASTER_DROP_WHALE_CONTRA", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

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
            f"heavy_enable={self.heavy_enable} heavy_topk={self.heavy_topk} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

        # --- LIVE SAFETY POLICY ---
        self.armed = _env_bool("ARMED", False)
        self.kill_switch = _env_bool("LIVE_KILL_SWITCH", False)
        self.arm_token = _env_str("ARM_TOKEN", "")
        self.dry_run_env = _env_bool("DRY_RUN", True)

        self.live_allowed = (not self.dry_run_env) and self.armed and (not self.kill_switch) and (len(self.arm_token) >= 16)
        if not self.dry_run_env and not self.live_allowed:
            print(
                f"[MasterExecutor][SAFE] live blocked: DRY_RUN=0 but "
                f"ARMED={self.armed} KILL={self.kill_switch} ARM_TOKEN_len={len(self.arm_token)} "
                f"-> will NOT publish intents"
            )

        self._warned_heavy_stub = False

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
        return _safe_float(c.get('_score_total_final', c.get('_score_selected', c.get('score_total', raw.get('_score_total', 0.0)))), 0.0)

        def _heavy_score_one(self, c: Dict[str, Any]) -> Tuple[float, List[str]]:
            """Best-effort heavy scorer.
        If no heavy model wired yet, returns fast score.
        Returns: (score_heavy, reasons)
        """
        raw = c.get('raw') or {}
        # default to fast score
        base = float(self._candidate_score(c))
        # TODO: wire real MTF+SGD+LSTM here (repo-specific)
        return base, ['heavy_passthrough']

def _is_whale_contra(self, side: str, raw: Dict[str, Any]) -> bool:
        whale_dir = _safe_str(raw.get("whale_dir", "none")).lower()
        whale_is_buy = whale_dir in ("buy", "long", "in", "inflow")
        whale_is_sell = whale_dir in ("sell", "short", "out", "outflow")
        if whale_is_buy and side == "short":
            return True
        if whale_is_sell and side == "long":
            return True
        return False

    def _compute_final_score(self, c: Dict[str, Any]) -> float:
        """
        Whale-first final score.
        heavy_enable=True ise ileride burada MTF+LSTM çağrısı eklenecek.
        Şimdilik: mtf_score kaynağı olarak candidate score/fast_model_score/p_used kullanır.
        """
        raw = c.get("raw") or {}

        whale = _safe_float(raw.get("whale_score", c.get("whale_score", 0.0)), 0.0)
        micro = _safe_float(
            raw.get("micro_score", c.get("micro_score", raw.get("vol_score", 0.0))),
            0.0,
        )

        # mtf_score: heavy’de ayrı hesap olacak; şimdilik eldeki en iyi proxy
        mtf = _safe_float(
            raw.get("mtf_score", c.get("mtf_score", c.get("fast_model_score", raw.get("p_used", c.get("p_used", 0.0))))),
            0.0,
        )

        # clamp to 0..1
        whale = _clamp(whale, 0.0, 1.0)
        micro = _clamp(micro, 0.0, 1.0)
        mtf = _clamp(mtf, 0.0, 1.0)

        score = (self.w_w * whale) + (self.w_mtf * mtf) + (self.w_micro * micro)
        score = _clamp(score, 0.0, 1.0)

        if self.heavy_enable and not self._warned_heavy_stub:
            print("[MasterExecutor][HEAVY] enabled but using stub final-score (no MTF+LSTM call yet).")
            self._warned_heavy_stub = True

        return float(score)

    def _make_intent(self, c: Dict[str, Any]) -> Optional[TradeIntent]:
        raw = c.get("raw") or {}

        # Final score (whale-first)
        score = self._compute_final_score(c)

        # hard drop by score
        if score < float(self.min_trade_score):
            return None

        base_lev = int(_safe_float(c.get("recommended_leverage", 5), 5))
        base_npct = float(_safe_float(c.get("recommended_notional_pct", 0.05), 0.05))

        conf = _safe_float(raw.get("confidence", 0.5), 0.5)
        atr_pct = _safe_float(raw.get("atr_pct", 0.01), 0.01)
        spread_pct = _safe_float(raw.get("spread_pct", 0.0003), 0.0003)

        conf_adj = _clamp(0.7 + conf, 0.7, 1.7)
        atr_adj = _clamp(0.015 / max(atr_pct, 1e-6), 0.4, 1.2)
        spr_adj = _clamp(0.0004 / max(spread_pct, 1e-9), 0.4, 1.2)

        side = self._normalize_side(_safe_str(c.get("side", raw.get("side", ""))))

        reasons = [str(x) for x in _as_list(c.get("reasons"))]
        risk_tags = [str(x) for x in _as_list(c.get("risk_tags"))]

        whale_aligned = ("whale_align_long" in reasons) or ("whale_align_short" in reasons)
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

        if self.high_vol_tag in (risk_tags or []):
            lev = int(_clamp(float(int(round(float(lev) * float(self.high_vol_lev_mult)))), float(self.lev_min), float(self.lev_max)))
            npct = float(_clamp(float(npct) * float(self.high_vol_npct_mult), float(self.notional_min_pct), float(self.notional_max_pct)))
            reasons = reasons + ["master_high_vol_clamp"]

        if whale_aligned and whale_score >= float(self.whale_boost_thr):
            lev = max(lev, int(self.whale_lev_floor))
            npct = max(npct, float(self.whale_npct_floor))

        # score_total_final'i raw içine yaz (debug için)
        c = dict(c)
        c["_score_total_final"] = float(score)

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
        """
        items: top5 paket içinden gelir.
        heavy_enable ise: önce final-score’a göre topK seç, sonra unique+intent üret.
        """
        t0 = time.time()

        # 1) candidates -> score
        scored: List[Dict[str, Any]] = []
        for c in items:
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            c2["_score_total_final"] = self._compute_final_score(c2)
            scored.append(c2)

        scored.sort(key=lambda x: float(x.get("_score_total_final", 0.0)), reverse=True)

        # heavy_topk sadece “ön eleme” (latency azaltma için)
        if self.heavy_enable and self.heavy_topk > 0:
            scored = scored[: int(self.heavy_topk)]

        # 2) unique best-by-symbol
        best_by_symbol: Dict[str, Dict[str, Any]] = {}
        for c in scored:
            sym = _safe_str(c.get("symbol", "")).upper()
            if not sym:
                continue
            sc = _safe_float(c.get("_score_total_final", 0.0), 0.0)
            prev = best_by_symbol.get(sym)
            if (prev is None) or (sc > _safe_float(prev.get("_score_total_final", 0.0), 0.0)):
                best_by_symbol[sym] = c

        # 3) intents
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

        def _apply_heavy_stage(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optionally run heavy scoring on topK items, filter by threshold.
        Adds: _score_heavy, _score_total_final, _reasons_final
        """
        if not getattr(self, 'heavy_enable', False):
            return items
        k = max(0, int(getattr(self, 'heavy_topk', 0) or 0))
        thr = float(getattr(self, 'heavy_discard_below', 0.0) or 0.0)
        if k <= 0:
            return items
        out: List[Dict[str, Any]] = []
        # score topK; rest pass-through
        for i, c in enumerate(items):
            c2 = dict(c)
            fast = float(self._candidate_score(c2))
            if i < k:
                t0 = time.time()
                hs, hreas = self._heavy_score_one(c2)
                ms = int(round((time.time() - t0) * 1000.0))
                sym = _safe_str(c2.get('symbol','')).upper()
                print(f"[MasterExecutor][HEAVY][LAT] {sym} {ms}ms heavy={hs:.3f} fast={fast:.3f}")
                c2['_score_heavy'] = float(hs)
                # choose final score (max for now)
                final = float(max(fast, float(hs)))
                c2['_score_total_final'] = final
                c2['_reasons_final'] = list(_as_list(c2.get('reasons'))) + list(hreas or [])
                if final < thr:
                    continue
            else:
                c2['_score_total_final'] = fast
                c2['_reasons_final'] = list(_as_list(c2.get('reasons')))
            out.append(c2)
        # sort by final desc
        out.sort(key=lambda x: float(x.get('_score_total_final', 0.0)), reverse=True)
        return out

def _publish_intents(self, source_stream_id: str, intents: List[TradeIntent]) -> Optional[str]:
        if not getattr(self, "live_allowed", True):
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
                items = self._apply_heavy_stage(items)
                items = self._apply_heavy_stage(items)
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
                    print(f"[MasterExecutor] published intents={len(intents)} -> {self.out_stream} id={out_id} | {summary}")

            self._ack(mids)
            time.sleep(0.05)


if __name__ == "__main__":
    MasterExecutor().run_forever()
