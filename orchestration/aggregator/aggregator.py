# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestration.aggregator.scoring import compute_score, pass_quality_gates
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.schemas.events import SignalEvent


def _ts_from_stream_id(mid: str) -> str:
    try:
        ms = int(str(mid).split("-", 1)[0])
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _clamp01(x: Any) -> float:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    return max(0.0, min(1.0, f))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


def _norm_side_candidate(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    if s in ("none", "hold", ""):
        return "none"
    # unknown -> none (fail closed)
    return "none"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class Aggregator:
    """
    signals_stream -> candidates_stream

    Reads SignalEvent, applies:
      - quality gates (hard)
      - dedup/cooldown (spam-killer)
      - scoring (0..1)
    Publishes CandidateTrade-like dict to candidates_stream.

    NOTE:
      - price passthrough is CRITICAL (downstream needs it).
      - run as module: python -m orchestration.aggregator.aggregator
    """

    def __init__(
        self,
        bus: RedisBus,
        group: str = "agg_group",
        consumer: str = "agg_0",
        topk_out: int = 10,
        cooldown_sec: int = 45,
        include_raw_default: bool = True,
    ) -> None:
        self.bus = bus
        self.group = str(os.getenv("AGG_GROUP", group))
        self.consumer = str(os.getenv("AGG_CONSUMER", consumer))
        self.topk_out = int(_env_int("AGG_TOPK_OUT", int(topk_out)))

        # cooldown / dedup
        # NOTE: DupGuard lives in orchestration/state/dup_guard.py (existing in your repo)
        from orchestration.state.dup_guard import DupGuard  # local import: avoids import cycle

        self.dup = DupGuard(bus, prefix=os.getenv("AGG_DUP_PREFIX", "aggdup"), default_cooldown_sec=int(cooldown_sec))

        # raw payload stream'i şişirebilir; env ile kapatılabilir
        self.include_raw = _env_bool("CANDIDATE_INCLUDE_RAW", include_raw_default)

        # "0 trade normal" mantığının aggregator versiyonu:
        # skor bu eşikten küçükse candidates_stream'e basma.
        self.min_score = float(_env_float("AGG_MIN_SCORE", 0.0))

        # read tuning
        self.read_count = int(_env_int("AGG_READ_COUNT", 300))
        self.block_ms = int(_env_int("AGG_BLOCK_MS", 1000))

        # debug: neden drop oldu görmek için (smoke testte kullandın)
        self.debug_drops = _env_bool("AGG_DEBUG_DROPS", False)

    def _normalize_event(self, mid: str, evt: Dict[str, Any]) -> Dict[str, Any]:
        e = dict(evt)

        sym = str(e.get("symbol", "") or "").upper().strip()
        if sym:
            e["symbol"] = sym

        itv = str(e.get("interval", "") or "").strip() or "5m"
        e["interval"] = itv

        e["side_candidate"] = _norm_side_candidate(e.get("side_candidate", "none"))

        if not e.get("ts_utc"):
            e["ts_utc"] = _ts_from_stream_id(mid)

        # --- PRICE normalize (critical) ---
        p = _safe_float(e.get("price", 0.0), 0.0)
        e["price"] = float(p) if p > 0 else 0.0

        # dedup_key standard
        dk = str(e.get("dedup_key", "") or "").strip()
        if not dk and sym:
            dk = f"{sym}|{itv}|{e['side_candidate']}"
            e["dedup_key"] = dk

        e["_source_stream_id"] = mid
        return e

    def _build_candidate_payload(self, evt: Dict[str, Any]) -> Dict[str, Any]:
        sym = str(evt.get("symbol", "") or "").upper().strip()
        itv = str(evt.get("interval", "") or "5m").strip()
        side = str(evt.get("side_candidate", "none") or "none").strip().lower()
        ts_utc = str(evt.get("ts_utc") or _ts_from_stream_id(str(evt.get("_source_stream_id") or "")))

        src = str(evt.get("source") or evt.get("producer_id") or "w?").strip()
        dedup_key = str(evt.get("dedup_key") or f"{sym}|{itv}|{side}")

        # --- PRICE passthrough ---
        price = _safe_float(evt.get("price", 0.0), 0.0)
        if price < 0:
            price = 0.0

        payload: Dict[str, Any] = {
            "candidate_id": str(uuid.uuid4()),
            "ts_utc": ts_utc,
            "symbol": sym,
            "interval": itv,
            "side": side,
            "price": float(price),  # <<< CRITICAL
            "score_total": float(_clamp01(evt.get("_score_total", 0.0))),
            # leverage / notional: şimdilik default; MasterExecutor daha iyi hesaplar
            "recommended_leverage": int(evt.get("recommended_leverage", 5) or 5),
            "recommended_notional_pct": float(evt.get("recommended_notional_pct", 0.05) or 0.05),
            "risk_tags": list(evt.get("_risk_tags", []) or []),
            "reason_codes": list(evt.get("_reason_codes", []) or []),
            "source": src,
            "dedup_key": dedup_key,
        }

        # raw payload policy:
        # include_raw=1 -> full raw
        # include_raw=0 -> send "raw_min" (TopSelector/Master still can gate whale)
        if self.include_raw:
            payload["raw"] = evt
        else:
            meta = evt.get("meta") if isinstance(evt.get("meta"), dict) else {}
            payload["raw"] = {
                "price": float(price),
                "whale_score": float(evt.get("whale_score", 0.0) or 0.0),
                "whale_dir": str(evt.get("whale_dir") or meta.get("whale_dir", "none") or "none"),
                "meta": {
                    "p_used": float(meta.get("p_used", 0.0) or 0.0),
                    "fast_model_score": float(meta.get("fast_model_score", 0.0) or 0.0),
                    "micro_score": float(meta.get("micro_score", 0.0) or 0.0),
                },
            }

        return payload
    def run_forever(self) -> None:
        assert self.bus.ping(), "[Aggregator] Redis ping failed."
        print(
            f"[Aggregator] started. reading {self.bus.signals_stream} -> {self.bus.candidates_stream} "
            f"group={self.group} consumer={self.consumer} topk_out={self.topk_out} "
            f"min_score={self.min_score:.3f} include_raw={self.include_raw} "
            f"dup_prefix={getattr(self.dup, 'prefix', 'aggdup')} debug_drops={self.debug_drops}",
            flush=True,
        )

        while True:
            msgs = self.bus.xreadgroup_json(
                stream=self.bus.signals_stream,
                group=self.group,
                consumer=self.consumer,
                count=self.read_count,
                block_ms=self.block_ms,
                start_id=">",
                create_group=True,
                group_start_id="$",
            )

            if not msgs:
                continue

            mids: List[str] = []
            candidates: List[Dict[str, Any]] = []

            for mid, evt in msgs:
                mids.append(mid)

                if not isinstance(evt, dict) or not evt:
                    if self.debug_drops:
                        print(f"[Aggregator][DROP] invalid_evt mid={mid}", flush=True)
                    continue

                # schema normalize + local normalize
                evt = SignalEvent.normalize(evt)
                e = self._normalize_event(mid, evt)

                # ignore HOLD/none
                if e.get("side_candidate") not in ("long", "short"):
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] side_none symbol={e.get('symbol')} itv={e.get('interval')} "
                            f"side={e.get('side_candidate')} mid={mid}",
                            flush=True,
                        )
                    continue

                if not e.get("symbol"):
                    if self.debug_drops:
                        print(f"[Aggregator][DROP] no_symbol mid={mid}", flush=True)
                    continue

                ok, reason = pass_quality_gates(e)
                if not ok:
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] gate_fail reason={reason} "
                            f"sym={e.get('symbol')} itv={e.get('interval')} side={e.get('side_candidate')} mid={mid}",
                            flush=True,
                        )
                    continue

                # cooldown (spam killer)
                dk = str(e.get("dedup_key", "") or "").strip()
                if not dk:
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] no_dedup_key sym={e.get('symbol')} itv={e.get('interval')} mid={mid}",
                            flush=True,
                        )
                    continue

                if not self.dup.allow(dk, cooldown_sec=None):
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] dup_cooldown dk={dk} sym={e.get('symbol')} itv={e.get('interval')} mid={mid}",
                            flush=True,
                        )
                    continue

                score_total, reason_codes, risk_tags = compute_score(e)
                e["_score_total"] = _clamp01(score_total)
                e["_reason_codes"] = list(reason_codes or [])
                e["_risk_tags"] = list(risk_tags or [])

                # global min score gate (optional)
                if float(self.min_score) > 0.0 and float(e["_score_total"]) < float(self.min_score):
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] min_score "
                            f"score={float(e['_score_total']):.3f} < {float(self.min_score):.3f} "
                            f"sym={e.get('symbol')} itv={e.get('interval')} side={e.get('side_candidate')} mid={mid}",
                            flush=True,
                        )
                    continue

                candidates.append(e)

            # ACK everything (fail-open)
            if mids:
                try:
                    self.bus.xack(self.bus.signals_stream, self.group, mids)
                except Exception:
                    pass

            if not candidates:
                continue

            # sort by best score_total
            candidates.sort(key=lambda x: float(x.get("_score_total", 0.0)), reverse=True)
            top = candidates[: max(1, int(self.topk_out))]

            for evt in top:
                payload = self._build_candidate_payload(evt)
                try:
                    self.bus.publish_candidate(payload)
                except Exception:
                    pass

            time.sleep(0.01)


if __name__ == "__main__":
    # minimal runnable entrypoint
    bus = RedisBus()
    agg = Aggregator(bus=bus)
    agg.run_forever()
