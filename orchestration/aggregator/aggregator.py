from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from orchestration.aggregator.scoring import compute_score, pass_quality_gates
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.state.dup_guard import DupGuard
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
class Aggregator:
    """
    signals_stream -> candidates_stream

    Reads SignalEvent, applies:
      - quality gates (hard)
      - dedup/cooldown (spam-killer)
      - scoring (0..1)
    Publishes CandidateTrade to candidates_stream.

    SignalEvent (input) expects:
      symbol, interval, side_candidate, score_edge, confidence,
      atr_pct, spread_pct, liq_score, whale_score(optional),
      ts_utc, source(w1..w8), dedup_key

    CandidateTrade (output):
      symbol, side, score_total, recommended_leverage, recommended_notional_pct,
      risk_tags, reason_codes, ts_utc, source, dedup_key
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
        self.group = group
        self.consumer = consumer
        self.topk_out = int(topk_out)

        self.dup = DupGuard(bus, prefix="aggdup", default_cooldown_sec=int(cooldown_sec))

        # raw payload stream'i şişirebilir; env ile kapatılabilir
        self.include_raw = _env_bool("CANDIDATE_INCLUDE_RAW", include_raw_default)

        # "0 trade normal" mantığının aggregator versiyonu:
        # skor bu eşikten küçükse candidates_stream'e basma.
        self.min_score = float(_env_float("AGG_MIN_SCORE", 0.0))

        # read tuning
        self.read_count = int(_env_int("AGG_READ_COUNT", 300))
        self.block_ms = int(_env_int("AGG_BLOCK_MS", 1000))

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

        payload: Dict[str, Any] = {
            "candidate_id": str(uuid.uuid4()),
            "ts_utc": ts_utc,
            "symbol": sym,
            "interval": itv,
            "side": side,
            "score_total": float(_clamp01(evt.get("_score_total", 0.0))),
            # leverage / notional: şimdilik default; MasterExecutor daha iyi hesaplayacak
            "recommended_leverage": int(evt.get("recommended_leverage", 5) or 5),
            "recommended_notional_pct": float(evt.get("recommended_notional_pct", 0.05) or 0.05),
            "risk_tags": list(evt.get("_risk_tags", []) or []),
            "reason_codes": list(evt.get("_reason_codes", []) or []),
            "source": src,
            "dedup_key": dedup_key,
        }

        # --- raw payload policy ---
        # include_raw=1 -> full raw
        # include_raw=0 -> still send "raw_min" so TopSelector/MasterExecutor gates keep working
        if self.include_raw:
            payload["raw"] = evt
        else:
            meta = evt.get("meta") if isinstance(evt.get("meta"), dict) else {}
            payload["raw"] = {
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
        print(f"[Aggregator] started. reading {self.bus.signals_stream} -> {self.bus.candidates_stream} "
              f"topk_out={self.topk_out} min_score={self.min_score:.3f} include_raw={self.include_raw}")

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
                    continue


                evt = SignalEvent.normalize(evt)
                e = self._normalize_event(mid, evt)

                # ignore HOLD/none
                if e.get("side_candidate") not in ("long", "short"):
                    continue

                if not e.get("symbol"):
                    continue

                ok, _reason = pass_quality_gates(e)
                if not ok:
                    continue

                # cooldown (spam killer)
                dk = str(e.get("dedup_key", "") or "").strip()
                if not dk:
                    continue
                if not self.dup.allow(dk, cooldown_sec=None):
                    continue

                score_total, reason_codes, risk_tags = compute_score(e)
                e["_score_total"] = _clamp01(score_total)
                e["_reason_codes"] = list(reason_codes or [])
                e["_risk_tags"] = list(risk_tags or [])

                # global min score gate (optional)
                if float(self.min_score) > 0.0 and float(e["_score_total"]) < float(self.min_score):
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
