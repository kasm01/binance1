from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from orchestration.aggregator.scoring import compute_score, pass_quality_gates
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.state.dup_guard import DupGuard


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


class Aggregator:
    """
    signals_stream -> candidates_stream

    Reads SignalEvent, applies:
      - quality gates (hard)
      - dedup/cooldown (spam-killer)
      - scoring (0..1)
    Publishes CandidateTrade to candidates_stream.

    CandidateTrade (output):
      symbol, side, score_total, recommended_leverage, recommended_notional_pct,
      risk_tags, reason_codes, ts_utc, source, dedup_key
    """

    def __init__(
        self,
        bus: RedisBus,
        group: str = "agg_group",
        consumer: str = "agg_0",
        topk_out: int = 10,        # öneri: candidates_stream'e 10 aday (TopSelector 5'e indirecek)
        cooldown_sec: int = 45,    # öneri: 20-60s arası
    ) -> None:
        self.bus = bus
        self.group = group
        self.consumer = consumer
        self.topk_out = int(topk_out)
        self.dup = DupGuard(bus, prefix="aggdup", default_cooldown_sec=int(cooldown_sec))

    def run_forever(self) -> None:
        assert self.bus.ping(), "[Aggregator] Redis ping failed."
        print(f"[Aggregator] started. reading {self.bus.signals_stream} -> {self.bus.candidates_stream}")

        while True:
            msgs = self.bus.xreadgroup_json(
                stream=self.bus.signals_stream,
                group=self.group,
                consumer=self.consumer,
                count=300,
                block_ms=1000,
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

                # ignore HOLD
                side = str(evt.get("side_candidate", "none") or "none").strip().lower()
                if side not in ("long", "short"):
                    continue

                sym = str(evt.get("symbol", "") or "").upper().strip()
                if not sym:
                    continue
                evt["symbol"] = sym

                if not evt.get("ts_utc"):
                    evt["ts_utc"] = _ts_from_stream_id(mid)

                ok, _reason = pass_quality_gates(evt)
                if not ok:
                    continue

                # dedup_key preferred (new standard)
                dk = str(evt.get("dedup_key", "") or "").strip()
                if not dk:
                    itv = str(evt.get("interval", "") or "").strip()
                    dk = f"{sym}|{itv}|{side}"
                    evt["dedup_key"] = dk

                # cooldown (spam killer)
                if not self.dup.allow(dk, cooldown_sec=None):
                    continue

                score_total, reason_codes, risk_tags = compute_score(evt)
                evt["_score_total"] = _clamp01(score_total)
                evt["_reason_codes"] = list(reason_codes or [])
                evt["_risk_tags"] = list(risk_tags or [])
                evt["_source_stream_id"] = mid
                candidates.append(evt)

            # ACK everything (fail-open)
            try:
                self.bus.xack(self.bus.signals_stream, self.group, mids)
            except Exception:
                pass

            if not candidates:
                continue

            # sort by best score_total
            candidates.sort(key=lambda x: float(x.get("_score_total", 0.0)), reverse=True)
            top = candidates[: self.topk_out]

            for evt in top:
                sym = str(evt.get("symbol", "")).upper()
                itv = str(evt.get("interval", "") or "").strip()
                side = str(evt.get("side_candidate", "long")).strip().lower()
                ts_utc = str(evt.get("ts_utc") or _ts_from_stream_id(str(evt.get("_source_stream_id") or "")))

                # CandidateTrade (standard)
                payload = {
                    "candidate_id": str(uuid.uuid4()),
                    "ts_utc": ts_utc,
                    "symbol": sym,
                    "interval": itv,
                    "side": side,
                    "score_total": float(evt.get("_score_total", 0.0)),
                    "recommended_leverage": int(evt.get("recommended_leverage", 5) or 5),
                    "recommended_notional_pct": float(evt.get("recommended_notional_pct", 0.05) or 0.05),
                    "risk_tags": list(evt.get("_risk_tags", [])),
                    "reason_codes": list(evt.get("_reason_codes", [])),
                    "source": str(evt.get("source") or evt.get("producer_id") or "w?"),
                    "dedup_key": str(evt.get("dedup_key") or f"{sym}|{itv}|{side}"),
                    # ops/debug: raw event (istersen kapatırız)
                    "raw": evt,
                }

                self.bus.publish_candidate(payload)

            time.sleep(0.01)
