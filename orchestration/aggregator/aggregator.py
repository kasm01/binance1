from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Tuple

from orchestration.event_bus.redis_bus import RedisBus
from orchestration.state.dup_guard import DupGuard
from orchestration.aggregator.scoring import compute_score, pass_quality_gates


def _ts_from_stream_id(mid: str) -> str:
    """
    Redis stream id: '1768833274051-0' -> epoch ms -> ISO
    """
    try:
        ms = int(str(mid).split("-", 1)[0])
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


class Aggregator:
    """
    signals_stream -> candidates_stream

    Restart-safe: uses Redis consumer groups via RedisBus.xreadgroup_json(...)
    """

    def __init__(
        self,
        bus: RedisBus,
        group: str = "agg_group",
        consumer: str = "agg_0",
        topk_out: int = 5,
        cooldown_sec: int = 30,
    ) -> None:
        self.bus = bus
        self.group = group
        self.consumer = consumer
        self.topk_out = int(topk_out)
        self.dup = DupGuard(bus, prefix="aggdup", default_cooldown_sec=int(cooldown_sec))

    def run_forever(self) -> None:
        assert self.bus.ping(), "[Aggregator] Redis ping failed."
        print("[Aggregator] started. reading signals_stream ...")

        while True:
            msgs = self.bus.xreadgroup_json(
                stream=self.bus.signals_stream,
                group=self.group,
                consumer=self.consumer,
                count=200,
                block_ms=1000,
                start_id=">",
                create_group=True,
            )

            if not msgs:
                continue

            mids: List[str] = []
            candidates: List[Dict[str, Any]] = []

            for mid, evt in msgs:
                mids.append(mid)

                ok, _reason = pass_quality_gates(evt)
                if not ok:
                    continue

                ck = str(evt.get("cooldown_key") or "")
                if not ck:
                    sym = str(evt.get("symbol", "")).upper()
                    side = str(evt.get("side_candidate", "none"))
                    itv = str(evt.get("interval", ""))
                    ck = f"{sym}|{itv}|{side}"

                if not self.dup.allow(ck, cooldown_sec=None):
                    continue

                score, reasons, risk_tags = compute_score(evt)
                evt["_score_total"] = float(score)
                evt["_reasons"] = reasons
                evt["_risk_tags"] = risk_tags
                evt["_source_stream_id"] = mid
                candidates.append(evt)

            # ACK everything (fail-open)
            try:
                self.bus.xack(self.bus.signals_stream, self.group, mids)
            except Exception:
                pass

            if not candidates:
                continue

            # pick topk
            candidates.sort(key=lambda x: float(x.get("_score_total", 0.0)), reverse=True)
            top = candidates[: self.topk_out]

            for evt in top:
                sym = str(evt.get("symbol", "")).upper()
                itv = str(evt.get("interval", ""))
                side = str(evt.get("side_candidate", "long"))

                payload = {
                    "candidate_id": str(uuid.uuid4()),
                    "ts_utc": str(evt.get("ts_utc") or _ts_from_stream_id(str(evt.get("_source_stream_id") or ""))),
                    "symbol": sym,
                    "interval": itv,
                    "side": side,
                    "score_total": float(evt.get("_score_total", 0.0)),
                    "reasons": list(evt.get("_reasons", [])),
                    "risk_tags": list(evt.get("_risk_tags", [])),
                    "recommended_notional_pct": 0.05,  # placeholder -> risk module will refine
                    "recommended_leverage": 5,         # placeholder -> risk module will refine
                    "raw": evt,
                }
                self.bus.publish_candidate(payload)
