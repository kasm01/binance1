from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from orchestration.aggregator.scoring import compute_score, pass_quality_gates
from orchestration.event_bus.redis_bus import RedisBus
from orchestration.state.dup_guard import DupGuard


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
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        fx = float(x)
    except Exception:
        fx = 0.0
    return max(lo, min(hi, fx))


def _ms_from_stream_id(mid: str) -> int:
    try:
        return int(str(mid).split("-", 1)[0])
    except Exception:
        return 0


def _ts_from_stream_id(mid: str) -> str:
    ms = _ms_from_stream_id(mid)
    if ms > 0:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _age_s_from_mid(mid: str) -> int:
    ms = _ms_from_stream_id(mid)
    if ms <= 0:
        return -1
    now_ms = int(time.time() * 1000)
    return max(0, int((now_ms - ms) / 1000))


def _fmt_topline(items: List[Dict[str, Any]], limit: int = 6) -> str:
    out = []
    for it in items[:limit]:
        sym = str(it.get("symbol", "?"))
        side = str(it.get("side", it.get("side_candidate", "?")))
        sc = it.get("score_total", it.get("_score_total", 0.0))
        out.append(f"{sym}:{side}@{float(sc):.3f}")
    return " ".join(out)


class Aggregator:
    """
    signals_stream -> candidates_stream

    Reads SignalEvent from signals_stream (consumer group),
    applies quality gates + dedup + scoring,
    selects topk_out and publishes CandidateTrade into candidates_stream.

    Notes:
      - ACK is fail-open (we ack even if we drop).
      - side_candidate="none" is ignored (HOLD).
      - Default publish mode is per-candidate message (backward compatible).
        Optional batch mode can be enabled via env.
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

        # Defaults here; can be overridden by env at runtime (see run_forever)
        self.topk_out = int(topk_out)
        self.dup = DupGuard(bus, prefix="aggdup", default_cooldown_sec=int(cooldown_sec))

    def run_forever(self) -> None:
        assert self.bus.ping(), "[Aggregator] Redis ping failed."

        # Runtime tuning (env)
        topk_out = _env_int("AGG_TOPK_OUT", self.topk_out)
        read_count = _env_int("AGG_READ_COUNT", 200)
        block_ms = _env_int("AGG_BLOCK_MS", 1000)
        pace_sleep = _env_float("AGG_PACE_SLEEP", 0.01)

        # Filters
        min_score = _env_float("AGG_MIN_SCORE", 0.0)  # e.g. 0.60
        max_age_s = _env_int("AGG_MAX_SIGNAL_AGE_S", 0)  # 0 disables age filter

        # Output mode
        publish_batch = _env_bool("AGG_PUBLISH_BATCH", False)  # default False for compatibility
        log_every = _env_int("AGG_LOG_EVERY", 20)  # loop iterations

        print(
            "[Aggregator] started. "
            f"in={self.bus.signals_stream} out={self.bus.candidates_stream} "
            f"group={self.group} consumer={self.consumer} "
            f"topk_out={topk_out} read_count={read_count} block_ms={block_ms} "
            f"min_score={min_score:.3f} max_age_s={max_age_s} "
            f"publish_batch={publish_batch}",
            flush=True,
        )

        loop_i = 0
        while True:
            loop_i += 1

            msgs = self.bus.xreadgroup_json(
                stream=self.bus.signals_stream,
                group=self.group,
                consumer=self.consumer,
                count=read_count,
                block_ms=block_ms,
                start_id=">",
                create_group=True,
                group_start_id="$",
            )

            if not msgs:
                continue

            # telemetry
            mids: List[str] = []
            candidates: List[Dict[str, Any]] = []

            drop_invalid = 0
            drop_hold = 0
            drop_symbol = 0
            drop_old = 0
            drop_gate = 0
            drop_dup = 0
            drop_score = 0

            last_mid = ""
            for mid, evt in msgs:
                mids.append(mid)
                last_mid = mid

                if not isinstance(evt, dict) or not evt:
                    drop_invalid += 1
                    continue

                side = str(evt.get("side_candidate", "none")).strip().lower()
                if side in ("none", "", "hold"):
                    drop_hold += 1
                    continue

                sym = str(evt.get("symbol", "")).upper().strip()
                if not sym:
                    drop_symbol += 1
                    continue
                evt["symbol"] = sym

                # ts_utc
                if not evt.get("ts_utc"):
                    evt["ts_utc"] = _ts_from_stream_id(mid)

                # age gate (optional)
                if max_age_s and max_age_s > 0:
                    age_s = _age_s_from_mid(mid)
                    if age_s >= 0 and age_s > max_age_s:
                        drop_old += 1
                        continue

                ok, _reason = pass_quality_gates(evt)
                if not ok:
                    drop_gate += 1
                    continue

                ck = str(evt.get("dedup_key") or evt.get("cooldown_key") or "").strip()
                if not ck:
                    itv = str(evt.get("interval", "")).strip()
                    ck = f"{sym}|{itv}|{side}"

                if not self.dup.allow(ck, cooldown_sec=None):
                    drop_dup += 1
                    continue

                score, reasons, risk_tags = compute_score(evt)
                score_total = _clamp(score, 0.0, 1.0)

                # score threshold (optional)
                if min_score > 0.0 and score_total < min_score:
                    drop_score += 1
                    continue

                evt["_score_total"] = score_total
                evt["_reasons"] = list(reasons or [])
                evt["_risk_tags"] = list(risk_tags or [])
                evt["_source_stream_id"] = mid
                evt["_age_s"] = _age_s_from_mid(mid)

                candidates.append(evt)

            # ACK everything (fail-open)
            try:
                if mids:
                    self.bus.xack(self.bus.signals_stream, self.group, mids)
            except Exception:
                pass

            if not candidates:
                if loop_i % log_every == 0:
                    age_last = _age_s_from_mid(last_mid) if last_mid else -1
                    print(
                        "[Aggregator][STAT] empty_out "
                        f"in_msgs={len(msgs)} "
                        f"drop_invalid={drop_invalid} drop_hold={drop_hold} drop_symbol={drop_symbol} "
                        f"drop_old={drop_old} drop_gate={drop_gate} drop_dup={drop_dup} drop_score={drop_score} "
                        f"last_signal_id={last_mid or 'na'} age_s={age_last}",
                        flush=True,
                    )
                continue

            # pick topk
            candidates.sort(key=lambda x: float(x.get("_score_total", 0.0)), reverse=True)
            top = candidates[: max(1, int(topk_out))]

            # build CandidateTrade payloads
            out_items: List[Dict[str, Any]] = []
            for evt in top:
                sym = str(evt.get("symbol", "")).upper()
                itv = str(evt.get("interval", "")).strip()
                side = str(evt.get("side_candidate", "long")).strip().lower()

                payload = {
                    "candidate_id": str(uuid.uuid4()),
                    "ts_utc": str(evt.get("ts_utc") or _ts_from_stream_id(str(evt.get("_source_stream_id") or ""))),
                    "symbol": sym,
                    "interval": itv,
                    "side": side,
                    "score_total": float(evt.get("_score_total", 0.0)),
                    "reasons": list(evt.get("_reasons", [])),
                    "risk_tags": list(evt.get("_risk_tags", [])),
                    # keep defaults but allow upstream to pass
                    "recommended_notional_pct": float(evt.get("recommended_notional_pct", 0.05) or 0.05),
                    "recommended_leverage": int(evt.get("recommended_leverage", 5) or 5),

                    # debug extras (helpful for later stages)
                    "source_worker": str(evt.get("producer_id") or evt.get("source") or ""),
                    "source_stream_id": str(evt.get("_source_stream_id") or ""),
                    "source_age_s": int(evt.get("_age_s") or 0),

                    # raw event (debug altın)
                    "raw": evt,
                }
                out_items.append(payload)

            # publish
            last_out_id = ""
            try:
                if publish_batch:
                    # batch mode: 1 message contains N items
                    batch_payload = {"count": len(out_items), "items": out_items}
                    last_out_id = str(self.bus.publish_candidate(batch_payload))
                else:
                    # per-item mode (backward compatible)
                    for it in out_items:
                        last_out_id = str(self.bus.publish_candidate(it))
            except Exception as e:
                # fail-open: don't crash the pipeline
                print(f"[Aggregator][WARN] publish failed: {e!r}", flush=True)

            # telemetry log
            if loop_i % log_every == 0:
                age_last = _age_s_from_mid(last_mid) if last_mid else -1
                print(
                    "[Aggregator][STAT] "
                    f"in_msgs={len(msgs)} mids={len(mids)} cand_ok={len(candidates)} out={len(out_items)} "
                    f"drop_invalid={drop_invalid} drop_hold={drop_hold} drop_symbol={drop_symbol} "
                    f"drop_old={drop_old} drop_gate={drop_gate} drop_dup={drop_dup} drop_score={drop_score} "
                    f"last_signal_id={last_mid or 'na'} age_s={age_last} "
                    f"last_candidate_id={last_out_id or 'na'} "
                    f"top={_fmt_topline(out_items)}",
                    flush=True,
                )

            # small pacing
            if pace_sleep > 0:
                time.sleep(pace_sleep)
