# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

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



def _norm_side_candidate(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return "none"


def _norm_whale_dir(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long", "inflow", "in"):
        return "long"
    if s in ("sell", "short", "outflow", "out"):
        return "short"
    return "none"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class Aggregator:
    """
    signals_stream -> candidates_stream

    SIGNAL NORMALIZATION authority.

    SignalEvent alır, canonical alan adlarına normalize eder ve Candidate payload basar.
    Hard model/context/risk gate, cooldown, dedup veya score recompute yapmaz.
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
        self.include_raw = _env_bool("CANDIDATE_INCLUDE_RAW", include_raw_default)
        self.read_count = int(_env_int("AGG_READ_COUNT", 300))
        self.block_ms = int(_env_int("AGG_BLOCK_MS", 1000))
        self.debug_drops = _env_bool("AGG_DEBUG_DROPS", False)

        self.require_price = _env_bool("AGG_REQUIRE_PRICE", True)

    def _normalize_event(self, mid: str, evt: Dict[str, Any]) -> Dict[str, Any]:
        e = dict(evt)

        sym = str(e.get("symbol", "") or "").upper().strip()
        e["symbol"] = sym

        itv = str(e.get("interval", "") or "").strip() or "5m"
        e["interval"] = itv

        side = _norm_side_candidate(e.get("side_candidate", e.get("side", "none")))
        e["side_candidate"] = side

        if not e.get("ts_utc"):
            e["ts_utc"] = _ts_from_stream_id(mid)

        price = _safe_float(e.get("price", 0.0), 0.0)
        e["price"] = float(price) if price > 0 else 0.0

        model_score = e.get("model_score", 0.0)
        e["model_score"] = _clamp01(model_score)
        meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
        e["model_provider"] = str(e.get("model_provider") or meta.get("model_provider") or "hybrid_coin_specific")
        e["liq_score"] = _clamp01(e.get("liq_score", 0.0))
        e["spread_pct"] = max(0.0, _safe_float(e.get("spread_pct", 0.0), 0.0))
        e["atr_pct"] = max(0.0, _safe_float(e.get("atr_pct", 0.0), 0.0))
        e["money_flow_15m"] = max(-1.0, min(1.0, _safe_float(e.get("money_flow_15m", 0.0), 0.0)))
        e["money_flow_60m"] = max(-1.0, min(1.0, _safe_float(e.get("money_flow_60m", 0.0), 0.0)))
        e["price_return_15m"] = _safe_float(e.get("price_return_15m", 0.0), 0.0)

        whale_score = e.get("whale_score", meta.get("whale_score", 0.0))
        whale_dir = e.get("whale_dir", meta.get("whale_dir", "none"))
        whale_action = e.get("whale_action", meta.get("whale_action", ""))

        e["whale_score"] = _clamp01(whale_score)
        e["whale_dir"] = _norm_whale_dir(whale_dir)
        e["whale_action"] = str(whale_action or "").strip().lower()

        dk = str(e.get("dedup_key", "") or "").strip()
        if not dk and sym:
            dk = f"{sym}|{itv}|{side}"
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
        price = _safe_float(evt.get("price", 0.0), 0.0)

        payload: Dict[str, Any] = {
            "candidate_id": str(uuid.uuid4()),
            "ts_utc": ts_utc,
            "symbol": sym,
            "interval": itv,
            "side": side,
            "price": float(price),
            "model_score": float(_clamp01(evt.get("model_score", 0.0))),
            "model_provider": str(evt.get("model_provider") or "hybrid_coin_specific"),
            "risk_tags": list(evt.get("_risk_tags", []) or []),
            "reason_codes": list(evt.get("_reason_codes", []) or []),
            "source": src,
            "dedup_key": dedup_key,
            "atr_pct": float(evt.get("atr_pct", 0.0) or 0.0),
            "spread_pct": float(evt.get("spread_pct", 0.0) or 0.0),
            "liq_score": float(evt.get("liq_score", 0.0) or 0.0),
            "money_flow_15m": float(evt.get("money_flow_15m", 0.0) or 0.0),
            "money_flow_60m": float(evt.get("money_flow_60m", 0.0) or 0.0),
            "price_return_15m": float(evt.get("price_return_15m", 0.0) or 0.0),
            "whale_score": float(evt.get("whale_score", 0.0) or 0.0),
            "whale_dir": str(evt.get("whale_dir", "none") or "none"),
            "whale_action": str(evt.get("whale_action", "") or ""),
        }

        if self.include_raw:
            payload["raw"] = evt
        else:
            meta = evt.get("meta") if isinstance(evt.get("meta"), dict) else {}
            payload["raw"] = {
                "price": float(price),
                "whale_score": float(evt.get("whale_score", 0.0) or 0.0),
                "whale_dir": str(evt.get("whale_dir", "none") or "none"),
                "meta": {
                    "p_buy": float(meta.get("p_buy", 0.0) or 0.0),
                    "spread_pct": float(evt.get("spread_pct", 0.0) or 0.0),
                    "liq_score": float(evt.get("liq_score", 0.0) or 0.0),
                    "atr_pct": float(evt.get("atr_pct", 0.0) or 0.0),
                    "money_flow_15m": float(evt.get("money_flow_15m", 0.0) or 0.0),
                    "money_flow_60m": float(evt.get("money_flow_60m", 0.0) or 0.0),
                    "price_return_15m": float(evt.get("price_return_15m", 0.0) or 0.0),
                },
            }

        return payload
    def run_forever(self) -> None:
        assert self.bus.ping(), "[Aggregator] Redis ping failed."
        print(
            f"[Aggregator] started. reading {self.bus.signals_stream} -> {self.bus.candidates_stream} "
            f"group={self.group} consumer={self.consumer} mode=normalize_only "
            f"include_raw={self.include_raw} "
            f"require_price={self.require_price} debug_drops={self.debug_drops}",
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

                evt = SignalEvent.normalize(evt)
                e = self._normalize_event(mid, evt)

                if e.get("side_candidate") not in ("long", "short"):
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] side_none symbol={e.get('symbol')} "
                            f"itv={e.get('interval')} mid={mid}",
                            flush=True,
                        )
                    continue

                if not e.get("symbol"):
                    if self.debug_drops:
                        print(f"[Aggregator][DROP] no_symbol mid={mid}", flush=True)
                    continue

                if self.require_price and float(e.get("price", 0.0) or 0.0) <= 0.0:
                    if self.debug_drops:
                        print(
                            f"[Aggregator][DROP] no_price sym={e.get('symbol')} "
                            f"itv={e.get('interval')} side={e.get('side_candidate')} mid={mid}",
                            flush=True,
                        )
                    continue

                # SIGNAL NORMALIZATION only: no score/context/risk hard gate here.
                # Preserve the MODEL score unchanged for downstream ranking/gating.
                e["_reason_codes"] = ["signal_normalized"]
                e["_risk_tags"] = []

                candidates.append(e)

            if mids:
                try:
                    self.bus.xack(self.bus.signals_stream, self.group, mids)
                except Exception:
                    pass

            if not candidates:
                continue

            for evt in candidates:
                payload = self._build_candidate_payload(evt)
                try:
                    self.bus.publish_candidate(payload)
                except Exception:
                    pass

            time.sleep(0.01)


if __name__ == "__main__":
    bus = RedisBus()
    agg = Aggregator(bus=bus)
    agg.run_forever()
