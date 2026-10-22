#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import List
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests

from orchestration.event_bus.redis_bus import RedisBus

FAPI = "https://fapi.binance.com"


def _symbols() -> List[str]:
    raw = os.getenv("CONTEXT_SYMBOLS") or os.getenv("WORKER_SYMBOLS") or os.getenv("SYMBOLS") or ""
    out: List[str] = []
    for item in raw.split(","):
        s = item.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def fetch_oi_context(session: requests.Session, symbol: str) -> dict:
    r = session.get(
        f"{FAPI}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": "5m", "limit": 5},
        timeout=8,
    )
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or len(rows) < 5:
        raise RuntimeError(f"oi_history_short:{symbol}:{len(rows) if isinstance(rows, list) else 'bad'}")
    rows = sorted(rows, key=lambda x: int(x.get("timestamp") or 0))
    oi = [_safe_float(x.get("sumOpenInterest")) for x in rows]
    ts = [int(x.get("timestamp") or 0) for x in rows]
    latest = oi[-1]
    previous = oi[-2]
    if latest <= 0 or previous <= 0 or oi[-3] <= 0 or oi[-4] <= 0 or oi[-5] <= 0:
        raise RuntimeError(f"oi_invalid:{symbol}")
    return {
        "symbol": symbol,
        "source": "binance_usdm_openInterestHist",
        "period": "5m",
        "source_ts_ms": ts[-1],
        "sum_open_interest": latest,
        "oi_delta_5m": latest / oi[-2] - 1.0,
        "oi_delta_15m": latest / oi[-4] - 1.0,
        # Strictly-prior fallback for an entry that occurs exactly on the
        # latest OI timestamp boundary. Same HTTP request; no extra API cost.
        "previous_source_ts_ms": ts[-2],
        "previous_sum_open_interest": previous,
        "previous_oi_delta_5m": previous / oi[-3] - 1.0,
        "previous_oi_delta_15m": previous / oi[-5] - 1.0,
        "fetched_at": _now_iso(),
    }


def run_once() -> int:
    symbols = _symbols()
    if not symbols:
        print("[OIContext][NO-SYMBOLS]", flush=True)
        return 2
    bus = RedisBus()
    if not bus.ping():
        print("[OIContext][REDIS-DOWN]", flush=True)
        return 3
    ttl = max(600, int(float(os.getenv("OI_CONTEXT_TTL_SEC", "900") or 900)))
    session = requests.Session()
    ok = 0
    for symbol in symbols:
        try:
            ctx = fetch_oi_context(session, symbol)
            bus.r.setex(f"context:oi:{symbol}", ttl, json.dumps(ctx, separators=(",", ":")))
            print(
                f"[OIContext][OK] {symbol} oi5={ctx['oi_delta_5m']:.6f} "
                f"oi15={ctx['oi_delta_15m']:.6f} ts={ctx['source_ts_ms']}",
                flush=True,
            )
            ok += 1
        except Exception as exc:
            print(f"[OIContext][WARN] {symbol} err={str(exc)[:180]}", flush=True)
    return 0 if ok == len(symbols) else 1


def main() -> None:
    once = "--once" in os.sys.argv
    if once:
        raise SystemExit(run_once())
    poll = max(60, int(float(os.getenv("OI_CONTEXT_POLL_SEC", "300") or 300)))
    while True:
        started = time.time()
        run_once()
        elapsed = time.time() - started
        time.sleep(max(5.0, poll - elapsed))


if __name__ == "__main__":
    main()
