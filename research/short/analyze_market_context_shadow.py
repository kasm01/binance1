#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.event_bus.redis_bus import RedisBus


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except Exception:
        return int(default)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def main() -> None:
    stream = os.getenv("SHORT_ENTRY_CONTEXT_SHADOW_STREAM", "telemetry:market_context_shadow")
    count = max(1, _i("SHORT_ENTRY_CONTEXT_SHADOW_ANALYZE_COUNT", 5000))
    bus = RedisBus()
    if not bus.ping():
        raise SystemExit("MARKET_CONTEXT_SHADOW_ANALYZER: Redis unavailable")

    rows = bus.r.xrevrange(stream, max="+", min="-", count=count)
    events: List[Dict[str, Any]] = []
    for _sid, fields in rows:
        try:
            raw = (fields or {}).get("json")
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            obj = json.loads(raw or "{}")
            if isinstance(obj, dict) and obj.get("schema_version") == "market_context_shadow.v1":
                events.append(obj)
        except Exception:
            continue

    if not events:
        print(f"[CTX-SHADOW-ANALYZE] stream={stream} events=0")
        return

    transitions = Counter()
    paths = Counter()
    symbols = Counter()
    available = 0
    changed = 0
    sizing_changed = 0
    master_build = 0
    entry_build = 0

    for e in events:
        cmp0 = e.get("comparison") if isinstance(e.get("comparison"), dict) else {}
        risk = e.get("hypothetical_risk") if isinstance(e.get("hypothetical_risk"), dict) else {}
        tr = str(cmp0.get("transition") or "unknown")
        transitions[tr] += 1
        paths[str(e.get("entry_path") or "unknown")] += 1
        symbols[str(e.get("symbol") or "unknown")] += 1
        if bool(cmp0.get("available")):
            available += 1
        if cmp0.get("changed") is True:
            changed += 1
        if bool(risk.get("sizing_signal_changed")):
            sizing_changed += 1
        mr = cmp0.get("master_regime") if isinstance(cmp0.get("master_regime"), dict) else {}
        er = cmp0.get("entry_regime") if isinstance(cmp0.get("entry_regime"), dict) else {}
        master_build += int(str(mr.get("name") or "") == "confirmed_build")
        entry_build += int(str(er.get("name") or "") == "confirmed_build")

    n = len(events)
    print(f"[CTX-SHADOW-ANALYZE] stream={stream} events={n} available={available}/{n}")
    print(f"changed={changed}/{n} ({100.0*changed/n:.2f}%) sizing_signal_changed={sizing_changed}/{n} ({100.0*sizing_changed/n:.2f}%)")
    print(f"master_confirmed_build={master_build}/{n} ({100.0*master_build/n:.2f}%) entry_confirmed_build={entry_build}/{n} ({100.0*entry_build/n:.2f}%)")
    print("transitions:")
    for k, v in transitions.most_common():
        print(f"  {k:34s} {v:6d} {100.0*v/n:7.2f}%")
    print("entry_paths:")
    for k, v in paths.most_common():
        print(f"  {k:34s} {v:6d} {100.0*v/n:7.2f}%")
    print("symbols:")
    for k, v in symbols.most_common():
        print(f"  {k:34s} {v:6d} {100.0*v/n:7.2f}%")


if __name__ == "__main__":
    main()
