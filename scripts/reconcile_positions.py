#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import redis


def _redis() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(redis_url, decode_responses=True)


def _load_json_value(r: redis.Redis, key: str) -> Any:
    raw = r.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _save_json_value(r: redis.Redis, key: str, value: Any) -> None:
    r.set(key, json.dumps(value, ensure_ascii=False))


def _load_orch_state(r: redis.Redis) -> Dict[str, Dict[str, Any]]:
    data = _load_json_value(r, "open_positions_state")
    if isinstance(data, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for sym, item in data.items():
            s = str(sym).upper().strip()
            if s and isinstance(item, dict):
                out[s] = item
        return out
    return {}


def _load_exec_state(r: redis.Redis, prefix: str) -> Dict[str, Dict[str, Any]]:
    pattern = f"{prefix}:*"
    out: Dict[str, Dict[str, Any]] = {}
    for key in r.keys(pattern):
        try:
            sym = str(key).split(":")[-1].upper().strip()
            raw = r.get(key)
            if raw is None:
                continue
            val = json.loads(raw)
            if isinstance(val, dict) and sym:
                out[sym] = val
        except Exception:
            continue
    return out


def _summ_orch(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "side": x.get("side"),
        "interval": x.get("interval"),
        "intent_id": x.get("intent_id"),
        "expires_at": x.get("expires_at"),
    }


def _summ_exec(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "side": x.get("side"),
        "interval": x.get("interval"),
        "entry_price": x.get("entry_price"),
        "qty": x.get("qty"),
        "opened_at": x.get("opened_at"),
    }


def _diff(
    orch: Dict[str, Dict[str, Any]],
    execs: Dict[str, Dict[str, Any]],
) -> Tuple[list[str], list[str], list[str]]:
    orch_syms = set(orch.keys())
    exec_syms = set(execs.keys())
    only_orch = sorted(orch_syms - exec_syms)
    only_exec = sorted(exec_syms - orch_syms)
    both = sorted(orch_syms & exec_syms)
    return only_orch, only_exec, both


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile orchestration and executor position states")
    parser.add_argument("--apply", action="store_true", help="delete symbols from open_positions_state that do not exist in executor")
    parser.add_argument("--apply-exec", action="store_true", help="delete executor keys that do not exist in open_positions_state")
    parser.add_argument("--exec-prefix", default=os.getenv("REDIS_KEY_PREFIX", "bot:positions"), help="executor redis key prefix")
    args = parser.parse_args()

    r = _redis()

    orch = _load_orch_state(r)
    execs = _load_exec_state(r, args.exec_prefix)

    only_orch, only_exec, both = _diff(orch, execs)

    print("=== ONLY IN ORCH ===")
    if only_orch:
        for sym in only_orch:
            print(f"{sym}: {_summ_orch(orch[sym])}")
    else:
        print("(none)")

    print("\n=== ONLY IN EXECUTOR ===")
    if only_exec:
        for sym in only_exec:
            print(f"{sym}: {_summ_exec(execs[sym])}")
    else:
        print("(none)")

    print("\n=== IN BOTH ===")
    if both:
        for sym in both:
            print(
                f"{sym}: "
                f"orch={_summ_orch(orch[sym])} | "
                f"exec={_summ_exec(execs[sym])}"
            )
    else:
        print("(none)")

    changed = False

    if args.apply and only_orch:
        for sym in only_orch:
            orch.pop(sym, None)
        _save_json_value(r, "open_positions_state", orch)
        changed = True
        print(f"\n[APPLY] removed {len(only_orch)} symbol(s) from open_positions_state")

    if args.apply_exec and only_exec:
        for sym in only_exec:
            key = f"{args.exec_prefix}:{sym}"
            r.delete(key)
        changed = True
        print(f"\n[APPLY-EXEC] removed {len(only_exec)} executor key(s)")

    if not args.apply and not args.apply_exec:
        print("\n[DRY-RUN] no changes applied")

    if changed:
        print("[OK] reconciliation changes applied")

    return 0


if __name__ == "__main__":
    sys.exit(main())
