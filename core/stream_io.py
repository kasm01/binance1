# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(k: str, default: int) -> int:
    try:
        return int(str(os.getenv(k, str(default))).strip())
    except Exception:
        return default


def _env_str(k: str, default: str = "") -> str:
    v = os.getenv(k)
    if v is None:
        return default
    return str(v).strip()


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


@dataclass
class StreamConn:
    host: str
    port: int
    db: int
    password: Optional[str] = None
    decode_responses: bool = True
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

    def create(self) -> redis.Redis:
        return redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=self.decode_responses,
            socket_timeout=self.socket_timeout,
            socket_connect_timeout=self.socket_connect_timeout,
        )


def default_stream_conn() -> StreamConn:
    return StreamConn(
        host=_env_str("REDIS_HOST", "127.0.0.1"),
        port=_env_int("REDIS_PORT", 6379),
        db=_env_int("REDIS_DB", 0),
        password=os.getenv("REDIS_PASSWORD") or None,
    )
def ensure_group(r: redis.Redis, stream: str, group: str, start_id: str = "$") -> None:
    """
    Stream + XGROUP create (idempotent).
    """
    try:
        r.xgroup_create(stream, group, id=start_id, mkstream=True)
        print(f"[stream_io] XGROUP created: stream={stream} group={group} start_id={start_id}")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            return
        raise
    except Exception:
        return


def xadd_json(
    r: redis.Redis,
    stream: str,
    payload: Dict[str, Any],
    *,
    maxlen: int = 5000,
    approximate: bool = True,
) -> Optional[str]:
    """
    Publishes {"json": "..."} to stream.
    Returns stream_id or None.
    """
    try:
        s = json.dumps(payload, ensure_ascii=False)
        sid = r.xadd(
            stream,
            {"json": s},
            maxlen=maxlen,
            approximate=approximate,
        )
        return sid
    except Exception:
        return None


def parse_json_field(fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Standard: fields["json"] -> dict
    """
    try:
        s = fields.get("json")
        if not s:
            return None
        d = json.loads(s)
        return d if isinstance(d, dict) else None
    except Exception:
        return None
def normalize_pkg(d: Dict[str, Any], *, source_stream_id: str = "") -> Dict[str, Any]:
    """
    Paket normalize:
      - ts_utc default
      - _source_stream_id ekle
      - items yoksa, tek event'i items=[...] yapabilir (opsiyon)
    """
    out = dict(d) if isinstance(d, dict) else {}
    out.setdefault("ts_utc", _now_utc_iso())
    if source_stream_id:
        out["_source_stream_id"] = source_stream_id
    return out


def pkg_items(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Aşağıdaki inputları destekler:
      1) {"items":[{...},{...}]}
      2) {"event":{...}}  -> items=[event]
      3) doğrudan event dict (caller normalize_pkg ile) -> items=[d]
    """
    if not isinstance(d, dict):
        return []

    items = d.get("items")
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]

    ev = d.get("event")
    if isinstance(ev, dict):
        return [ev]

    # "tek event" gibi duruyorsa:
    # (symbol+interval alanı varsa event kabul et)
    if ("symbol" in d) and ("interval" in d):
        return [d]

    return []


def xack_safe(r: redis.Redis, stream: str, group: str, ids: List[str]) -> None:
    if not ids:
        return
    try:
        r.xack(stream, group, *ids)
    except Exception:
        pass
def xreadgroup_json(
    r: redis.Redis,
    *,
    stream: str,
    group: str,
    consumer: str,
    start_id: str,
    count: int = 50,
    block_ms: int = 2000,
    auto_recover_group: bool = True,
    group_start_id: str = "$",
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns: [(stream_id, pkg_dict), ...]
    """
    try:
        resp = r.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: start_id},
            count=count,
            block=block_ms,
        )
    except redis.exceptions.ResponseError as e:
        msg = str(e)
        # bazen stream blocking read sırasında silinirse:
        if auto_recover_group and ("UNBLOCKED" in msg and "no longer exists" in msg):
            try:
                ensure_group(r, stream, group, start_id=group_start_id)
            except Exception:
                pass
            return []
        return []
    except Exception:
        return []

    out: List[Tuple[str, Dict[str, Any]]] = []
    if not resp:
        return out

    for _stream_name, entries in resp:
        for sid, fields in entries:
            d = parse_json_field(fields) or {}
            d = normalize_pkg(d, source_stream_id=sid)
            out.append((sid, d))
    return out
def sleep_backoff(idle_count: int, *, base: float = 0.02, cap: float = 0.50) -> None:
    """
    Idle durumunda CPU yakmamak için küçük backoff.
    """
    try:
        t = min(cap, base * max(1, idle_count))
        time.sleep(float(t))
    except Exception:
        time.sleep(0.05)


def debug_print_pkg(prefix: str, sid: str, pkg: Dict[str, Any], *, max_items: int = 2) -> None:
    try:
        items = pkg_items(pkg)
        head = items[: max(0, int(max_items))]
        print(
            f"{prefix} sid={sid} ts={_safe_str(pkg.get('ts_utc',''))} "
            f"items={len(items)} head={head}"
        )
    except Exception:
        pass
