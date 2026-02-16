from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        fx = float(x)
    except Exception:
        fx = lo
    return max(lo, min(hi, fx))


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


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_id_to_epoch_ms(stream_id: str) -> int:
    try:
        return int(stream_id.split("-", 1)[0])
    except Exception:
        return int(time.time() * 1000)


def _epoch_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


class TopSelector:
    """
    candidates_stream -> top5_stream

    Reads CandidateTrade messages from candidates_stream.
    Selects best TOPSEL_TOPK candidates within a time window,
    de-dupes by dedup_key (cooldown key), applies cooldown, then publishes:

      {"ts_utc": "...", "topk": N, "items": [candidate,...]}

    Notes:
      - Restart-safe via consumer groups.
      - "0 trade normal" rule supported via TOPSEL_MIN_SCORE (or MIN_TOPSEL_SCORE).
    """

    def __init__(self) -> None:
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None
        self.redis_db = _env_int("REDIS_DB", 0)

        self.in_stream = os.getenv("TOPSEL_IN_STREAM", os.getenv("CANDIDATES_STREAM", "candidates_stream"))
        self.out_stream = os.getenv("TOPSEL_OUT_STREAM", os.getenv("TOP5_STREAM", "top5_stream"))

        self.group = os.getenv("TOPSEL_GROUP", "topsel_g")
        self.consumer = os.getenv("TOPSEL_CONSUMER", "topsel_1")
        self.group_start_id = os.getenv("TOPSEL_GROUP_START_ID", "$")
        self.drain_pending = _env_bool("TOPSEL_DRAIN_PENDING", False)

        self.window_sec = _env_int("TOPSEL_WINDOW_SEC", 30)
        self.topk = _env_int("TOPSEL_TOPK", 5)
        self.read_block_ms = _env_int("TOPSEL_BLOCK_MS", 2000)
        self.batch_count = _env_int("TOPSEL_BATCH", 200)
        self.cooldown_sec = _env_int("TOPSEL_COOLDOWN_SEC", _env_int("TG_DUPLICATE_SIGNAL_COOLDOWN_SEC", 20))

        # "0 trade normal" threshold
        self.min_score = _env_float("TOPSEL_MIN_SCORE", _env_float("MIN_TOPSEL_SCORE", 0.10))

        # Penalties
        self.penalty_wide_spread = _env_float("TOPSEL_PENALTY_WIDE_SPREAD", 0.15)
        self.penalty_high_vol = _env_float("TOPSEL_PENALTY_HIGH_VOL", 0.05)

        # If enabled, drop candidates missing required core fields
        self.require_fields = _env_bool("TOPSEL_REQUIRE_FIELDS", True)

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        self.last_sent: Dict[str, float] = {}  # cooldown_key -> epoch_sec

        print(
            f"[TopSelector] init ok. in={self.in_stream} out={self.out_stream} group={self.group} consumer={self.consumer} "
            f"topk={self.topk} window={self.window_sec}s cooldown={self.cooldown_sec}s min_score={self.min_score}"
        )

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[TopSelector] XGROUP created: stream={stream} group={group} start_id={start_id}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise
        except Exception:
            return

    def _required_ok(self, c: Dict[str, Any]) -> bool:
        if not self.require_fields:
            return True
        sym = str(c.get("symbol", "") or "").strip()
        side = str(c.get("side", "") or "").strip().lower()
        itv = str(c.get("interval", "") or "").strip()
        if not sym or not itv or side not in ("long", "short"):
            return False
        return True

    def _risk_tags(self, c: Dict[str, Any]) -> List[str]:
        tags = c.get("risk_tags") or []
        try:
            return [str(x) for x in list(tags)]
        except Exception:
            return []

    def _score(self, c: Dict[str, Any]) -> float:
        # base
        s = _clamp(float(c.get("score_total", 0.0) or 0.0), 0.0, 1.0)

        # penalties
        tags = self._risk_tags(c)
        if "wide_spread" in tags:
            s -= self.penalty_wide_spread
        if "high_vol" in tags:
            s -= self.penalty_high_vol

        return _clamp(s, 0.0, 1.0)

    def _cooldown_key(self, c: Dict[str, Any]) -> str:
        # new standard
        dk = c.get("dedup_key")
        if dk:
            return str(dk)

        # backward-compat
        raw = c.get("raw") or {}
        if isinstance(raw, dict):
            ck = raw.get("cooldown_key")
            if ck:
                return str(ck)

        return f"{c.get('symbol','')}|{c.get('interval','')}|{c.get('side','')}"

    def _normalize_ts(self, c: Dict[str, Any], stream_id: str) -> None:
        if not c.get("ts_utc"):
            c["ts_utc"] = _epoch_ms_to_iso(_stream_id_to_epoch_ms(stream_id))

    def _xreadgroup(self, start_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            resp = self.r.xreadgroup(
                groupname=self.group,
                consumername=self.consumer,
                streams={self.in_stream: start_id},
                count=self.batch_count,
                block=self.read_block_ms,
            )
        except redis.exceptions.ResponseError as e:
            msg = str(e)
            if "UNBLOCKED" in msg and "no longer exists" in msg:
                self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)
                return []
            return []
        except Exception:
            return []

        if not resp:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        for _stream_name, entries in resp:
            for sid, fields in entries:
                js = fields.get("json")
                if not js:
                    out.append((sid, {}))
                    continue
                try:
                    c = json.loads(js)
                    out.append((sid, c if isinstance(c, dict) else {}))
                except Exception:
                    out.append((sid, {}))
        return out

    def _ack(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self.r.xack(self.in_stream, self.group, *ids)
        except Exception:
            pass

    def _select_topk(self, items: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not items:
            return []

        now_sec = time.time()
        min_ms = int((now_sec - self.window_sec) * 1000)

        windowed: List[Tuple[str, Dict[str, Any]]] = []
        for sid, c in items:
            if not c:
                continue
            if not isinstance(c, dict):
                continue

            # time window filter uses stream id
            ms = _stream_id_to_epoch_ms(sid)
            if ms < min_ms:
                continue

            self._normalize_ts(c, sid)

            # optional field sanity
            if not self._required_ok(c):
                continue

            windowed.append((sid, c))

        if not windowed:
            return []

        # best per cooldown key
        best_by_key: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
        for sid, c in windowed:
            ck = self._cooldown_key(c)
            sc = self._score(c)
            prev = best_by_key.get(ck)
            if (prev is None) or (sc > prev[0]):
                best_by_key[ck] = (sc, sid, c)

        # cooldown filter + min_score
        filtered: List[Tuple[float, str, Dict[str, Any]]] = []
        for ck, (sc, sid, c) in best_by_key.items():
            if sc < self.min_score:
                continue
            last = self.last_sent.get(ck, 0.0)
            if (now_sec - last) < self.cooldown_sec:
                continue
            filtered.append((sc, sid, c))

        if not filtered:
            return []

        filtered.sort(key=lambda t: t[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for sc, sid, c in filtered[: self.topk]:
            c["_score_selected"] = float(sc)
            c["_source_stream_id"] = sid
            out.append(c)
        return out

    def _publish(self, top: List[Dict[str, Any]]) -> Optional[str]:
        if not top:
            return None

        payload = {
            "ts_utc": _now_utc_iso(),
            "topk": len(top),
            "items": top,
        }

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=2000,
                approximate=True,
            )
        except Exception:
            return None

        now_sec = time.time()
        for c in top:
            ck = self._cooldown_key(c)
            self.last_sent[ck] = now_sec

        return sid

    def run_forever(self) -> None:
        print(
            f"[TopSelector] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"window={self.window_sec}s topk={self.topk} cooldown={self.cooldown_sec}s min_score={self.min_score} "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

        if self.drain_pending:
            print("[TopSelector] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break

                top = self._select_topk(rows)
                out_id = self._publish(top) if top else None
                if out_id:
                    syms = ", ".join([f"{x.get('symbol')}:{x.get('side')}" for x in top])
                    print(f"[TopSelector] (PEL) published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

                self._ack([sid for sid, _ in rows])
                time.sleep(0.05)

            print("[TopSelector] pending drained.")

        while True:
            rows = self._xreadgroup(">")
            if not rows:
                continue

            top = self._select_topk(rows)
            out_id = self._publish(top) if top else None

            if out_id:
                syms = ", ".join([f"{x.get('symbol')}:{x.get('side')}" for x in top])
                print(f"[TopSelector] published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

            self._ack([sid for sid, _ in rows])
            time.sleep(0.05)


if __name__ == "__main__":
    TopSelector().run_forever()
