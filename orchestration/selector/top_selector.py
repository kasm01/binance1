from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis

from orchestration.schemas.events import TopKBatchEvent


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_side(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return s


def _as_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t) for t in x if str(t).strip()]
    if isinstance(x, tuple):
        return [str(t) for t in list(x) if str(t).strip()]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    try:
        return [str(x)]
    except Exception:
        return []


class TopSelector:
    """
    candidates_stream -> top5_stream

    Buffers incoming candidates for WINDOW seconds, then selects best TOPK in that window,
    de-dupes by dedup_key, applies cooldown, then publishes ONE batch event:

      {"ts_utc": "...", "topk": N, "items": [candidate,...]}

    Notes:
      - MIN_SCORE fallback supported for min_score
      - W_MIN / TOPSEL_W_MIN whale gate: candidate must have whale_score >= threshold (if threshold > 0)
      - price alanını korur (top-level + raw içinde).
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
        if self.window_sec <= 0:
            self.window_sec = 30

        self.topk = _env_int("TOPSEL_TOPK", 5)
        self.read_block_ms = _env_int("TOPSEL_BLOCK_MS", 2000)
        self.batch_count = _env_int("TOPSEL_BATCH", 200)

        # cooldown: prevent repeating same dedup_key too frequently
        self.cooldown_sec = _env_int("TOPSEL_COOLDOWN_SEC", _env_int("TG_DUPLICATE_SIGNAL_COOLDOWN_SEC", 20))

        # "0 trade normal" threshold
        self.min_score = _env_float(
            "TOPSEL_MIN_SCORE",
            _env_float("MIN_TOPSEL_SCORE", _env_float("MIN_SCORE", 0.10)),
        )

        # whale minimum gate (TopSelector stage)
        self.w_min = _env_float("TOPSEL_W_MIN", _env_float("W_MIN", 0.0))

        # penalties
        self.penalty_wide_spread = _env_float("TOPSEL_PENALTY_WIDE_SPREAD", 0.15)
        self.penalty_high_vol = _env_float("TOPSEL_PENALTY_HIGH_VOL", 0.05)

        self.require_fields = _env_bool("TOPSEL_REQUIRE_FIELDS", True)

        self.out_maxlen = _env_int("TOPSEL_OUT_MAXLEN", 5000)
        self.buffer_max = _env_int("TOPSEL_BUFFER_MAX", 2000)

        # If 1: ack immediately (at-most-once). Default 0: ack after flush (at-least-once)
        self.ack_immediate = _env_bool("TOPSEL_ACK_IMMEDIATE", False)

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

        # Window buffers
        self._buf: List[Tuple[str, Dict[str, Any]]] = []   # (sid, candidate)
        self._buf_ids: List[str] = []                      # sids to ACK after flush
        self._window_started_at: float = time.time()

        print(
            f"[TopSelector] init ok. in={self.in_stream} out={self.out_stream} group={self.group} consumer={self.consumer} "
            f"topk={self.topk} window={self.window_sec}s cooldown={self.cooldown_sec}s min_score={self.min_score} "
            f"w_min={self.w_min} ack_immediate={self.ack_immediate}"
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

    def _normalize_candidate(self, c: Dict[str, Any], stream_id: str) -> Dict[str, Any]:
        d = dict(c) if isinstance(c, dict) else {}

        sym = str(d.get("symbol", "") or "").upper().strip()
        if sym:
            d["symbol"] = sym

        itv = str(d.get("interval", "") or "").strip() or "5m"
        d["interval"] = itv

        d["side"] = _norm_side(d.get("side", ""))

        if not d.get("ts_utc"):
            d["ts_utc"] = _epoch_ms_to_iso(_stream_id_to_epoch_ms(stream_id))

        dk = str(d.get("dedup_key", "") or "").strip()
        if not dk and sym:
            dk = f"{sym}|{itv}|{d.get('side','')}"
            d["dedup_key"] = dk

        # risk_tags normalize
        d["risk_tags"] = _as_str_list(d.get("risk_tags", []))

        # price normalize (keep top-level)
        price = _safe_float(d.get("price", 0.0), 0.0)
        if price < 0:
            price = 0.0
        d["price"] = float(price)

        # whale_score normalize (top-level preferred)
        ws = d.get("whale_score", None)
        if ws is None:
            raw = d.get("raw") or {}
            if isinstance(raw, dict):
                ws = raw.get("whale_score", None)
        if ws is not None:
            d["whale_score"] = _clamp(_safe_float(ws, 0.0), 0.0, 1.0)

        # score_total fallback normalization
        st = d.get("score_total", None)
        if st is None:
            st = d.get("_score_total", None)
        if st is None:
            st = d.get("_score_selected", None)
        if st is None:
            raw = d.get("raw") or {}
            if isinstance(raw, dict):
                st = raw.get("score_total", None)
        if st is not None:
            d["score_total"] = _clamp(_safe_float(st, 0.0), 0.0, 1.0)

        # ensure raw carries price too (downstream fallback)
        try:
            raw = d.get("raw")
            if isinstance(raw, dict):
                raw["price"] = float(price)
        except Exception:
            pass

        return d

    def _required_ok(self, c: Dict[str, Any]) -> bool:
        if not self.require_fields:
            return True
        sym = str(c.get("symbol", "") or "").strip()
        side = str(c.get("side", "") or "").strip().lower()
        itv = str(c.get("interval", "") or "").strip()
        if not sym or not itv or side not in ("long", "short"):
            return False
        return True

    def _extract_whale_score(self, c: Dict[str, Any]) -> float:
        ws = _safe_float(c.get("whale_score", 0.0), 0.0)
        if ws > 0:
            return float(ws)
        raw = c.get("raw") or {}
        if isinstance(raw, dict):
            return float(_safe_float(raw.get("whale_score", 0.0), 0.0))
        return 0.0

    def _score(self, c: Dict[str, Any]) -> float:
        s = _clamp(float(c.get("score_total", 0.0) or 0.0), 0.0, 1.0)
        tags = c.get("risk_tags") or []
        if "wide_spread" in tags:
            s -= float(self.penalty_wide_spread)
        if "high_vol" in tags:
            s -= float(self.penalty_high_vol)
        return _clamp(s, 0.0, 1.0)

    def _cooldown_key(self, c: Dict[str, Any]) -> str:
        dk = c.get("dedup_key")
        if dk:
            return str(dk)
        raw = c.get("raw") or {}
        if isinstance(raw, dict):
            ck = raw.get("cooldown_key")
            if ck:
                return str(ck)
        return f"{c.get('symbol','')}|{c.get('interval','')}|{c.get('side','')}"
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
                    obj = json.loads(js)
                    out.append((sid, obj if isinstance(obj, dict) else {}))
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
        min_ms = int((now_sec - float(self.window_sec)) * 1000)

        windowed: List[Tuple[str, Dict[str, Any]]] = []
        for sid, c in items:
            if not c or not isinstance(c, dict):
                continue

            ms = _stream_id_to_epoch_ms(sid)
            if ms < min_ms:
                continue

            c2 = self._normalize_candidate(c, sid)
            if not self._required_ok(c2):
                continue

            # whale gate (optional)
            if float(self.w_min) > 0.0:
                ws = self._extract_whale_score(c2)
                if ws < float(self.w_min):
                    continue

            windowed.append((sid, c2))

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

        filtered: List[Tuple[float, str, Dict[str, Any]]] = []
        for ck, (sc, sid, c) in best_by_key.items():
            if sc < float(self.min_score):
                continue
            last = self.last_sent.get(ck, 0.0)
            if float(self.cooldown_sec) > 0 and (now_sec - last) < float(self.cooldown_sec):
                continue
            filtered.append((sc, sid, c))

        if not filtered:
            return []

        filtered.sort(key=lambda t: t[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for sc, sid, c in filtered[: max(1, int(self.topk))]:
            c["_score_selected"] = float(sc)
            c["_source_stream_id"] = sid
            out.append(c)
        return out

    def _publish(self, top: List[Dict[str, Any]]) -> Optional[str]:
        if not top:
            return None

        payload = TopKBatchEvent(
            ts_utc=_now_utc_iso(),
            topk=len(top),
            items=top,
            window_sec=int(self.window_sec),
            selector_id=str(self.consumer),
            min_score=float(self.min_score),
            w_min=float(self.w_min),
        ).to_dict()

        try:
            sid = self.r.xadd(
                self.out_stream,
                {"json": json.dumps(payload, ensure_ascii=False)},
                maxlen=self.out_maxlen,
                approximate=True,
            )
        except Exception:
            return None

        now_sec = time.time()
        for c in top:
            ck = self._cooldown_key(c)
            self.last_sent[ck] = now_sec

        return sid

    def _buffer_add(self, rows: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not rows:
            return

        ack_ids: List[str] = []
        for sid, c in rows:
            if not c or not isinstance(c, dict):
                continue
            c2 = self._normalize_candidate(c, sid)
            self._buf.append((sid, c2))
            self._buf_ids.append(sid)
            ack_ids.append(sid)

        if len(self._buf) > self.buffer_max:
            self._buf = self._buf[-self.buffer_max :]
        if len(self._buf_ids) > self.buffer_max:
            self._buf_ids = self._buf_ids[-self.buffer_max :]

        if self.ack_immediate and ack_ids:
            self._ack(ack_ids)

    def _window_ready(self) -> bool:
        return (time.time() - self._window_started_at) >= float(self.window_sec)

    def _flush_window(self) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        if not self._buf:
            self._window_started_at = time.time()
            self._buf_ids = []
            return None

        top = self._select_topk(self._buf)
        out_id = self._publish(top) if top else None

        # ACK after flush (at-least-once)
        if not self.ack_immediate and self._buf_ids:
            self._ack(self._buf_ids)

        # reset window
        self._buf = []
        self._buf_ids = []
        self._window_started_at = time.time()

        if out_id:
            return str(out_id), top
        return None

    def run_forever(self) -> None:
        print(
            f"[TopSelector] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"window={self.window_sec}s topk={self.topk} cooldown={self.cooldown_sec}s min_score={self.min_score} "
            f"w_min={self.w_min} redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
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
                    syms = ", ".join(
                        [f"{x.get('symbol')}:{x.get('side')}({x.get('_score_selected', x.get('score_total')):.3f})" for x in top]
                    )
                    print(f"[TopSelector] (PEL) published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

                self._ack([sid for sid, _ in rows])
                time.sleep(0.05)

            print("[TopSelector] pending drained.")

        while True:
            rows = self._xreadgroup(">")
            if rows:
                self._buffer_add(rows)

            if self._window_ready():
                res = self._flush_window()
                if res:
                    out_id, top = res
                    syms = ", ".join(
                        [f"{x.get('symbol')}:{x.get('side')}({x.get('_score_selected', x.get('score_total')):.3f})" for x in top]
                    )
                    print(f"[TopSelector] published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

            time.sleep(0.05)


if __name__ == "__main__":
    TopSelector().run_forever()
