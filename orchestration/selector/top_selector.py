from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import redis


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stream_id_to_epoch_ms(stream_id: str) -> int:
    # Redis stream id: "1768642244264-0" => ms part
    try:
        return int(stream_id.split("-", 1)[0])
    except Exception:
        return int(time.time() * 1000)


def _epoch_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


class TopSelector:
    """
    candidates_stream -> top5_stream

    Candidate json example (from your output):
      {
        "candidate_id": "...",
        "ts_utc": null,
        "symbol": "...",
        "interval": "5m",
        "side": "long",
        "score_total": 0.91,
        "risk_tags": [...],
        "recommended_notional_pct": 0.05,
        "recommended_leverage": 5,
        "raw": {...}
      }
    """

    def __init__(self) -> None:
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD") or None
        self.redis_db = int(os.getenv("REDIS_DB", "0"))

        self.candidates_stream = os.getenv("CANDIDATES_STREAM", "candidates_stream")
        self.out_stream = os.getenv("TOP5_STREAM", "top5_stream")

        self.window_sec = _env_int("TOPSEL_WINDOW_SEC", 30)
        self.topk = _env_int("TOPSEL_TOPK", 5)
        self.read_block_ms = _env_int("TOPSEL_BLOCK_MS", 2000)
        self.batch_count = _env_int("TOPSEL_BATCH", 200)

        # cooldown: aynı cooldown_key için tekrar göndermeyi engelle
        self.cooldown_sec = _env_int("TOPSEL_COOLDOWN_SEC", _env_int("TG_DUPLICATE_SIGNAL_COOLDOWN_SEC", 20))

        # Risk tag filtreleri (istersen kapat / değiştir)
        # Örn: wide_spread çoksa eleyelim; şimdilik soft filtre: skor kırpma yapıyoruz
        self.penalty_wide_spread = _env_float("TOPSEL_PENALTY_WIDE_SPREAD", 0.15)
        self.penalty_high_vol = _env_float("TOPSEL_PENALTY_HIGH_VOL", 0.05)

        self.r = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            db=self.redis_db,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

        # state
        self.last_id = os.getenv("TOPSEL_START_ID", "$")  # only new by default
        self.last_sent: Dict[str, float] = {}  # cooldown_key -> epoch_sec

    def _score(self, c: Dict[str, Any]) -> float:
        s = float(c.get("score_total", 0.0) or 0.0)
        tags = c.get("risk_tags") or []
        try:
            tags = list(tags)
        except Exception:
            tags = []

        if "wide_spread" in tags:
            s -= self.penalty_wide_spread
        if "high_vol" in tags:
            s -= self.penalty_high_vol
        return s

    def _cooldown_key(self, c: Dict[str, Any]) -> str:
        # Öncelik: raw.cooldown_key -> yoksa symbol|interval|side
        raw = c.get("raw") or {}
        ck = raw.get("cooldown_key")
        if ck:
            return str(ck)
        return f"{c.get('symbol','')}|{c.get('interval','')}|{c.get('side','')}"

    def _normalize_ts(self, c: Dict[str, Any], stream_id: str) -> None:
        if not c.get("ts_utc"):
            c["ts_utc"] = _epoch_ms_to_iso(_stream_id_to_epoch_ms(stream_id))

    def _read_new(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns list of (stream_id, candidate_dict)
        Reads ONLY candidates_stream.
        """
        resp = self.r.xread({self.candidates_stream: self.last_id}, count=self.batch_count, block=self.read_block_ms)
        if not resp:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        for _stream_name, entries in resp:
            for sid, fields in entries:
                self.last_id = sid
                js = fields.get("json")
                if not js:
                    continue
                try:
                    c = json.loads(js)
                    if isinstance(c, dict):
                        out.append((sid, c))
                except Exception:
                    continue
        return out

    def _select_topk(self, items: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Window + dedupe + cooldown + sort
        """
        if not items:
            return []

        now_sec = time.time()
        min_ms = int((now_sec - self.window_sec) * 1000)

        # 1) window filter + normalize ts
        windowed: List[Tuple[str, Dict[str, Any]]] = []
        for sid, c in items:
            ms = _stream_id_to_epoch_ms(sid)
            if ms < min_ms:
                continue
            self._normalize_ts(c, sid)
            windowed.append((sid, c))

        if not windowed:
            return []

        # 2) dedupe by cooldown_key (keep best score)
        best_by_key: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
        for sid, c in windowed:
            ck = self._cooldown_key(c)
            sc = self._score(c)
            prev = best_by_key.get(ck)
            if (prev is None) or (sc > prev[0]):
                best_by_key[ck] = (sc, sid, c)

        # 3) cooldown filter
        filtered: List[Tuple[float, str, Dict[str, Any]]] = []
        for ck, (sc, sid, c) in best_by_key.items():
            last = self.last_sent.get(ck, 0.0)
            if (now_sec - last) < self.cooldown_sec:
                continue
            filtered.append((sc, sid, c))

        if not filtered:
            return []

        # 4) sort desc score
        filtered.sort(key=lambda t: t[0], reverse=True)

        # 5) take topk
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

        sid = self.r.xadd(self.out_stream, {"json": json.dumps(payload)}, maxlen=2000, approximate=True)

        # update cooldown table
        now_sec = time.time()
        for c in top:
            ck = self._cooldown_key(c)
            self.last_sent[ck] = now_sec

        return sid

    def run_forever(self) -> None:
        print(f"[TopSelector] started. in={self.candidates_stream} out={self.out_stream} "
              f"window={self.window_sec}s topk={self.topk} cooldown={self.cooldown_sec}s "
              f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}")

        buf: List[Tuple[str, Dict[str, Any]]] = []

        while True:
            new_items = self._read_new()
            if new_items:
                buf.extend(new_items)

            # küçük bir temizlik: buffer çok büyümesin
            if len(buf) > 5000:
                buf = buf[-2000:]

            top = self._select_topk(buf)
            if top:
                out_id = self._publish(top)
                if out_id:
                    syms = ", ".join([f"{x.get('symbol')}:{x.get('side')}" for x in top])
                    print(f"[TopSelector] published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

            # loop pacing
            time.sleep(0.2)


if __name__ == "__main__":
    TopSelector().run_forever()
