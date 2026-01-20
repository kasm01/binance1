from __future__ import annotations

import os
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import redis


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

    Restart-safe: uses Redis Consumer Groups.
    Reads from candidates_stream with XREADGROUP and ACKs after successful publish.
    """

    def __init__(self) -> None:
        self.redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        self.redis_port = _env_int("REDIS_PORT", 6379)
        self.redis_password = os.getenv("REDIS_PASSWORD") or None
        self.redis_db = _env_int("REDIS_DB", 0)

        # Streams
        self.in_stream = os.getenv("TOPSEL_IN_STREAM", os.getenv("CANDIDATES_STREAM", "candidates_stream"))
        self.out_stream = os.getenv("TOPSEL_OUT_STREAM", os.getenv("TOP5_STREAM", "top5_stream"))

        # Consumer group
        self.group = os.getenv("TOPSEL_GROUP", "topsel_g")
        self.consumer = os.getenv("TOPSEL_CONSUMER", "topsel_1")

        # Group start policy (only matters when group is first created)
        # "$" -> only new messages
        # "0-0" -> read from beginning
        self.group_start_id = os.getenv("TOPSEL_GROUP_START_ID", "$")

        # Drain pending on startup?
        self.drain_pending = os.getenv("TOPSEL_DRAIN_PENDING", "0").strip().lower() in ("1", "true", "yes", "on")

        # selection params
        self.window_sec = _env_int("TOPSEL_WINDOW_SEC", 30)
        self.topk = _env_int("TOPSEL_TOPK", 5)
        self.read_block_ms = _env_int("TOPSEL_BLOCK_MS", 2000)
        self.batch_count = _env_int("TOPSEL_BATCH", 200)

        # cooldown: aynı cooldown_key için tekrar göndermeyi engelle
        self.cooldown_sec = _env_int("TOPSEL_COOLDOWN_SEC", _env_int("TG_DUPLICATE_SIGNAL_COOLDOWN_SEC", 20))

        # Risk tag soft penalties
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

        self._ensure_group(self.in_stream, self.group, start_id=self.group_start_id)

        # in-memory cooldown table (not persisted)
        self.last_sent: Dict[str, float] = {}  # cooldown_key -> epoch_sec

    def _ensure_group(self, stream: str, group: str, start_id: str = "$") -> None:
        try:
            self.r.xgroup_create(stream, group, id=start_id, mkstream=True)
            print(f"[TopSelector] XGROUP created: stream={stream} group={group} start_id={start_id}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return
            raise

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

    def _xreadgroup(self, start_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        start_id:
          ">" => new messages only
          "0" => pending (PEL) messages for this group
        Returns list of (stream_id, candidate_dict)
        """
        resp = self.r.xreadgroup(
            groupname=self.group,
            consumername=self.consumer,
            streams={self.in_stream: start_id},
            count=self.batch_count,
            block=self.read_block_ms,
        )
        if not resp:
            return []

        out: List[Tuple[str, Dict[str, Any]]] = []
        for _stream_name, entries in resp:
            for sid, fields in entries:
                js = fields.get("json")
                if not js:
                    # boş/bozuk entry -> ack edip geçmek daha temiz
                    out.append((sid, {}))
                    continue
                try:
                    c = json.loads(js)
                    if isinstance(c, dict):
                        out.append((sid, c))
                    else:
                        out.append((sid, {}))
                except Exception:
                    out.append((sid, {}))
        return out

    def _ack(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            # redis-py xack tek tek veya listeyle destekleyebilir; güvenli olarak *ids açıyoruz
            self.r.xack(self.in_stream, self.group, *ids)
        except Exception:
            pass

    def _select_topk(self, items: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Window + dedupe + cooldown + sort
        items: list of (sid, candidate)
        """
        if not items:
            return []

        now_sec = time.time()
        min_ms = int((now_sec - self.window_sec) * 1000)

        # 1) window filter + normalize ts
        windowed: List[Tuple[str, Dict[str, Any]]] = []
        for sid, c in items:
            if not c:
                continue
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

        sid = self.r.xadd(
            self.out_stream,
            {"json": json.dumps(payload, ensure_ascii=False)},
            maxlen=2000,
            approximate=True,
        )

        # update cooldown table (in-memory)
        now_sec = time.time()
        for c in top:
            ck = self._cooldown_key(c)
            self.last_sent[ck] = now_sec

        return sid

    def run_forever(self) -> None:
        print(
            f"[TopSelector] started. in={self.in_stream} out={self.out_stream} "
            f"group={self.group} consumer={self.consumer} drain_pending={self.drain_pending} "
            f"window={self.window_sec}s topk={self.topk} cooldown={self.cooldown_sec}s "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

        # optional: drain PEL first
        if self.drain_pending:
            print("[TopSelector] draining pending (PEL) ...")
            while True:
                rows = self._xreadgroup("0")
                if not rows:
                    break

                # hepsini “ok” kabul edip publish/ack yapacağız
                # PEL’deki aşırı eski kayıtlar window’da elenebilir.
                top = self._select_topk(rows)
                if top:
                    out_id = self._publish(top)
                    if out_id:
                        syms = ", ".join([f"{x.get('symbol')}:{x.get('side')}" for x in top])
                        print(f"[TopSelector] (PEL) published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

                # pending okundu -> ack hepsini
                self._ack([sid for sid, _ in rows])

                time.sleep(0.05)

            print("[TopSelector] pending drained.")

        # main loop
        while True:
            rows = self._xreadgroup(">")
            if not rows:
                continue

            # publish topk from current batch
            top = self._select_topk(rows)
            out_id = self._publish(top) if top else None

            if out_id:
                syms = ", ".join([f"{x.get('symbol')}:{x.get('side')}" for x in top])
                print(f"[TopSelector] published {len(top)} -> {self.out_stream} id={out_id} | {syms}")

            # ack everything we consumed (fail-open)
            self._ack([sid for sid, _ in rows])

            # pacing
            time.sleep(0.05)


if __name__ == "__main__":
    TopSelector().run_forever()
