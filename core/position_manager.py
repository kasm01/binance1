# core/position_manager.py

import csv
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import psycopg2  # type: ignore
except ImportError:
    psycopg2 = None  # type: ignore


class PositionManager:
    """
    Pozisyon yöneticisi:

    - Açık pozisyonları memory + opsiyonel Redis'te tutar
    - İstenirse Postgres'te `open_positions` tablosuna kaydeder
    - Kapanan pozisyonları trade journal CSV'ye loglar
    - SL/TP + trailing stop mantığı için helper sağlar
    """

    def __init__(
        self,
        redis_client: Any = None,
        redis_prefix: str = "bot:position:",
        journal_csv_path: str = "data/trade_journal.csv",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("system")
        self.redis = redis_client
        self.redis_prefix = redis_prefix
        self.journal_csv_path = journal_csv_path

        # Memory cache: symbol -> position dict
        self._positions: Dict[str, Dict[str, Any]] = {}

        # Postgres DSN (örn: "postgresql://user:pass@host:5432/dbname")
        self.pg_dsn: Optional[str] = os.getenv("PG_DSN") or None
        if self.pg_dsn and psycopg2 is None:
            self.logger.warning(
                "[PM] PG_DSN tanımlı fakat psycopg2 yok. Postgres logging pasif."
            )
            self.pg_dsn = None

        # CSV klasörü yoksa oluştur
        try:
            os.makedirs(os.path.dirname(self.journal_csv_path), exist_ok=True)
        except Exception:
            # data/ gibi klasör yoksa ilk journal yazımında tekrar deneriz
            pass

        if self.pg_dsn:
            self._ensure_pg_schema()

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _utcnow_iso(self) -> str:
        return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    def _redis_key(self, symbol: str) -> str:
        return f"{self.redis_prefix}{symbol}"

    # ------------------------------------------------------------------
    #  Redis sync
    # ------------------------------------------------------------------
    def _load_from_redis(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            raw = self.redis.get(self._redis_key(symbol))
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            pos = json.loads(raw)
            self._positions[symbol] = pos
            return pos
        except Exception as exc:
            self.logger.exception(
                "[PM] Redis'ten pozisyon okunamadı (%s): %s", symbol, exc
            )
            return None

    def _save_to_redis(self, symbol: str, pos: Dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            self.redis.set(self._redis_key(symbol), json.dumps(pos))
        except Exception as exc:
            self.logger.exception(
                "[PM] Redis'e pozisyon yazılamadı (%s): %s", symbol, exc
            )

    def _delete_from_redis(self, symbol: str) -> None:
        if not self.redis:
            return
        try:
            self.redis.delete(self._redis_key(symbol))
        except Exception as exc:
            self.logger.exception(
                "[PM] Redis'ten pozisyon silinemedi (%s): %s", symbol, exc
            )

    # ------------------------------------------------------------------
    #  Postgres
    # ------------------------------------------------------------------
    def _ensure_pg_schema(self) -> None:
        """
        open_positions tablosunu basitçe oluşturur (yoksa).
        """
        if not self.pg_dsn or psycopg2 is None:
            return

        sql = """
        CREATE TABLE IF NOT EXISTS open_positions (
            symbol TEXT PRIMARY KEY,
            side TEXT NOT NULL,
            qty DOUBLE PRECISION NOT NULL,
            entry_price DOUBLE PRECISION NOT NULL,
            notional DOUBLE PRECISION NOT NULL,
            interval TEXT,
            opened_at TIMESTAMPTZ,
            sl_price DOUBLE PRECISION,
            tp_price DOUBLE PRECISION,
            trailing_pct DOUBLE PRECISION,
            atr_value DOUBLE PRECISION,
            highest_price DOUBLE PRECISION,
            lowest_price DOUBLE PRECISION,
            meta_json JSONB
        );
        """
        try:
            conn = psycopg2.connect(self.pg_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.close()
            self.logger.info(
                "[PM] Postgres open_positions şeması kontrol edildi/oluşturuldu."
            )
        except Exception as exc:
            self.logger.exception("[PM] Postgres şema oluşturma hatası: %s", exc)
            # Tek sefer uyar, DSN'i devre dışı bırak
            self.pg_dsn = None

    def _pg_upsert_open_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        if not self.pg_dsn or psycopg2 is None:
            return

        sql = """
        INSERT INTO open_positions (
            symbol, side, qty, entry_price, notional, interval,
            opened_at, sl_price, tp_price, trailing_pct, atr_value,
            highest_price, lowest_price, meta_json
        ) VALUES (
            %(symbol)s, %(side)s, %(qty)s, %(entry_price)s, %(notional)s, %(interval)s,
            %(opened_at_ts)s, %(sl_price)s, %(tp_price)s, %(trailing_pct)s, %(atr_value)s,
            %(highest_price)s, %(lowest_price)s, %(meta_json)s
        )
        ON CONFLICT (symbol) DO UPDATE SET
            side = EXCLUDED.side,
            qty = EXCLUDED.qty,
            entry_price = EXCLUDED.entry_price,
            notional = EXCLUDED.notional,
            interval = EXCLUDED.interval,
            opened_at = EXCLUDED.opened_at,
            sl_price = EXCLUDED.sl_price,
            tp_price = EXCLUDED.tp_price,
            trailing_pct = EXCLUDED.trailing_pct,
            atr_value = EXCLUDED.atr_value,
            highest_price = EXCLUDED.highest_price,
            lowest_price = EXCLUDED.lowest_price,
            meta_json = EXCLUDED.meta_json;
        """
        try:
            opened_at_ts = None
            opened_at_str = pos.get("opened_at")
            if opened_at_str:
                try:
                    opened_at_ts = datetime.fromisoformat(opened_at_str)
                except Exception:
                    opened_at_ts = datetime.utcnow()

            params = {
                "symbol": pos.get("symbol"),
                "side": pos.get("side"),
                "qty": float(pos.get("qty", 0.0)),
                "entry_price": float(pos.get("entry_price", 0.0)),
                "notional": float(pos.get("notional", 0.0)),
                "interval": pos.get("interval"),
                "opened_at_ts": opened_at_ts,
                "sl_price": float(pos.get("sl_price"))
                if pos.get("sl_price") is not None
                else None,
                "tp_price": float(pos.get("tp_price"))
                if pos.get("tp_price") is not None
                else None,
                "trailing_pct": float(pos.get("trailing_pct"))
                if pos.get("trailing_pct") is not None
                else None,
                "atr_value": float(pos.get("atr_value"))
                if pos.get("atr_value") is not None
                else None,
                "highest_price": float(pos.get("highest_price"))
                if pos.get("highest_price") is not None
                else None,
                "lowest_price": float(pos.get("lowest_price"))
                if pos.get("lowest_price") is not None
                else None,
                "meta_json": json.dumps(pos.get("meta") or {}),
            }

            conn = psycopg2.connect(self.pg_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.close()
        except Exception as exc:
            self.logger.exception(
                "[PM] Postgres open_positions upsert hatası (%s): %s", symbol, exc
            )

    def _pg_delete_open_position(self, symbol: str) -> None:
        if not self.pg_dsn or psycopg2 is None:
            return
        sql = "DELETE FROM open_positions WHERE symbol = %s;"
        try:
            conn = psycopg2.connect(self.pg_dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql, (symbol,))
            conn.close()
        except Exception as exc:
            self.logger.exception(
                "[PM] Postgres open_positions delete hatası (%s): %s", symbol, exc
            )

    # ------------------------------------------------------------------
    #  CSV Journal
    # ------------------------------------------------------------------
    def _append_journal_row(self, row: Dict[str, Any]) -> None:
        """
        Günlük trade journal CSV'sine satır ekler.
        """
        try:
            dirname = os.path.dirname(self.journal_csv_path)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            file_exists = os.path.isfile(self.journal_csv_path)
            fieldnames = [
                "timestamp",
                "symbol",
                "side",
                "entry_price",
                "close_price",
                "qty",
                "notional",
                "pnl_abs",
                "pnl_pct",
                "interval",
                "opened_at",
                "closed_at",
                "reason",
                "extra_json",
            ]
            with open(self.journal_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            self.logger.exception("[PM] Trade journal CSV yazılamadı: %s", exc)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Sembol için açık pozisyonu döndürür (yoksa None).
        Önce memory, yoksa Redis denenir.
        """
        pos = self._positions.get(symbol)
        if pos is not None:
            return pos
        return self._load_from_redis(symbol)

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    def _sync_set_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        """
        TradeExecutor gibi üst seviye componentler doğrudan bunu çağırıyor.

        - Memory'de tut
        - Redis'e yaz
        - Postgres open_positions'a upsert
        """
        self._positions[symbol] = pos
        self._save_to_redis(symbol, pos)
        self._pg_upsert_open_position(symbol, pos)
        self.logger.info(
            "[PM] Pozisyon set edildi | symbol=%s side=%s qty=%s entry=%.4f sl=%.4f tp=%.4f",
            symbol,
            pos.get("side"),
            pos.get("qty"),
            float(pos.get("entry_price", 0.0)),
            float(pos.get("sl_price", 0.0))
            if pos.get("sl_price") is not None
            else 0.0,
            float(pos.get("tp_price", 0.0))
            if pos.get("tp_price") is not None
            else 0.0,
        )

    def _sync_clear_position(self, symbol: str) -> None:
        """
        Pozisyonu tüm storage'lardan temizler (memory, Redis, Postgres).
        Genellikle TradeExecutor içinde pozisyon kapatıldığında çağrılır.
        """
        if symbol in self._positions:
            del self._positions[symbol]
        self._delete_from_redis(symbol)
        self._pg_delete_open_position(symbol)
        self.logger.info("[PM] Pozisyon silindi | symbol=%s", symbol)
    # ------------------------------------------------------------------
    #  SL/TP & Trailing Stop mantığı için yardımcı
    # ------------------------------------------------------------------
    def evaluate_sl_tp_trailing(
        self,
        symbol: str,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Verilen sembol ve anlık fiyata göre:
          - SL/TP vurdu mu?
          - Trailing stop update gerekiyor mu?

        Dönüş:
        {
            "action": "hold" | "close_sl" | "close_tp" | "none",
            "new_sl": Optional[float],
            "new_highest": Optional[float],
            "new_lowest": Optional[float],
            "position": updated_position_dict_or_None,
        }
        """
        pos = self.get_position(symbol)
        if pos is None:
            return {"action": "none", "position": None}

        side = pos.get("side")
        sl_price = pos.get("sl_price")
        tp_price = pos.get("tp_price")
        trailing_pct = float(pos.get("trailing_pct") or 0.0)
        highest_price = float(
            pos.get("highest_price") or pos.get("entry_price") or 0.0
        )
        lowest_price = float(
            pos.get("lowest_price") or pos.get("entry_price") or 0.0
        )

        action = "hold"
        new_sl: Optional[float] = None

        # SL/TP check
        if side == "long":
            if sl_price is not None and current_price <= float(sl_price):
                action = "close_sl"
            elif tp_price is not None and current_price >= float(tp_price):
                action = "close_tp"
        elif side == "short":
            if sl_price is not None and current_price >= float(sl_price):
                action = "close_sl"
            elif tp_price is not None and current_price <= float(tp_price):
                action = "close_tp"

        # Trailing stop (sadece action hold ise güncelle)
        if action == "hold" and trailing_pct > 0.0:
            if side == "long":
                if current_price > highest_price:
                    highest_price = current_price
                trail_sl = highest_price * (1.0 - trailing_pct)
                if sl_price is None or trail_sl > float(sl_price):
                    new_sl = trail_sl
                    sl_price = trail_sl
            elif side == "short":
                if current_price < lowest_price:
                    lowest_price = current_price
                trail_sl = lowest_price * (1.0 + trailing_pct)
                if sl_price is None or trail_sl < float(sl_price):
                    new_sl = trail_sl
                    sl_price = trail_sl

        # Position dict'i update et
        pos["highest_price"] = highest_price
        pos["lowest_price"] = lowest_price
        pos["sl_price"] = sl_price

        # Sync updated position
        self._sync_set_position(symbol, pos)

        return {
            "action": action,
            "new_sl": new_sl,
            "new_highest": highest_price,
            "new_lowest": lowest_price,
            "position": pos,
        }

    # ------------------------------------------------------------------
    #  Pozisyon kapanınca journal'a yazmak için helper
    # ------------------------------------------------------------------
    def log_closed_position_to_journal(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        close_price: float,
        qty: float,
        notional: float,
        pnl_abs: float,
        pnl_pct: float,
        interval: str,
        opened_at: Optional[str],
        closed_at: Optional[str],
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        TradeExecutor, pozisyonu kapattıktan veya flip ettikten sonra çağırabilir.
        """
        row = {
            "timestamp": self._utcnow_iso(),
            "symbol": symbol,
            "side": side,
            "entry_price": float(entry_price),
            "close_price": float(close_price),
            "qty": float(qty),
            "notional": float(notional),
            "pnl_abs": float(pnl_abs),
            "pnl_pct": float(pnl_pct),
            "interval": interval,
            "opened_at": opened_at or "",
            "closed_at": closed_at or "",
            "reason": reason,
            "extra_json": json.dumps(extra or {}),
        }
        self._append_journal_row(row)
