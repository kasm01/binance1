# core/position_manager.py
"""
core.position_manager

- Açık pozisyon state'ini Redis üzerinde tutar
- İsteğe bağlı olarak Postgres'te 'positions' tablosuna loglar
- TradeExecutor ile entegre çalışmak üzere tasarlandı

Tek tip shutdown kontratı:
  - close()    (idempotent)
  - shutdown() (idempotent wrapper, reason alır)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

try:
    import psycopg2  # type: ignore
except ImportError:
    psycopg2 = None  # type: ignore


class PositionManager:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        redis_db: int = 0,
        redis_key_prefix: str = "positions",
        enable_pg: bool = False,
        pg_dsn: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("PositionManager")

        self.redis_url = redis_url
        self.redis_db = int(redis_db)
        self.redis_key_prefix = str(redis_key_prefix).rstrip(":")

        self.redis_client: Optional["redis.Redis"] = None
        self.enable_pg = bool(enable_pg)
        self.pg_dsn = pg_dsn
        self.pg_conn: Optional["psycopg2.extensions.connection"] = None

        self._closed = False
        self._local_positions: Dict[str, Dict[str, Any]] = {}

        # Redis init
        if redis is None:
            self.logger.warning("[POS] redis paketi yok; in-memory fallback kullanılacak.")
        else:
            try:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    decode_responses=True,
                )
                self.redis_client.ping()
                self.logger.info("[POS] Redis OK (%s)", self.redis_url)
            except Exception as e:
                self.logger.error("[POS] Redis bağlantı hatası (%s): %s", self.redis_url, e)
                self.redis_client = None

        # Postgres init
        if self.enable_pg and self.pg_dsn and psycopg2 is not None:
            try:
                self.pg_conn = psycopg2.connect(self.pg_dsn)
                self.pg_conn.autocommit = True
                self._ensure_pg_schema()
                self.logger.info("[POS] Postgres OK + schema checked.")
            except Exception as e:
                self.logger.error("[POS] Postgres bağlantı hatası: %s", e)
                self.enable_pg = False
                self.pg_conn = None
        else:
            self.logger.info("[POS] Postgres logging pasif (ENABLE_PG_POS_LOG=0 veya PG_DSN yok).")

    def _redis_key(self, symbol: str) -> str:
        return f"{self.redis_key_prefix}:{str(symbol).upper()}"

    def _ensure_pg_schema(self) -> None:
        if not self.pg_conn:
            return
        try:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        side TEXT,
                        qty DOUBLE PRECISION,
                        entry_price DOUBLE PRECISION,
                        notional DOUBLE PRECISION,
                        interval TEXT,
                        opened_at TIMESTAMPTZ,
                        sl_price DOUBLE PRECISION,
                        tp_price DOUBLE PRECISION,
                        trailing_pct DOUBLE PRECISION,
                        atr_value DOUBLE PRECISION,
                        highest_price DOUBLE PRECISION,
                        lowest_price DOUBLE PRECISION,
                        raw_json JSONB
                    );
                    """
                )
            self.pg_conn.commit()
        except Exception as e:
            self.logger.error("[POS] PG şema oluşturma hatası: %s", e)

    # -------------------------
    # CRUD
    # -------------------------
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()

        if self.redis_client is not None:
            try:
                key = self._redis_key(sym)
                data = self.redis_client.get(key)
                if not data:
                    return None
                return json.loads(data)
            except Exception as e:
                self.logger.error("[POS] Redis get_position hata: %s", e)

        return self._local_positions.get(sym)

    def set_position(self, symbol: str, position: Dict[str, Any]) -> None:
        sym = str(symbol).upper()

        if self.redis_client is not None:
            try:
                key = self._redis_key(sym)
                self.redis_client.set(key, json.dumps(position))
            except Exception as e:
                self.logger.error("[POS] Redis set_position hata: %s", e)
        else:
            self._local_positions[sym] = position

        if self.enable_pg and self.pg_conn is not None:
            try:
                self._upsert_position_pg(position)
            except Exception as e:
                self.logger.error("[POS] PG upsert hata: %s", e)

    def clear_position(self, symbol: str) -> None:
        sym = str(symbol).upper()

        if self.redis_client is not None:
            try:
                key = self._redis_key(sym)
                self.redis_client.delete(key)
            except Exception as e:
                self.logger.error("[POS] Redis clear_position hata: %s", e)
        else:
            self._local_positions.pop(sym, None)

        if self.enable_pg and self.pg_conn is not None:
            try:
                self._delete_position_pg(sym)
            except Exception as e:
                self.logger.error("[POS] PG delete hata: %s", e)

    # -------------------------
    # Backtest compat API (NEW)
    # -------------------------
    def has_open_position(self, symbol: str) -> bool:
        pos = self.get_position(symbol)
        if not pos:
            return False
        side = str(pos.get("side") or "").lower()
        qty = float(pos.get("qty") or 0.0)
        return (side in ("long", "short")) and qty > 0

    def close_position(self, symbol: str, price: float = 0.0, reason: str = "manual") -> Optional[Dict[str, Any]]:
        """
        PositionManager gerçek emir kapatmaz; sadece state'i temizler.
        Backtest/utility için: pozisyon dict'ine kapama alanları ekleyip clear eder.
        """
        pos = self.get_position(symbol)
        if not pos:
            return None

        try:
            pos["closed_at"] = datetime.utcnow().isoformat()
            pos["close_price"] = float(price)
            pos["close_reason"] = str(reason)
        except Exception:
            pass

        self.clear_position(symbol)
        return pos

    # -------------------------
    # List / Count
    # -------------------------
    def list_symbols(self) -> List[str]:
        if self.redis_client is None:
            return list(self._local_positions.keys())

        try:
            pattern = f"{self.redis_key_prefix}:*"
            keys = self.redis_client.keys(pattern)
            symbols: List[str] = []
            for k in keys:
                parts = k.split(":", 1)
                if len(parts) == 2:
                    symbols.append(parts[1])
                else:
                    symbols.append(k.split(":")[-1])
            return symbols
        except Exception as e:
            self.logger.error("[POS] Redis list_symbols hata: %s", e)
            return []

    def list_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sym in self.list_symbols():
            pos = self.get_position(sym)
            if pos:
                out.append(pos)
        return out

    def count_open_positions(self) -> int:
        return len(self.list_symbols())

    # -------------------------
    # Postgres helpers
    # -------------------------
    def _upsert_position_pg(self, pos: Dict[str, Any]) -> None:
        if not self.pg_conn:
            return

        symbol = pos.get("symbol")
        if not symbol:
            return

        payload = {
            "symbol": symbol,
            "side": pos.get("side"),
            "qty": pos.get("qty"),
            "entry_price": pos.get("entry_price"),
            "notional": pos.get("notional"),
            "interval": pos.get("interval"),
            "opened_at": pos.get("opened_at"),
            "sl_price": pos.get("sl_price"),
            "tp_price": pos.get("tp_price"),
            "trailing_pct": pos.get("trailing_pct"),
            "atr_value": pos.get("atr_value"),
            "highest_price": pos.get("highest_price"),
            "lowest_price": pos.get("lowest_price"),
            "raw_json": json.dumps(pos),
        }

        query = """
            INSERT INTO positions (
                symbol, side, qty, entry_price, notional, interval,
                opened_at, sl_price, tp_price, trailing_pct,
                atr_value, highest_price, lowest_price, raw_json
            )
            VALUES (
                %(symbol)s, %(side)s, %(qty)s, %(entry_price)s, %(notional)s,
                %(interval)s, %(opened_at)s, %(sl_price)s, %(tp_price)s,
                %(trailing_pct)s, %(atr_value)s, %(highest_price)s,
                %(lowest_price)s, %(raw_json)s
            )
            ON CONFLICT (symbol) DO UPDATE SET
                side          = EXCLUDED.side,
                qty           = EXCLUDED.qty,
                entry_price   = EXCLUDED.entry_price,
                notional      = EXCLUDED.notional,
                interval      = EXCLUDED.interval,
                opened_at     = EXCLUDED.opened_at,
                sl_price      = EXCLUDED.sl_price,
                tp_price      = EXCLUDED.tp_price,
                trailing_pct  = EXCLUDED.trailing_pct,
                atr_value     = EXCLUDED.atr_value,
                highest_price = EXCLUDED.highest_price,
                lowest_price  = EXCLUDED.lowest_price,
                raw_json      = EXCLUDED.raw_json;
        """

        with self.pg_conn.cursor() as cur:
            cur.execute(query, payload)

    def _delete_position_pg(self, symbol: str) -> None:
        if not self.pg_conn:
            return
        with self.pg_conn.cursor() as cur:
            cur.execute("DELETE FROM positions WHERE symbol = %s;", (symbol,))

    # ---------------------------------------------------------
    #  Shutdown contract (deterministic)
    # ---------------------------------------------------------
    def shutdown(self, reason: str = "unknown") -> None:
        try:
            if getattr(self, "logger", None):
                self.logger.info("[POS] shutdown requested | reason=%s", reason)
        except Exception:
            pass
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            if self.pg_conn is not None:
                try:
                    self.pg_conn.close()
                except Exception as e:
                    self.logger.error("[POS] PG close hata: %s", e)
        finally:
            self.pg_conn = None

        try:
            if self.redis_client is not None:
                rc = self.redis_client
                close_fn = getattr(rc, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception:
            pass
        finally:
            self.redis_client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
