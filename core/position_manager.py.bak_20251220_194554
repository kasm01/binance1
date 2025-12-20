"""
core.position_manager

- Açık pozisyon state'ini Redis üzerinde tutar
- İsteğe bağlı olarak Postgres'te 'positions' tablosuna loglar
- TradeExecutor ile entegre çalışmak üzere tasarlandı
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    import redis  # type: ignore
except ImportError:  # Redis yoksa graceful degrade
    redis = None  # type: ignore

try:
    import psycopg2  # type: ignore
except ImportError:  # Postgres client yoksa graceful degrade
    psycopg2 = None  # type: ignore


class PositionManager:
    """
    PositionManager:
      - Redis üzerinde symbol bazlı tek açık pozisyon tutar
      - İsteğe bağlı Postgres logging yapar
      - TradeExecutor, RiskManager vs. ile entegre kullanılmak üzere yazıldı
    """

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

        # -----------------------------------------------------
        # Redis tarafı
        # -----------------------------------------------------
        self.redis_url = redis_url
        self.redis_db = redis_db
        # prefix: örn "BTCUSDT:5m" veya "positions"
        self.redis_key_prefix = redis_key_prefix.rstrip(":")

        self.redis_client: Optional["redis.Redis"] = None

        if redis is None:
            self.logger.warning(
                "[POS] redis paketi yüklü değil; in-memory pozisyon tutulacak."
            )
        else:
            try:
                self.redis_client = redis.Redis.from_url(
                    self.redis_url,
                    db=self.redis_db,
                    decode_responses=True,
                )
                # basit ping
                self.redis_client.ping()
                self.logger.info(
                    "[POS] Redis'e bağlanıldı (%s)", self.redis_url
                )
            except Exception as e:
                self.logger.error(
                    "[POS] Redis bağlantı hatası (%s): %s",
                    self.redis_url,
                    e,
                )
                self.redis_client = None

        # -----------------------------------------------------
        # Postgres tarafı
        # -----------------------------------------------------
        self.enable_pg = bool(enable_pg)
        self.pg_dsn = pg_dsn
        self.pg_conn: Optional["psycopg2.extensions.connection"] = None

        if self.enable_pg and self.pg_dsn and psycopg2 is not None:
            try:
                self.pg_conn = psycopg2.connect(self.pg_dsn)
                self.pg_conn.autocommit = True
                self._ensure_pg_schema()
                self.logger.info(
                    "[POS] Postgres'e bağlanıldı ve şema kontrol edildi."
                )
            except Exception as e:
                self.logger.error("[POS] Postgres bağlantı hatası: %s", e)
                self.enable_pg = False
        else:
            self.logger.info(
                "[POS] PG_DSN yok veya ENABLE_PG_POS_LOG=0; Postgres logging pasif."
            )

        # Eğer Redis yoksa en azından in-memory fallback
        self._local_positions: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------
    #  Dahili yardımcılar
    # ---------------------------------------------------------
    def _redis_key(self, symbol: str) -> str:
        """
        Redis key: <redis_key_prefix>:<symbol>
        Örn: "BTCUSDT:5m:BTCUSDT" değil, prefix'i mantıklı ver:
          redis_key_prefix="BTCUSDT:5m" => "BTCUSDT:5m:BTCUSDT" olur.
        Eğer genel kullanacaksan: prefix="positions" => "positions:BTCUSDT"
        """
        return f"{self.redis_key_prefix}:{symbol}"

    def _ensure_pg_schema(self) -> None:
        """
        Postgres'te 'positions' tablosu yoksa oluşturur.
        """
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
            if self.logger:
                self.logger.error("[POS] PG şema oluşturma hatası: %s", e)

    # ---------------------------------------------------------
    #  Temel CRUD: get / set / clear
    # ---------------------------------------------------------
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Belirli bir sembol için tek açık pozisyonu döndürür.
        Redis yoksa in-memory dict'ten okur.
        """
        # Önce Redis
        if self.redis_client is not None:
            try:
                key = self._redis_key(symbol)
                data = self.redis_client.get(key)
                if not data:
                    return None
                return json.loads(data)
            except Exception as e:
                self.logger.error("[POS] Redis get_position hata: %s", e)

        # Fallback: in-memory
        return self._local_positions.get(symbol)

    def set_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """
        Pozisyonu Redis'e yazar, opsiyonel olarak Postgres'e upsert eder.
        """
        # Redis
        if self.redis_client is not None:
            try:
                key = self._redis_key(symbol)
                self.redis_client.set(key, json.dumps(position))
            except Exception as e:
                self.logger.error("[POS] Redis set_position hata: %s", e)
        else:
            # fallback in-memory
            self._local_positions[symbol] = position

        # Postgres
        if self.enable_pg and self.pg_conn is not None:
            try:
                self._upsert_position_pg(position)
            except Exception as e:
                self.logger.error("[POS] PG upsert hata: %s", e)

    def clear_position(self, symbol: str) -> None:
        """
        Belirtilen sembol için açık pozisyonu siler (Redis + Postgres).
        """
        # Redis
        if self.redis_client is not None:
            try:
                key = self._redis_key(symbol)
                self.redis_client.delete(key)
            except Exception as e:
                self.logger.error("[POS] Redis clear_position hata: %s", e)
        else:
            self._local_positions.pop(symbol, None)

        # Postgres
        if self.enable_pg and self.pg_conn is not None:
            try:
                self._delete_position_pg(symbol)
            except Exception as e:
                self.logger.error("[POS] PG delete hata: %s", e)

    # ---------------------------------------------------------
    #  Listeleme / sayma
    # ---------------------------------------------------------
    def list_symbols(self) -> List[str]:
        """
        Redis'te kayıtlı tüm sembolleri döndürür.
        """
        if self.redis_client is None:
            return list(self._local_positions.keys())

        try:
            pattern = f"{self.redis_key_prefix}:*"
            keys = self.redis_client.keys(pattern)
            symbols: List[str] = []
            for k in keys:
                # k: "<prefix>:<symbol>"
                parts = k.split(":", 1)
                if len(parts) == 2:
                    symbols.append(parts[1])
                else:
                    # çok segmentli prefix varsa son parçayı sembol say
                    symbols.append(k.split(":")[-1])
            return symbols
        except Exception as e:
            self.logger.error("[POS] Redis list_symbols hata: %s", e)
            return []

    def list_positions(self) -> List[Dict[str, Any]]:
        """
        Tüm açık pozisyonları (dict listesi olarak) döndürür.
        """
        symbols = self.list_symbols()
        result: List[Dict[str, Any]] = []
        for sym in symbols:
            pos = self.get_position(sym)
            if pos:
                result.append(pos)
        return result

    def count_open_positions(self) -> int:
        """
        Açık pozisyon sayısı. RiskManager için faydalı.
        """
        return len(self.list_symbols())

    # ---------------------------------------------------------
    #  Postgres yardımcıları
    # ---------------------------------------------------------
    def _upsert_position_pg(self, pos: Dict[str, Any]) -> None:
        """
        positions tablosuna upsert.
        """
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
        """
        positions tablosundan ilgili sembol kaydını siler.
        """
        if not self.pg_conn:
            return

        with self.pg_conn.cursor() as cur:
            cur.execute("DELETE FROM positions WHERE symbol = %s;", (symbol,))

    # ---------------------------------------------------------
    #  Temizlik
    # ---------------------------------------------------------
    def close(self) -> None:
        """
        Postgres bağlantısını kapatmak için opsiyonel çağrı.
        """
        try:
            if self.pg_conn is not None:
                self.pg_conn.close()
        except Exception as e:
            self.logger.error("[POS] PG bağlantı kapatma hatası: %s", e)
        finally:
            self.pg_conn = None

