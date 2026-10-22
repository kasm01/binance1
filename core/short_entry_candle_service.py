from __future__ import annotations

import os
import time
from typing import Any, Optional, Tuple

from core.redis_price_cache import (
    read_kline_cache,
    write_kline_cache,
)


class ShortEntryCandleService:
    """
    SHORT giriş mumlarını okur, yeniler,
    doğrular ve previous/current candle pair üretir.
    """

    def __init__(
        self,
        *,
        client: Any,
        logger: Optional[Any],
    ) -> None:
        self._client = client
        self._logger = logger
        self._cache_max_age_sec = max(0.1, float(os.getenv("SHORT_ENTRY_CANDLE_CACHE_MAX_AGE_SEC", "10") or 10))
        self._refresh_retries = max(1, int(float(os.getenv("SHORT_ENTRY_CANDLE_REFRESH_RETRIES", "3") or 3)))
        self._refresh_sleep_sec = max(0.0, float(os.getenv("SHORT_ENTRY_CANDLE_REFRESH_SLEEP_SEC", "0.35") or 0.35))
        self._fetch_limit = max(20, int(float(os.getenv("SHORT_ENTRY_CANDLE_FETCH_LIMIT", "20") or 20)))
        self._cache_ttl_sec = max(1, int(float(os.getenv("SHORT_ENTRY_CANDLE_CACHE_TTL_SEC", "90") or 90)))
        # Exact rows used by the latest confirmation check. Keeping them in
        # memory lets entry-time MARKET CONTEXT reuse the same snapshot without
        # adding a second Binance request.
        self._last_rows = {}

    def _read_rows(
        self,
        symbol: str,
        interval: str,
    ):
        try:
            return read_kline_cache(
                symbol,
                interval,
                max_age_sec=float(self._cache_max_age_sec),
            )
        except Exception:
            return None

    def _open_time_ms(
        self,
        row: Any,
    ) -> Optional[int]:
        try:
            if isinstance(row, (list, tuple)):
                return int(float(row[0]))

            if isinstance(row, dict):
                raw_ts = (
                    row.get("open_time")
                    or row.get("openTime")
                    or row.get("timestamp")
                    or row.get("time")
                    or row.get("t")
                )

                if raw_ts is None:
                    return None

                ts = int(float(raw_ts))

                # Saniye geldiyse milisaniyeye çevir.
                if ts < 10_000_000_000:
                    ts *= 1000

                return ts

        except Exception:
            return None

        return None

    def _extract_open_close(
        self,
        row: Any,
    ) -> Optional[Tuple[float, float]]:
        try:
            if isinstance(row, (list, tuple)):
                open_px = float(row[1])
                close_px = float(row[4])

            elif isinstance(row, dict):
                open_px = float(
                    row.get("open")
                    or row.get("o")
                    or 0.0
                )

                close_px = float(
                    row.get("close")
                    or row.get("c")
                    or 0.0
                )

            else:
                return None

            if open_px <= 0.0 or close_px <= 0.0:
                return None

            return float(open_px), float(close_px)

        except Exception:
            return None

    def _refresh_1m_rows(
        self,
        *,
        symbol: str,
        interval: str,
        rows: Any,
        cache_has_current_candle: bool,
        last_open_ms: Optional[int],
        expected_current_open_ms: int,
    ):
        if (
            interval != "1m"
            or cache_has_current_candle
        ):
            return rows

        try:
            if self._logger:
                self._logger.info(
                    "[EXEC-SHORT][1M-FORCE-REFRESH] "
                    "symbol=%s cached_open_ms=%s "
                    "expected_open_ms=%s",
                    symbol,
                    last_open_ms,
                    expected_current_open_ms,
                )
        except Exception:
            pass

        # Güncel mumu Redis üzerinden tekrar okumak yerine
        # Binance cevabını doğrudan kullan.
        try:
            direct_rows = None

            for attempt in range(1, self._refresh_retries + 1):
                direct_rows = (
                    self._client.futures_klines(
                        symbol=symbol,
                        interval="1m",
                        limit=int(self._fetch_limit),
                    )
                )

                direct_last_open_ms = None

                if (
                    isinstance(direct_rows, list)
                    and len(direct_rows) >= 2
                ):
                    direct_last_open_ms = (
                        self._open_time_ms(
                            direct_rows[-1]
                        )
                    )

                try:
                    if self._logger:
                        self._logger.info(
                            "[EXEC-SHORT][1M-DIRECT-TRY] "
                            "symbol=%s attempt=%s "
                            "last_open_ms=%s "
                            "expected_open_ms=%s",
                            symbol,
                            attempt,
                            direct_last_open_ms,
                            expected_current_open_ms,
                        )
                except Exception:
                    pass

                if (
                    direct_last_open_ms
                    == expected_current_open_ms
                ):
                    break

                if attempt < self._refresh_retries:
                    time.sleep(float(self._refresh_sleep_sec))

            if (
                isinstance(direct_rows, list)
                and len(direct_rows) >= 2
            ):
                rows = direct_rows

                try:
                    write_kline_cache(
                        symbol,
                        "1m",
                        direct_rows,
                        int(self._cache_ttl_sec),
                    )
                except Exception:
                    pass

                try:
                    if self._logger:
                        direct_last_open_ms = (
                            self._open_time_ms(
                                direct_rows[-1]
                            )
                        )

                        self._logger.info(
                            "[EXEC-SHORT][1M-DIRECT] "
                            "symbol=%s last_open_ms=%s "
                            "expected_open_ms=%s",
                            symbol,
                            direct_last_open_ms,
                            expected_current_open_ms,
                        )
                except Exception:
                    pass

            else:
                rows = self._read_rows(
                    symbol,
                    interval,
                )

        except Exception as exc:
            try:
                if self._logger:
                    self._logger.warning(
                        "[EXEC-SHORT][1M-DIRECT][WARN] "
                        "symbol=%s err=%s",
                        symbol,
                        str(exc)[:200],
                    )
            except Exception:
                pass

            rows = self._read_rows(
                symbol,
                interval,
            )

        return rows

    def _validate_rows(
        self,
        *,
        symbol: str,
        interval: str,
        rows: Any,
        expected_current_open_ms: int,
    ) -> Optional[
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        if not isinstance(rows, list) or len(rows) < 2:
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[CANDLE-PAIR] "
                        "symbol=%s reason=missing_rows",
                        symbol,
                    )
            except Exception:
                pass

            return None

        previous_row = rows[-2]
        current_row = rows[-1]

        prev_open_ms = (
            self._open_time_ms(
                previous_row
            )
        )

        curr_open_ms = (
            self._open_time_ms(
                current_row
            )
        )

        # Yenilemeden sonra da son mum mevcut
        # dakikaya ait değilse engelle.
        if (
            interval == "1m"
            and curr_open_ms
            != expected_current_open_ms
        ):
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[STALE-CURRENT-CANDLE] "
                        "symbol=%s curr_open_ms=%s "
                        "expected_open_ms=%s",
                        symbol,
                        curr_open_ms,
                        expected_current_open_ms,
                    )
            except Exception:
                pass

            return None

        expected_previous_open_ms = (
            expected_current_open_ms - 60_000
        )

        # Önceki mum tam olarak bir önceki
        # dakikaya ait olmalı.
        if (
            interval == "1m"
            and prev_open_ms
            != expected_previous_open_ms
        ):
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[STALE-PREVIOUS-CANDLE] "
                        "symbol=%s prev_open_ms=%s "
                        "expected_open_ms=%s",
                        symbol,
                        prev_open_ms,
                        expected_previous_open_ms,
                    )
            except Exception:
                pass

            return None

        previous = (
            self._extract_open_close(
                previous_row
            )
        )

        current = (
            self._extract_open_close(
                current_row
            )
        )

        if previous is None or current is None:
            return None

        try:
            if self._logger:
                self._logger.info(
                    "[EXEC-SHORT][CANDLE-PAIR] "
                    "symbol=%s prev_time=%s "
                    "curr_time=%s "
                    "prev_open=%.8f "
                    "prev_close=%.8f "
                    "curr_open=%.8f "
                    "curr_close=%.8f",
                    symbol,
                    prev_open_ms,
                    curr_open_ms,
                    previous[0],
                    previous[1],
                    current[0],
                    current[1],
                )
        except Exception:
            pass

        return previous, current

    def get_pair(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
        ]
    ]:
        """
        Gerçek önceki ve mevcut 1m mumunu döndürür.

        Yeni dakika başladığında cache henüz güncellenmemişse
        Binance verisini zorunlu yeniler.

        previous = tamamlanmış önceki mum
        current  = halen açık mevcut mum
        """
        sym_u = str(symbol or "").upper().strip()
        interval_s = str(interval or "1m").strip()

        if not sym_u:
            return None


        rows = self._read_rows(sym_u, interval_s)

        # Mevcut 1 dakikalık mumun beklenen açılış zamanı.
        now_ms = int(time.time() * 1000)
        expected_current_open_ms = (
            now_ms // 60_000
        ) * 60_000

        last_open_ms = None

        if isinstance(rows, list) and rows:
            last_open_ms = self._open_time_ms(rows[-1])

        cache_has_current_candle = (
            isinstance(rows, list)
            and len(rows) >= 2
            and last_open_ms is not None
            and last_open_ms == expected_current_open_ms
        )

        # Cache yeni dakikaya ait değilse mutlaka Binance'den yenile.
        rows = self._refresh_1m_rows(
            symbol=sym_u,
            interval=interval_s,
            rows=rows,
            cache_has_current_candle=(
                cache_has_current_candle
            ),
            last_open_ms=last_open_ms,
            expected_current_open_ms=(
                expected_current_open_ms
            ),
        )

        if isinstance(rows, list):
            self._last_rows[(sym_u, interval_s)] = list(rows)

        return self._validate_rows(
            symbol=sym_u,
            interval=interval_s,
            rows=rows,
            expected_current_open_ms=(
                expected_current_open_ms
            ),
        )

    def get_last_rows(self, symbol: str, interval: str = "1m"):
        """Return the exact rows last used by get_pair(), with no I/O."""
        sym_u = str(symbol or "").upper().strip()
        interval_s = str(interval or "1m").strip()
        rows = self._last_rows.get((sym_u, interval_s))
        return list(rows) if isinstance(rows, list) else None
