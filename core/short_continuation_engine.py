from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional


class ShortContinuationEngine:
    """
    SHORT continuation mumlarını ve kurallarını
    değerlendiren bağımsız setup bileşeni.
    """

    def __init__(
        self,
        *,
        client: Any,
        logger: Optional[Any],
        env_float: Callable[[str, float], float],
    ) -> None:
        self._client = client
        self._logger = logger
        self._env_float = env_float

    def _prepare_candles(
        self,
        *,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        SHORT continuation için 1m mumlarını çeker,
        güncelliğini kontrol eder ve parse eder.
        """
        sym_u = str(symbol or "").upper().strip()

        try:
            rows = self._client.futures_klines(
                symbol=sym_u,
                interval="1m",
                limit=5,
            )
        except Exception as exc:
            try:
                if self._logger:
                    self._logger.warning(
                        "[EXEC-SHORT][CONTINUATION][WARN] "
                        "symbol=%s err=%s",
                        sym_u,
                        str(exc)[:200],
                    )
            except Exception:
                pass

            return {
                "ok": False,
                "reason": "kline_exception",
            }

        if (
            not isinstance(rows, list)
            or len(rows) < 4
        ):
            return {
                "ok": False,
                "reason": "insufficient_klines",
            }

        try:
            now_ms = int(
                time.time() * 1000
            )

            expected_current_open_ms = (
                now_ms // 60_000
            ) * 60_000

            current_row = rows[-1]

            current_open_ms = int(
                float(current_row[0])
            )

            if (
                current_open_ms
                != expected_current_open_ms
            ):
                try:
                    if self._logger:
                        self._logger.info(
                            "[EXEC-SHORT]"
                            "[CONTINUATION-BLOCK]"
                            "[STALE-CURRENT] "
                            "symbol=%s curr_open_ms=%s "
                            "expected_open_ms=%s",
                            sym_u,
                            current_open_ms,
                            expected_current_open_ms,
                        )
                except Exception:
                    pass

                return {
                    "ok": False,
                    "reason": "stale_current",
                }

            completed = []

            for row in rows[-4:-1]:
                completed.append(
                    {
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )

            current = {
                "open": float(current_row[1]),
                "high": float(current_row[2]),
                "low": float(current_row[3]),
                "close": float(current_row[4]),
            }

        except Exception:
            return {
                "ok": False,
                "reason": "kline_parse_failed",
            }

        invalid_completed = any(
            candle["open"] <= 0.0
            or candle["close"] <= 0.0
            or candle["high"] <= 0.0
            or candle["low"] <= 0.0
            for candle in completed
        )

        if invalid_completed:
            return {
                "ok": False,
                "reason": "invalid_completed_candle",
            }

        if (
            current["open"] <= 0.0
            or current["close"] <= 0.0
        ):
            return {
                "ok": False,
                "reason": "invalid_current_candle",
            }

        return {
            "ok": True,
            "completed": completed,
            "current": current,
        }

    def _get_config(
        self,
    ) -> Dict[str, Any]:
        """
        SHORT continuation eşiklerini döndürür.
        """
        return {
            "min_red_count": int(
                self._env_float(
                    "SHORT_CONTINUATION_MIN_RED_COUNT",
                    2,
                )
            ),
            "min_trend_drop_pct": self._env_float(
                "SHORT_CONTINUATION_MIN_TREND_DROP_PCT",
                0.0008,
            ),
            "min_trigger_drop_pct": self._env_float(
                "SHORT_CONTINUATION_MIN_TRIGGER_DROP_PCT",
                0.0001,
            ),
            "max_current_rise_pct": self._env_float(
                "SHORT_CONTINUATION_MAX_CURRENT_RISE_PCT",
                0.0003,
            ),
            "max_last_body_pct": self._env_float(
                "SHORT_CONTINUATION_MAX_LAST_BODY_PCT",
                0.0030,
            ),
            "break_buffer_pct": self._env_float(
                "SHORT_CONTINUATION_BREAK_BUFFER_PCT",
                0.0000,
            ),
        }

    def _evaluate_rules(
        self,
        *,
        symbol: str,
        completed: list,
        current: Dict[str, Any],
        config: Dict[str, Any],
    ) -> bool:
        """
        SHORT continuation mumlarını ve eşiklerini
        değerlendirerek giriş kararını döndürür.
        """
        sym_u = str(symbol or "").upper().strip()

        min_red_count = config["min_red_count"]
        min_trend_drop_pct = config[
            "min_trend_drop_pct"
        ]
        min_trigger_drop_pct = config[
            "min_trigger_drop_pct"
        ]
        max_current_rise_pct = config[
            "max_current_rise_pct"
        ]
        max_last_body_pct = config[
            "max_last_body_pct"
        ]
        break_buffer_pct = config[
            "break_buffer_pct"
        ]

        red_count = sum(
            1
            for candle in completed
            if candle["close"] < candle["open"]
        )

        if red_count < min_red_count:
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][CONTINUATION-BLOCK]"
                        "[RED-COUNT] "
                        "symbol=%s red_count=%s min=%s",
                        sym_u,
                        red_count,
                        min_red_count,
                    )
            except Exception:
                pass

            return False

        first_open = completed[0]["open"]
        last_completed = completed[-1]
        last_close = last_completed["close"]

        trend_drop_pct = (
            (first_open - last_close)
            / first_open
        )

        if trend_drop_pct < min_trend_drop_pct:
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][CONTINUATION-BLOCK]"
                        "[TREND-WEAK] "
                        "symbol=%s trend_drop_pct=%.6f min=%.6f",
                        sym_u,
                        trend_drop_pct,
                        min_trend_drop_pct,
                    )
            except Exception:
                pass

            return False

        last_body_pct = abs(
            last_completed["close"]
            - last_completed["open"]
        ) / last_completed["open"]

        if last_body_pct > max_last_body_pct:
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][CONTINUATION-BLOCK]"
                        "[CHASE-GUARD] "
                        "symbol=%s last_body_pct=%.6f max=%.6f",
                        sym_u,
                        last_body_pct,
                        max_last_body_pct,
                    )
            except Exception:
                pass

            return False

        current_rise_pct = max(
            0.0,
            (
                current["close"]
                - current["open"]
            )
            / current["open"],
        )

        if current_rise_pct > max_current_rise_pct:
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][CONTINUATION-BLOCK]"
                        "[CURRENT-RISE] "
                        "symbol=%s rise_pct=%.6f max=%.6f",
                        sym_u,
                        current_rise_pct,
                        max_current_rise_pct,
                    )
            except Exception:
                pass

            return False

        current_drop_pct = max(
            0.0,
            (
                current["open"]
                - current["close"]
            )
            / current["open"],
        )

        previous_low = last_completed["low"]

        break_price = previous_low * (
            1.0 - break_buffer_pct
        )

        broke_previous_low = (
            current["close"] <= break_price
        )

        current_down_trigger = (
            current_drop_pct
            >= min_trigger_drop_pct
        )

        # İki tetikten biri yeterlidir:
        # - mevcut mum belirgin şekilde aşağı dönmüş,
        # - önceki tamamlanmış mumun dibi kırılmış.
        if not (
            current_down_trigger
            or broke_previous_low
        ):
            try:
                if self._logger:
                    self._logger.info(
                        "[EXEC-SHORT][CONTINUATION-BLOCK]"
                        "[NO-TRIGGER] "
                        "symbol=%s current_drop_pct=%.6f "
                        "min_drop=%.6f curr_close=%.8f "
                        "break_price=%.8f",
                        sym_u,
                        current_drop_pct,
                        min_trigger_drop_pct,
                        current["close"],
                        break_price,
                    )
            except Exception:
                pass

            return False

        try:
            if self._logger:
                self._logger.info(
                    "[EXEC-SHORT][CONTINUATION-CONFIRM] "
                    "symbol=%s red_count=%s "
                    "trend_drop_pct=%.6f "
                    "last_body_pct=%.6f "
                    "current_drop_pct=%.6f "
                    "broke_previous_low=%s allow=1",
                    sym_u,
                    red_count,
                    trend_drop_pct,
                    last_body_pct,
                    current_drop_pct,
                    int(broke_previous_low),
                )
        except Exception:
            pass

        return True

    def check(
        self,
        symbol: str,
    ) -> bool:
        """
        Düşüş trendinin devamında SHORT girişini doğrular.

        Kurallar:
        1. Son 3 tamamlanmış mumun en az 2'si kırmızı olmalı.
        2. Üç mumluk toplam hareket aşağı yönlü olmalı.
        3. Son tamamlanmış mum aşırı büyük olmamalı.
        4. Mevcut mum güçlü biçimde yukarı kaçmamalı.
        5. Mevcut fiyat aşağı yönlü tetik üretmeli.
        """
        enabled = str(
            os.getenv(
                "SHORT_CONTINUATION_ENABLE",
                "1",
            )
            or "1"
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        if not enabled:
            return False

        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return False

        candle_context = (
            self._prepare_candles(
                symbol=sym_u,
            )
        )

        if not candle_context.get("ok", False):
            return False

        parsed = candle_context["completed"]
        current = candle_context["current"]

        config = (
            self._get_config()
        )

        min_red_count = config["min_red_count"]
        min_trend_drop_pct = config[
            "min_trend_drop_pct"
        ]
        min_trigger_drop_pct = config[
            "min_trigger_drop_pct"
        ]
        max_current_rise_pct = config[
            "max_current_rise_pct"
        ]
        max_last_body_pct = config[
            "max_last_body_pct"
        ]
        break_buffer_pct = config[
            "break_buffer_pct"
        ]

        return (
            self._evaluate_rules(
                symbol=sym_u,
                completed=parsed,
                current=current,
                config=config,
            )
        )
