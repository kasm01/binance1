# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import math
from decimal import Decimal, ROUND_DOWN
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from core.redis_price_cache import read_kline_cache, write_kline_cache


class TradeExecutorShort:
    """
    Clean SHORT-only futures executor.

    Entry authority:
        SHORT signal
        -> score
        -> coin min score
        -> dynamic score
        -> duplicate block
        -> 1m sniper candle progress
        -> 1m red candle direction
        -> SELL MARKET

    Exit authority:
        lifecycle
        -> fixed ROI TP
        -> ROI hard stop
        -> BUY reduce-only close

    Intentionally excluded:
        early entry
        micro pullback
        volume spike trigger
        fake breakout
        scalp boost
        entry confirmation
        whale veto
        adaptive leverage
        scalp exit
        signal exit
        dead trade
        profit lock
        trailing
        TP runner
        anti reversal
        legacy SL/TP chain
    """

    def __init__(
        self,
        client: Any = None,
        redis_client: Any = None,
        logger: Any = None,
        price_cache: Any = None,
        position_manager: Any = None,
        risk_manager: Any = None,
        telegram_bot: Any = None,
        dry_run: bool = False,
        base_order_notional: float = 120.0,
        max_position_notional: float = 500.0,
        max_leverage: float = 3.0,
        **kwargs: Any,
    ) -> None:
        # -----------------------------------------------------
        # injected dependencies
        # -----------------------------------------------------
        self.client = client
        self.redis = redis_client
        self.logger = logger
        self.price_cache = price_cache
        self.position_manager = position_manager
        self.risk_manager = risk_manager

        # main.py currently passes tg_bot instead of telegram_bot.
        self.telegram_bot = (
            telegram_bot
            if telegram_bot is not None
            else kwargs.get("tg_bot")
        )

        self.redis_price_cache = kwargs.get("redis_price_cache")
        self.hedge_mode_enabled = True

        # Binance Futures hedge mode compatibility.

        # -----------------------------------------------------
        # runtime compatibility fields used by main.py
        # -----------------------------------------------------
        self._last_exec_stream_hb_ts = 0.0
        self._last_kline_close: Dict[str, Any] = {}

        # -----------------------------------------------------
        # base execution config
        # -----------------------------------------------------
        self.dry_run = self._env_bool_value(dry_run)

        self.base_order_notional = float(
            base_order_notional or 120.0
        )

        self.max_position_notional = float(
            max_position_notional or 500.0
        )

        self.max_leverage = max(
            1.0,
            float(max_leverage or 3.0),
        )

        self.default_order_leverage = max(
            1,
            int(
                float(
                    os.getenv(
                        "DEFAULT_ORDER_LEVERAGE",
                        str(int(self.max_leverage)),
                    )
                    or self.max_leverage
                )
            ),
        )

        # -----------------------------------------------------
        # SHORT-only side authority
        # -----------------------------------------------------
        self.side_mode = "short"

        self.side_mode_block_enable = self._env_bool(
            "SIDE_MODE_BLOCK_ENABLE",
            True,
        )

        # -----------------------------------------------------
        # score config
        # -----------------------------------------------------
        self.min_score = self._env_float(
            "MIN_SCORE",
            0.495,
        )

        self.open_min_score = self._env_float(
            "OPEN_MIN_SCORE",
            self.min_score,
        )

        self.short_threshold = self._env_float(
            "SHORT_THRESHOLD",
            self.open_min_score,
        )

        self.model_short_threshold = self._env_float(
            "MODEL_SHORT_THRESHOLD",
            self.short_threshold,
        )

        self.dynamic_coin_score_enable = self._env_bool(
            "DYNAMIC_COIN_SCORE_ENABLE",
            False,
        )

        self.dynamic_score_min = self._env_float(
            "DYNAMIC_SCORE_MIN",
            self.open_min_score,
        )

        self.dynamic_score_max = self._env_float(
            "DYNAMIC_SCORE_MAX",
            0.56,
        )

        self.dynamic_score_spread_boost = self._env_float(
            "DYNAMIC_SCORE_SPREAD_BOOST",
            0.0,
        )

        self.dynamic_score_atr_boost = self._env_float(
            "DYNAMIC_SCORE_ATR_BOOST",
            0.0,
        )

        # -----------------------------------------------------
        # sniper entry timing
        # -----------------------------------------------------
        self.sniper_entry_enable = self._env_bool(
            "SNIPER_ENTRY_ENABLE",
            True,
        )

        self.entry_interval = str(
            os.getenv("ENTRY_INTERVAL", "1m") or "1m"
        ).strip()

        self.sniper_entry_interval = str(
            os.getenv(
                "SNIPER_ENTRY_INTERVAL",
                self.entry_interval,
            )
            or self.entry_interval
        ).strip()

        self.sniper_entry_min_candle_progress = self._env_float(
            "SNIPER_ENTRY_MIN_CANDLE_PROGRESS",
            0.03,
        )

        self.sniper_entry_max_candle_progress = self._env_float(
            "SNIPER_ENTRY_MAX_CANDLE_PROGRESS",
            0.45,
        )

        # -----------------------------------------------------
        # 1m candle direction
        # -----------------------------------------------------
        self.entry_candle_dir_filter_enable = self._env_bool(
            "ENTRY_CANDLE_DIR_FILTER_ENABLE",
            True,
        )

        self.entry_candle_dir_interval = str(
            os.getenv(
                "ENTRY_CANDLE_DIR_INTERVAL",
                "1m",
            )
            or "1m"
        ).strip()

        self.entry_candle_dir_require_red_for_short = self._env_bool(
            "ENTRY_CANDLE_DIR_REQUIRE_RED_FOR_SHORT",
            True,
        )

        # -----------------------------------------------------
        # duplicate / position limits
        # -----------------------------------------------------
        self.max_open_positions = self._env_int(
            "MAX_OPEN_POSITIONS",
            3,
        )

        self.block_same_symbol_reentry = self._env_bool(
            "BLOCK_SAME_SYMBOL_REENTRY",
            True,
        )

        self.same_symbol_open_cooldown_sec = self._env_float(
            "SAME_SYMBOL_OPEN_COOLDOWN_SEC",
            300.0,
        )

        self._symbol_open_locks: Dict[str, float] = {}
        self._pending_short_entries: Dict[str, Dict[str, Any]] = {}

        # -----------------------------------------------------
        # exchange info cache
        # -----------------------------------------------------
        self.exchange_info_ttl_sec = self._env_float(
            "EXCHANGE_INFO_TTL_SEC",
            300.0,
        )

        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._exchange_info_cache_ts = 0.0

        # -----------------------------------------------------
        # local positions
        # -----------------------------------------------------
        self._positions: Dict[str, Dict[str, Any]] = {}

        # -----------------------------------------------------
        # position sync
        # -----------------------------------------------------
        self.position_sync_enabled = self._env_bool(
            "POSITION_SYNC_ENABLED",
            True,
        )

        self.position_sync_interval_sec = self._env_int(
            "POSITION_SYNC_INTERVAL_SEC",
            300,
        )

        self.position_sync_remove_orphans = self._env_bool(
            "POSITION_SYNC_REMOVE_ORPHANS",
            True,
        )

        # -----------------------------------------------------
        # lifecycle
        # -----------------------------------------------------
        self.position_lifecycle_enabled = self._env_bool(
            "POSITION_LIFECYCLE_ENABLED",
            True,
        )

        self.position_lifecycle_interval_sec = self._env_int(
            "POSITION_LIFECYCLE_INTERVAL_SEC",
            15,
        )

        # -----------------------------------------------------
        # single exit authority
        # -----------------------------------------------------
        self.fixed_roi_tp_enable = self._env_bool(
            "FIXED_ROI_TP_ENABLE",
            True,
        )

        self.fixed_roi_tp_pct = self._env_float(
            "FIXED_ROI_TP_PCT",
            0.0050,
        )

        self.fixed_roi_tp_giveback = self._env_float(
            "FIXED_ROI_TP_GIVEBACK",
            0.0,
        )

        self.roi_hard_stop_pct = self._env_float(
            "ROI_HARD_STOP_PCT",
            -0.0060,
        )

        self._log_init_summary()

    # =========================================================
    # BASIC ENV HELPERS
    # =========================================================

    @staticmethod
    def _env_bool_value(value: Any) -> bool:
        return str(value or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @classmethod
    def _env_bool(
        cls,
        name: str,
        default: bool,
    ) -> bool:
        raw = os.getenv(
            name,
            "1" if default else "0",
        )
        return cls._env_bool_value(raw)

    @staticmethod
    def _env_float(
        name: str,
        default: float,
    ) -> float:
        try:
            return float(
                os.getenv(name, str(default))
                or default
            )
        except Exception:
            return float(default)

    @staticmethod
    def _env_int(
        name: str,
        default: int,
    ) -> int:
        try:
            return int(
                float(
                    os.getenv(name, str(default))
                    or default
                )
            )
        except Exception:
            return int(default)

    @staticmethod
    def _clip_float(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return default

        if not math.isfinite(out):
            return default

        return out

    # =========================================================
    # LOGGING
    # =========================================================

    def _log_init_summary(self) -> None:
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][INIT] "
                    "side=short dry_run=%s "
                    "base_notional=%.4f max_notional=%.4f "
                    "max_leverage=%.2f default_leverage=%s "
                    "min_score=%.4f open_min=%.4f short_thr=%.4f "
                    "sniper=%s sniper_interval=%s sniper_window=%.3f..%.3f "
                    "candle_dir=%s candle_interval=%s "
                    "fixed_roi=%s tp=%.6f giveback=%.6f "
                    "hard_stop=%.6f",
                    self.dry_run,
                    self.base_order_notional,
                    self.max_position_notional,
                    self.max_leverage,
                    self.default_order_leverage,
                    self.min_score,
                    self.open_min_score,
                    self.short_threshold,
                    self.sniper_entry_enable,
                    self.sniper_entry_interval,
                    self.sniper_entry_min_candle_progress,
                    self.sniper_entry_max_candle_progress,
                    self.entry_candle_dir_filter_enable,
                    self.entry_candle_dir_interval,
                    self.fixed_roi_tp_enable,
                    self.fixed_roi_tp_pct,
                    self.fixed_roi_tp_giveback,
                    self.roi_hard_stop_pct,
                )
        except Exception:
            pass

    # =========================================================
    # POSITION STORE
    # =========================================================

    def _get_position(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        sym_u = str(symbol or "").upper().strip()

        pos = self._positions.get(sym_u)

        if not isinstance(pos, dict):
            return None

        return dict(pos)

    def _set_position(
        self,
        symbol: str,
        pos: Dict[str, Any],
    ) -> None:
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return

        self._positions[sym_u] = dict(pos)

    def _del_position(
        self,
        symbol: str,
    ) -> None:
        sym_u = str(symbol or "").upper().strip()

        self._positions.pop(sym_u, None)

    def _get_all_local_positions(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            str(symbol): dict(pos)
            for symbol, pos in self._positions.items()
            if isinstance(pos, dict)
        }

    # =========================================================
    # SHORT SCORE HELPERS
    # =========================================================

    def _get_coin_side_min_open_score(
        self,
        symbol: str,
        side: str,
        default_score: float,
    ) -> float:
        sym_u = str(symbol or "").upper().strip()
        side_u = str(side or "").lower().strip()

        if side_u != "short":
            return float(default_score)

        if not self._env_bool(
            "COIN_SIDE_MIN_SCORE_ENABLE",
            False,
        ):
            return float(default_score)

        key = f"COIN_SHORT_MIN_SCORE_{sym_u}"
        raw = os.getenv(key, "")

        if not str(raw or "").strip():
            return float(default_score)

        try:
            return float(raw)
        except Exception:
            return float(default_score)

    def _resolve_signal_score_for_open(
        self,
        side: str,
        extra: Optional[Dict[str, Any]] = None,
        probs: Optional[Dict[str, Any]] = None,
    ) -> float:
        extra0 = extra if isinstance(extra, dict) else {}
        probs0 = probs if isinstance(probs, dict) else {}
        side_u = str(side or "").lower().strip()

        def _f(value: Any) -> float:
            try:
                return float(value)
            except Exception:
                return 0.0

        raw0 = extra0.get("raw") or {}

        if isinstance(raw0, str):
            try:
                raw0 = json.loads(raw0)
            except Exception:
                raw0 = {}

        if not isinstance(raw0, dict):
            raw0 = {}

        score = max(
            _f(extra0.get("signal_score")),
            _f(extra0.get("score")),
            _f(extra0.get("_score_total_final")),
            _f(extra0.get("_score_selected")),
            _f(extra0.get("score_total")),
            _f(raw0.get("signal_score")),
            _f(raw0.get("score")),
            _f(raw0.get("_score_total_final")),
            _f(raw0.get("_score_selected")),
            _f(raw0.get("score_total")),
        )

        if score > 0.0:
            return float(score)

        p_buy = max(
            _f(extra0.get("p_buy_ema")),
            _f(extra0.get("p_buy_raw")),
            _f(probs0.get("p_buy")),
        )

        if 0.0 < p_buy <= 1.0:
            if side_u == "short":
                return float(1.0 - p_buy)

            if side_u == "long":
                return float(p_buy)

        mcf = _f(
            extra0.get("model_confidence_factor")
        )

        return float(mcf) if mcf > 0.0 else 0.0

    def _resolve_dynamic_open_score(
        self,
        base_score: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        score = float(base_score)

        if not self.dynamic_coin_score_enable:
            return score

        extra0 = extra if isinstance(extra, dict) else {}

        atr_v = self._clip_float(
            extra0.get("atr")
            or extra0.get("atr_pct"),
            0.0,
        ) or 0.0

        spread_v = self._clip_float(
            extra0.get("spread_pct")
            or extra0.get("spread"),
            0.0,
        ) or 0.0

        if atr_v > 0.0:
            score += self.dynamic_score_atr_boost

        if spread_v > 0.0:
            score += self.dynamic_score_spread_boost

        return max(
            self.dynamic_score_min,
            min(self.dynamic_score_max, score),
        )

    # =========================================================
    def _refresh_entry_1m_cache(self, symbol):
        try:
            rows = self.client.futures_klines(symbol=symbol, interval="1m", limit=3)
            if rows:
                write_kline_cache(symbol, "1m", rows, 90)
                return True
        except Exception as exc:
            if self.logger:
                self.logger.warning("[EXEC-SHORT][1M-REFRESH][WARN] symbol=%s err=%s", symbol, str(exc)[:200])
        return False


    # FAST ENTRY KLINE
    # =========================================================

    def _get_entry_kline(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[Tuple[float, float]]:
        sym_u = str(symbol or "").upper().strip()
        interval_s = str(interval or "1m").strip()

        if not sym_u:
            return None

        try:
            rows = read_kline_cache(
                sym_u,
                interval_s,
                max_age_sec=10.0,
            )
        except Exception:
            rows = None

        # Cache yoksa veya eskiyse, 1m veriyi Binance'ten
        # anında yenile ve bir kez daha oku.
        if (
            (not isinstance(rows, list) or not rows)
            and interval_s == "1m"
        ):
            refreshed = self._refresh_entry_1m_cache(sym_u)

            if refreshed:
                try:
                    rows = read_kline_cache(
                        sym_u,
                        interval_s,
                        max_age_sec=10.0,
                    )
                except Exception:
                    rows = None

                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC-SHORT][1M-REFRESH] "
                            "symbol=%s success=%s",
                            sym_u,
                            bool(
                                isinstance(rows, list)
                                and rows
                            ),
                        )
                except Exception:
                    pass

        if not isinstance(rows, list) or not rows:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK][KLINE-CACHE] "
                        "symbol=%s interval=%s reason=missing_after_refresh",
                        sym_u,
                        interval_s,
                    )
            except Exception:
                pass

            return None

        row = rows[-1]

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

        except Exception:
            return None

        if open_px <= 0.0 or close_px <= 0.0:
            return None

        return float(open_px), float(close_px)

    def _read_entry_kline_rows(
        self,
        symbol: str,
        interval: str,
    ):
        try:
            return read_kline_cache(
                symbol,
                interval,
                max_age_sec=10.0,
            )
        except Exception:
            return None

    def _entry_candle_open_time_ms(
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

    def _extract_entry_open_close(
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

    def _refresh_entry_1m_rows(
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
            if self.logger:
                self.logger.info(
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

            for attempt in range(1, 4):
                direct_rows = (
                    self.client.futures_klines(
                        symbol=symbol,
                        interval="1m",
                        limit=3,
                    )
                )

                direct_last_open_ms = None

                if (
                    isinstance(direct_rows, list)
                    and len(direct_rows) >= 2
                ):
                    direct_last_open_ms = (
                        self._entry_candle_open_time_ms(
                            direct_rows[-1]
                        )
                    )

                try:
                    if self.logger:
                        self.logger.info(
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

                if attempt < 3:
                    time.sleep(0.35)

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
                        90,
                    )
                except Exception:
                    pass

                try:
                    if self.logger:
                        direct_last_open_ms = (
                            self._entry_candle_open_time_ms(
                                direct_rows[-1]
                            )
                        )

                        self.logger.info(
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
                rows = self._read_entry_kline_rows(
                    symbol,
                    interval,
                )

        except Exception as exc:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][1M-DIRECT][WARN] "
                        "symbol=%s err=%s",
                        symbol,
                        str(exc)[:200],
                    )
            except Exception:
                pass

            rows = self._read_entry_kline_rows(
                symbol,
                interval,
            )

        return rows

    def _validate_entry_candle_rows(
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
                if self.logger:
                    self.logger.info(
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
            self._entry_candle_open_time_ms(
                previous_row
            )
        )

        curr_open_ms = (
            self._entry_candle_open_time_ms(
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
                if self.logger:
                    self.logger.info(
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
                if self.logger:
                    self.logger.info(
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
            self._extract_entry_open_close(
                previous_row
            )
        )

        current = (
            self._extract_entry_open_close(
                current_row
            )
        )

        if previous is None or current is None:
            return None

        try:
            if self.logger:
                self.logger.info(
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

    def _get_entry_candle_pair(
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


        rows = self._read_entry_kline_rows(sym_u, interval_s)

        # Mevcut 1 dakikalık mumun beklenen açılış zamanı.
        now_ms = int(time.time() * 1000)
        expected_current_open_ms = (
            now_ms // 60_000
        ) * 60_000

        last_open_ms = None

        if isinstance(rows, list) and rows:
            last_open_ms = self._entry_candle_open_time_ms(rows[-1])

        cache_has_current_candle = (
            isinstance(rows, list)
            and len(rows) >= 2
            and last_open_ms is not None
            and last_open_ms == expected_current_open_ms
        )

        # Cache yeni dakikaya ait değilse mutlaka Binance'den yenile.
        rows = self._refresh_entry_1m_rows(
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

        return self._validate_entry_candle_rows(
            symbol=sym_u,
            interval=interval_s,
            rows=rows,
            expected_current_open_ms=(
                expected_current_open_ms
            ),
        )

    # =========================================================
    # SNIPER ENTRY
    # =========================================================

    def _calc_candle_progress(
        self,
        interval: str,
    ) -> float:
        interval_sec = self._interval_to_seconds(
            interval
        )

        return (
            time.time() % interval_sec
        ) / interval_sec

    def _check_sniper_entry(
        self,
        symbol: str,
        signal_score: float,
        min_open_score: float,
    ) -> bool:
        if not self.sniper_entry_enable:
            return True

        progress = self._calc_candle_progress(
            self.sniper_entry_interval
        )

        min_progress = float(
            self.sniper_entry_min_candle_progress
        )

        max_progress = float(
            self.sniper_entry_max_candle_progress
        )

        if progress < min_progress:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK][SNIPER-EARLY] "
                        "symbol=%s progress=%.3f min=%.3f "
                        "score=%.4f min_open=%.4f",
                        symbol,
                        progress,
                        min_progress,
                        signal_score,
                        min_open_score,
                    )
            except Exception:
                pass

            return False

        if progress > max_progress:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK][SNIPER-LATE] "
                        "symbol=%s progress=%.3f max=%.3f "
                        "score=%.4f min_open=%.4f",
                        symbol,
                        progress,
                        max_progress,
                        signal_score,
                        min_open_score,
                    )
            except Exception:
                pass

            return False

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][SNIPER-ENTRY] "
                    "symbol=%s progress=%.3f min=%.3f max=%.3f "
                    "score=%.4f min_open=%.4f",
                    symbol,
                    progress,
                    min_progress,
                    max_progress,
                    signal_score,
                    min_open_score,
                )
        except Exception:
            pass

        return True

    # =========================================================
    # SHORT 1M CANDLE DIRECTION
    # =========================================================

    def _check_short_candle_direction(
        self,
        symbol: str,
    ) -> bool:
        """
        SHORT giriş şartı:

        1. Önceki tamamlanmış 1m mum yeşil olmalı.
        2. Mevcut mum yeni başlamış olmalı (sniper kontrolü ayrı).
        3. Mevcut mum aşağı dönüş göstermeli.
        4. Aşağı dönüş minimum yüzdeyi aşmalı.
        """
        if not self.entry_candle_dir_filter_enable:
            return True

        candles = self._get_entry_candle_pair(
            symbol=symbol,
            interval=self.entry_candle_dir_interval,
        )

        if candles is None:
            return False

        previous, current = candles

        prev_open, prev_close = previous
        curr_open, curr_close = current

        # Önceki tamamlanmış mum mutlaka yeşil olmalı.
        if prev_close <= prev_open:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[PREV-CANDLE-NOT-GREEN] "
                        "symbol=%s interval=%s "
                        "prev_open=%.8f prev_close=%.8f",
                        symbol,
                        self.entry_candle_dir_interval,
                        prev_open,
                        prev_close,
                    )
            except Exception:
                pass

            return False

        # Mevcut mum aşağı dönmüş olmalı.
        if curr_close >= curr_open:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[CURRENT-NOT-REVERSING-DOWN] "
                        "symbol=%s interval=%s "
                        "curr_open=%.8f curr_close=%.8f",
                        symbol,
                        self.entry_candle_dir_interval,
                        curr_open,
                        curr_close,
                    )
            except Exception:
                pass

            return False

        # Çok küçük fiyat oynamasını dönüş olarak kabul etme.
        reversal_pct = (
            (curr_open - curr_close)
            / curr_open
        )

        min_reversal_pct = self._env_float(
            "SHORT_CURRENT_REVERSAL_MIN_PCT",
            0.0001,
        )

        if reversal_pct < min_reversal_pct:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[CURRENT-REVERSAL-WEAK] "
                        "symbol=%s reversal_pct=%.6f min=%.6f "
                        "curr_open=%.8f curr_close=%.8f",
                        symbol,
                        reversal_pct,
                        min_reversal_pct,
                        curr_open,
                        curr_close,
                    )
            except Exception:
                pass

            return False

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][CANDLE-CONFIRM] "
                    "symbol=%s interval=%s "
                    "prev_open=%.8f prev_close=%.8f prev=green "
                    "curr_open=%.8f curr_close=%.8f "
                    "current=down reversal_pct=%.6f allow=1",
                    symbol,
                    self.entry_candle_dir_interval,
                    prev_open,
                    prev_close,
                    curr_open,
                    curr_close,
                    reversal_pct,
                )
        except Exception:
            pass

        return True

    # =========================================================
    # SHORT CONTINUATION ENTRY
    # =========================================================

    def _prepare_short_continuation_candles(
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
            rows = self.client.futures_klines(
                symbol=sym_u,
                interval="1m",
                limit=5,
            )
        except Exception as exc:
            try:
                if self.logger:
                    self.logger.warning(
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
                    if self.logger:
                        self.logger.info(
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

    def _get_short_continuation_config(
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

    def _evaluate_short_continuation_rules(
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
                if self.logger:
                    self.logger.info(
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
                if self.logger:
                    self.logger.info(
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
                if self.logger:
                    self.logger.info(
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
                if self.logger:
                    self.logger.info(
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
                if self.logger:
                    self.logger.info(
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
            if self.logger:
                self.logger.info(
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

    def _check_short_continuation(
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
            self._prepare_short_continuation_candles(
                symbol=sym_u,
            )
        )

        if not candle_context.get("ok", False):
            return False

        parsed = candle_context["completed"]
        current = candle_context["current"]

        config = (
            self._get_short_continuation_config()
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
            self._evaluate_short_continuation_rules(
                symbol=sym_u,
                completed=parsed,
                current=current,
                config=config,
            )
        )

    def _check_short_entry_pattern(
        self,
        symbol: str,
    ) -> bool:
        """
        SHORT giriş modelleri birbirinin alternatifidir.

        Öncelik:
        1. Reversal
        2. Continuation
        """
        if self._check_short_candle_direction(
            symbol=symbol,
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][ENTRY-MODE] "
                        "symbol=%s mode=reversal",
                        symbol,
                    )
            except Exception:
                pass

            return True

        if self._check_short_continuation(
            symbol=symbol,
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][ENTRY-MODE] "
                        "symbol=%s mode=continuation",
                        symbol,
                    )
            except Exception:
                pass

            return True

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][OPEN-BLOCK]"
                    "[NO-ENTRY-PATTERN] symbol=%s",
                    symbol,
                )
        except Exception:
            pass

        return False

    # =========================================================
    # PRICE
    # =========================================================

    def _get_cached_mid_price(
        self,
        symbol: str,
    ) -> Optional[float]:
        sym_u = str(symbol or "").upper().strip()

        try:
            pc = self.price_cache

            if (
                pc is not None
                and hasattr(pc, "get_mid")
            ):
                px = pc.get_mid(
                    sym_u,
                    max_age_sec=2.0,
                )

                px_f = self._clip_float(px, None)

                if px_f is not None and px_f > 0.0:
                    return float(px_f)

        except Exception:
            pass

        try:
            rpc = self.redis_price_cache

            if (
                rpc is not None
                and hasattr(rpc, "get_mid")
            ):
                px = rpc.get_mid(
                    sym_u,
                    max_age_sec=2.0,
                )

                px_f = self._clip_float(px, None)

                if px_f is not None and px_f > 0.0:
                    return float(px_f)

        except Exception:
            pass

        return None

    def _resolve_price(
        self,
        symbol: str,
        price: Any = None,
    ) -> Optional[float]:
        px = self._clip_float(
            price,
            None,
        )

        if px is not None and px > 0.0:
            return float(px)

        return self._get_cached_mid_price(symbol)

    def _resolve_order_price(
        self,
        symbol: str,
        fallback_price: Any = None,
    ) -> Optional[float]:
        sym_u = str(symbol or "").upper().strip()

        try:
            fn = getattr(
                self.client,
                "futures_mark_price",
                None,
            )

            if callable(fn):
                out = fn(symbol=sym_u)

                if isinstance(out, dict):
                    px = self._clip_float(
                        out.get("markPrice"),
                        None,
                    )

                    if px is not None and px > 0.0:
                        return float(px)

        except Exception:
            pass

        return self._resolve_price(
            symbol=sym_u,
            price=fallback_price,
        )

    # =========================================================
    # EXCHANGE INFO
    # =========================================================

    def _get_exchange_info_cached(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        now_ts = time.time()

        cache_ok = (
            not force_refresh
            and isinstance(
                self._exchange_info_cache,
                dict,
            )
            and (
                now_ts
                - self._exchange_info_cache_ts
            ) < self.exchange_info_ttl_sec
        )

        if cache_ok:
            return dict(self._exchange_info_cache)

        try:
            fn = getattr(
                self.client,
                "futures_exchange_info",
                None,
            )

            if not callable(fn):
                return {}

            info = fn()

            if not isinstance(info, dict):
                return {}

            self._exchange_info_cache = dict(info)
            self._exchange_info_cache_ts = now_ts

            return dict(info)

        except Exception:
            return {}

    def _get_symbol_info(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        sym_u = str(symbol or "").upper().strip()

        info = self._get_exchange_info_cached()

        symbols = info.get("symbols")

        if not isinstance(symbols, list):
            return {}

        for item in symbols:
            if not isinstance(item, dict):
                continue

            if (
                str(item.get("symbol") or "")
                .upper()
                .strip()
                == sym_u
            ):
                return dict(item)

        return {}

    @staticmethod
    def _floor_to_step(
        qty: float,
        step: float,
    ) -> float:
        if step <= 0.0:
            return float(qty)

        qty_d = Decimal(str(qty))
        step_d = Decimal(str(step))

        return float(
            (
                (qty_d / step_d)
                .to_integral_value(
                    rounding=ROUND_DOWN
                )
            ) * step_d
        )

    def _normalize_order_qty(
        self,
        symbol: str,
        raw_qty: float,
        price: float,
    ) -> Tuple[float, Dict[str, Any]]:
        info = self._get_symbol_info(symbol)

        step = 0.0
        min_qty = 0.0
        min_notional = 0.0

        filters = info.get("filters")

        if isinstance(filters, list):
            for item in filters:
                if not isinstance(item, dict):
                    continue

                ftype = str(
                    item.get("filterType") or ""
                ).upper()

                if ftype in (
                    "LOT_SIZE",
                    "MARKET_LOT_SIZE",
                ):
                    try:
                        step = max(
                            step,
                            float(
                                item.get("stepSize")
                                or 0.0
                            ),
                        )
                    except Exception:
                        pass

                    try:
                        min_qty = max(
                            min_qty,
                            float(
                                item.get("minQty")
                                or 0.0
                            ),
                        )
                    except Exception:
                        pass

                if ftype in (
                    "MIN_NOTIONAL",
                    "NOTIONAL",
                ):
                    try:
                        min_notional = max(
                            min_notional,
                            float(
                                item.get("notional")
                                or item.get("minNotional")
                                or 0.0
                            ),
                        )
                    except Exception:
                        pass

        qty = self._floor_to_step(
            float(raw_qty),
            float(step),
        )

        try:
            qp = int(
                info.get("quantityPrecision")
                or 8
            )
            qty = round(float(qty), qp)
        except Exception:
            pass

        if qty < min_qty:
            return 0.0, {
                "reason": "below_min_qty",
                "min_qty": min_qty,
                "step": step,
            }

        notional = float(qty) * float(price)

        if (
            min_notional > 0.0
            and notional < min_notional
        ):
            return 0.0, {
                "reason": "below_min_notional",
                "min_notional": min_notional,
                "notional": notional,
            }

        return float(qty), {
            "step": step,
            "min_qty": min_qty,
            "min_notional": min_notional,
            "notional": notional,
        }

    # =========================================================
    # SHORT OPEN HELPERS
    # =========================================================

    def _count_open_positions(self) -> int:
        return len(self._get_all_local_positions())

    def _set_symbol_leverage(
        self,
        symbol: str,
        leverage: int,
    ) -> int:
        lev = max(
            1,
            min(
                int(leverage),
                int(self.max_leverage),
            ),
        )

        if self.dry_run:
            return lev

        try:
            fn = getattr(
                self.client,
                "futures_change_leverage",
                None,
            )

            if callable(fn):
                out = fn(
                    symbol=str(symbol).upper(),
                    leverage=lev,
                )

                if isinstance(out, dict):
                    return int(
                        out.get("leverage")
                        or lev
                    )

        except Exception as exc:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][LEVERAGE][WARN] "
                        "symbol=%s lev=%s err=%s",
                        symbol,
                        lev,
                        str(exc)[:200],
                    )
            except Exception:
                pass

        return lev

    def _exchange_open_market(
        self,
        symbol: str,
        qty: float,
    ) -> Dict[str, Any]:
        sym_u = str(symbol).upper().strip()

        if self.dry_run:
            return {
                "status": "dry_run",
                "symbol": sym_u,
                "side": "SELL",
                "qty": float(qty),
            }

        fn = getattr(
            self.client,
            "futures_create_order",
            None,
        )

        if not callable(fn):
            raise RuntimeError(
                "futures_create_order unavailable"
            )

        payload: Dict[str, Any] = {
            "symbol": sym_u,
            "side": "SELL",
            "type": "MARKET",
            "quantity": float(qty),
        }

        if self.hedge_mode_enabled:
            payload["positionSide"] = "SHORT"

        return fn(**payload)

    # =========================================================
    # REAL SHORT OPEN
    # =========================================================

    def _validate_short_open_request(
        self,
        *,
        symbol: str,
        now_ts: float,
    ) -> Optional[Dict[str, Any]]:
        """
        SHORT pozisyon açılmadan önce yerel giriş engellerini kontrol eder.

        Open lock bu helper dışında yönetilir. Böylece lock her durumda
        open_position_from_signal() içindeki finally bloğunda temizlenir.
        """
        sym_u = str(symbol or "").upper().strip()

        if (
            self.block_same_symbol_reentry
            and self._get_position(sym_u) is not None
        ):
            return {
                "status": "skip",
                "reason": "position_exists",
                "symbol": sym_u,
            }

        last_open_ts = 0.0

        try:
            if self.redis is not None:
                raw = self.redis.get(
                    f"bot:last_open_ts:{sym_u}"
                )

                last_open_ts = float(raw or 0.0)

        except Exception:
            last_open_ts = 0.0

        if (
            last_open_ts > 0.0
            and (
                float(now_ts) - last_open_ts
            ) < self.same_symbol_open_cooldown_sec
        ):
            return {
                "status": "skip",
                "reason": "symbol_cooldown",
                "symbol": sym_u,
            }

        if (
            self._count_open_positions()
            >= self.max_open_positions
        ):
            return {
                "status": "skip",
                "reason": "max_open_positions",
                "symbol": sym_u,
            }

        return None

    def _prepare_short_open_order(
        self,
        *,
        symbol: str,
        fallback_price: Any = None,
    ) -> Dict[str, Any]:
        """
        SHORT market emri için fiyat, miktar ve leverage hazırlar.

        Bu helper emir göndermez ve pozisyon kaydı oluşturmaz.
        """
        sym_u = str(symbol or "").upper().strip()

        order_price = self._resolve_order_price(
            symbol=sym_u,
            fallback_price=fallback_price,
        )

        if order_price is None or order_price <= 0.0:
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "missing_price",
                    "symbol": sym_u,
                },
            }

        notional = min(
            float(self.base_order_notional),
            float(self.max_position_notional),
        )

        notional = max(
            10.0,
            float(notional),
        )

        raw_qty = (
            float(notional)
            / float(order_price)
        )

        qty, qty_meta = self._normalize_order_qty(
            symbol=sym_u,
            raw_qty=raw_qty,
            price=float(order_price),
        )

        if qty <= 0.0:
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "bad_qty",
                    "symbol": sym_u,
                    "qty_meta": qty_meta,
                },
            }

        leverage = self._set_symbol_leverage(
            symbol=sym_u,
            leverage=self.default_order_leverage,
        )

        return {
            "ok": True,
            "symbol": sym_u,
            "order_price": float(order_price),
            "qty": float(qty),
            "notional": (
                float(qty)
                * float(order_price)
            ),
            "leverage": int(leverage),
            "qty_meta": qty_meta,
        }

    def _finalize_short_open(
        self,
        *,
        symbol: str,
        interval: str,
        qty: float,
        order_price: float,
        leverage: int,
        now_ts: float,
        order: Any,
    ) -> Dict[str, Any]:
        """
        Açılan SHORT pozisyonu yerel store ve Redis'e kaydeder.

        Bu helper market emri göndermez.
        """
        sym_u = str(symbol or "").upper().strip()

        pos = {
            "symbol": sym_u,
            "side": "short",
            "qty": float(qty),
            "entry_price": float(order_price),
            "notional": (
                float(qty)
                * float(order_price)
            ),
            "interval": str(
                interval or self.entry_interval
            ),
            "leverage": int(leverage),
            "created_ts": float(now_ts),
            "best_roi_pct": 0.0,
        }

        self._set_position(
            sym_u,
            pos,
        )

        try:
            if self.redis is not None:
                self.redis.setex(
                    f"bot:last_open_ts:{sym_u}",
                    3600,
                    str(now_ts),
                )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][OPENED] "
                    "symbol=%s qty=%.10f "
                    "price=%.8f notional=%.4f "
                    "leverage=%s",
                    sym_u,
                    qty,
                    order_price,
                    pos["notional"],
                    leverage,
                )
        except Exception:
            pass

        return {
            "status": (
                "dry_run"
                if self.dry_run
                else "opened"
            ),
            "symbol": sym_u,
            "side": "short",
            "qty": float(qty),
            "price": float(order_price),
            "notional": float(
                pos["notional"]
            ),
            "leverage": int(leverage),
            "order": order,
        }

    def open_position_from_signal(
        self,
        symbol: str,
        side: str,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sym_u = str(symbol or "").upper().strip()
        side_u = str(side or "").lower().strip()

        meta0 = (
            dict(meta)
            if isinstance(meta, dict)
            else {}
        )

        if not sym_u or side_u != "short":
            return {
                "status": "skip",
                "reason": "short_only",
            }

        now_ts = time.time()

        lock_ts = float(
            self._symbol_open_locks.get(
                sym_u,
                0.0,
            )
            or 0.0
        )

        if now_ts - lock_ts < 5.0:
            return {
                "status": "skip",
                "reason": "open_lock",
                "symbol": sym_u,
            }

        self._symbol_open_locks[sym_u] = now_ts

        try:
            guard_result = self._validate_short_open_request(
                symbol=sym_u,
                now_ts=now_ts,
            )

            if guard_result is not None:
                return guard_result

            order_plan = self._prepare_short_open_order(
                symbol=sym_u,
                fallback_price=meta0.get("price"),
            )

            if not order_plan.get("ok", False):
                result = order_plan.get("result")

                if isinstance(result, dict):
                    return result

                return {
                    "status": "skip",
                    "reason": "open_prepare_failed",
                    "symbol": sym_u,
                }

            order_price = float(
                order_plan["order_price"]
            )

            qty = float(
                order_plan["qty"]
            )

            leverage = int(
                order_plan["leverage"]
            )

            order = self._exchange_open_market(
                symbol=sym_u,
                qty=float(qty),
            )

            return self._finalize_short_open(
                symbol=sym_u,
                interval=interval,
                qty=qty,
                order_price=order_price,
                leverage=leverage,
                now_ts=now_ts,
                order=order,
            )

        except Exception as exc:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC-SHORT][OPEN][ERROR] "
                        "symbol=%s err=%s",
                        sym_u,
                        str(exc)[:300],
                    )
            except Exception:
                pass

            return {
                "status": "skip",
                "reason": "open_exception",
                "symbol": sym_u,
            }

        finally:
            self._symbol_open_locks.pop(
                sym_u,
                None,
            )

    # =========================================================
    # SHORT CLOSE
    # =========================================================

    def _exchange_close_market(
        self,
        symbol: str,
        qty: float,
    ) -> Dict[str, Any]:
        sym_u = str(symbol or "").upper().strip()

        if self.dry_run:
            return {
                "status": "dry_run",
                "symbol": sym_u,
                "side": "BUY",
                "qty": float(qty),
            }

        fn = getattr(
            self.client,
            "futures_create_order",
            None,
        )

        if not callable(fn):
            raise RuntimeError(
                "futures_create_order unavailable"
            )

        payload: Dict[str, Any] = {
            "symbol": sym_u,
            "side": "BUY",
            "type": "MARKET",
            "quantity": float(qty),
        }

        if self.hedge_mode_enabled:
            payload["positionSide"] = "SHORT"
        else:
            payload["reduceOnly"] = True

        return fn(**payload)

    def _validate_short_close_request(
        self,
        *,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        SHORT close öncesi pozisyon doğrulaması.
        """
        sym_u = str(symbol or "").upper().strip()

        pos = self._get_position(sym_u)

        if not isinstance(pos, dict):
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "position_not_found",
                    "symbol": sym_u,
                },
            }

        if str(pos.get("side") or "").lower() != "short":
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "not_short_position",
                    "symbol": sym_u,
                },
            }

        try:
            qty = float(pos.get("qty") or 0.0)
        except Exception:
            qty = 0.0

        if qty <= 0.0:
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "bad_qty",
                    "symbol": sym_u,
                },
            }

        return {
            "ok": True,
            "position": pos,
            "qty": float(qty),
        }

    def _finalize_short_close(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
        qty: float,
        exit_price: float,
        reason: str,
        interval: str,
        order: Any,
    ) -> Dict[str, Any]:
        """
        SHORT kapanış sonrası PnL/ROI hesabı ve yerel temizliği yapar.

        Bu helper borsaya kapatma emri göndermez.
        """
        sym_u = str(symbol or "").upper().strip()

        entry_price = float(
            position.get("entry_price")
            or 0.0
        )

        pnl_pct = 0.0

        if (
            entry_price > 0.0
            and float(exit_price) > 0.0
        ):
            pnl_pct = (
                entry_price
                - float(exit_price)
            ) / entry_price

        leverage = float(
            position.get("leverage")
            or self.default_order_leverage
            or 1.0
        )

        roi_pct = pnl_pct * leverage

        self._del_position(sym_u)

        try:
            if self.redis is not None:
                self.redis.delete(
                    f"fixed_roi_best:{sym_u}"
                )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][CLOSED] "
                    "symbol=%s reason=%s qty=%.10f "
                    "entry=%.8f exit=%.8f "
                    "pnl_pct=%.6f roi=%.6f "
                    "interval=%s",
                    sym_u,
                    str(reason),
                    qty,
                    entry_price,
                    float(exit_price),
                    pnl_pct,
                    roi_pct,
                    str(interval or ""),
                )
        except Exception:
            pass

        return {
            "status": (
                "dry_run"
                if self.dry_run
                else "closed"
            ),
            "symbol": sym_u,
            "side": "short",
            "qty": float(qty),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl_pct": float(pnl_pct),
            "roi_pct": float(roi_pct),
            "reason": str(reason),
            "order": order,
        }

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        reason: str = "manual",
        interval: str = "",
        intent_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del intent_id
        del kwargs

        sym_u = str(symbol or "").upper().strip()

        validation = self._validate_short_close_request(
            symbol=sym_u,
        )

        if not validation.get("ok", False):
            return validation["result"]

        pos = validation["position"]
        qty = float(validation["qty"])

        exit_price = self._resolve_price(
            symbol=sym_u,
            price=price,
        )

        if (
            exit_price is None
            or exit_price <= 0.0
        ):
            exit_price = float(
                pos.get("entry_price")
                or 0.0
            )

        try:
            order = self._exchange_close_market(
                symbol=sym_u,
                qty=float(qty),
            )

            return self._finalize_short_close(
                symbol=sym_u,
                position=pos,
                qty=qty,
                exit_price=exit_price,
                reason=reason,
                interval=interval,
                order=order,
            )

        except Exception as exc:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC-SHORT][CLOSE][ERROR] "
                        "symbol=%s reason=%s err=%s",
                        sym_u,
                        str(reason),
                        str(exc)[:300],
                    )
            except Exception:
                pass

            return {
                "status": "error",
                "reason": "close_exception",
                "symbol": sym_u,
            }

    # =========================================================
    # PUBLIC API PLACEHOLDERS
    # =========================================================

    # =========================================================
    # SHORT PENDING ENTRY HELPERS
    # =========================================================

    @staticmethod
    def _interval_to_seconds(
        interval: str,
    ) -> int:
        """
        Binance interval metnini saniyeye çevirir.

        Desteklenen örnekler:
        1m, 3m, 5m, 15m, 1h, 4h, 1d, 1w
        """
        value = str(interval or "").strip().lower()

        if not value:
            return 60

        unit = value[-1]

        try:
            amount = int(value[:-1])
        except (TypeError, ValueError):
            return 60

        if amount <= 0:
            return 60

        multipliers = {
            "s": 1,
            "m": 60,
            "h": 60 * 60,
            "d": 24 * 60 * 60,
            "w": 7 * 24 * 60 * 60,
        }

        multiplier = multipliers.get(unit)

        if multiplier is None:
            return 60

        return int(amount * multiplier)

    def _get_candle_id(
        self,
        interval: str,
        timestamp: Optional[float] = None,
    ) -> int:
        """
        Verilen zamanın ait olduğu mumun sayısal kimliğini döndürür.

        Örnek:
        1m interval için Unix zamanını 60 saniyelik bloklara böler.
        """
        interval_sec = self._interval_to_seconds(
            interval
        )

        ts = float(
            time.time()
            if timestamp is None
            else timestamp
        )

        return int(ts // interval_sec)

    def _get_short_pending(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return None

        pending = self._pending_short_entries.get(sym_u)

        if not isinstance(pending, dict):
            return None

        return pending

    def _pop_short_pending(
        self,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return None

        pending = self._pending_short_entries.pop(
            sym_u,
            None,
        )

        if not isinstance(pending, dict):
            return None

        return pending

    def _create_short_pending(
        self,
        symbol: str,
        signal_score: float,
        min_score: float,
        price: float,
    ) -> Dict[str, Any]:
        """
        SHORT pending kaydını mevcut mum kimliğiyle oluşturur.

        Aynı mum içinde yeni sinyal gelirse:
        - candle_id aynı kalır
        - score/min_score/price güncellenir

        Pending sabit saniye TTL yerine mum yaşam
        döngüsüne bağlı olarak yönetilir.
        """
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return {}

        candle_id = self._get_candle_id(
            self.sniper_entry_interval
        )

        pending = {
            "candle_id": int(candle_id),
            "interval": str(
                self.sniper_entry_interval
                or "1m"
            ),
            "score": float(signal_score),
            "min_score": float(min_score),
            "price": float(price or 0.0),
        }

        self._pending_short_entries[sym_u] = pending

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][PENDING] "
                    "symbol=%s score=%.4f "
                    "candle_id=%s interval=%s",
                    sym_u,
                    signal_score,
                    candle_id,
                    self.sniper_entry_interval,
                )
        except Exception:
            pass

        return pending

    def _cancel_short_pending(
        self,
        symbol: str,
        reason: str = "",
        signal_score: Optional[float] = None,
        min_score: Optional[float] = None,
    ) -> bool:
        """
        Pending kaydını güvenli şekilde iptal eder.

        reason=low_score olduğunda mevcut log davranışı korunur.
        """
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return False

        pending = self._pop_short_pending(sym_u)

        if pending is None:
            return False

        reason_norm = str(reason or "").lower().strip()

        if reason_norm == "low_score":
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-CANCEL] "
                        "symbol=%s reason=low_score "
                        "score=%.4f min=%.4f",
                        sym_u,
                        float(signal_score or 0.0),
                        float(min_score or 0.0),
                    )
            except Exception:
                pass

        return True

    def _classify_short_pending_window(
        self,
        *,
        created_candle_id: int,
        current_candle_id: int,
        progress: float,
        max_pending_progress: float,
    ):
        """
        Pending kaydının mevcut mum penceresindeki durumunu döndürür.

        Sonuç:
        - invalid
        - wait_same_candle
        - expired
        - wait_window
        - ready
        """
        created_id = int(created_candle_id or 0)
        current_id = int(current_candle_id or 0)

        if created_id <= 0:
            return "invalid", 0

        candle_age = current_id - created_id

        if candle_age <= 0:
            return "wait_same_candle", candle_age

        if candle_age > 1:
            return "expired", candle_age

        if (
            float(progress)
            < float(self.sniper_entry_min_candle_progress)
            or float(progress) > float(max_pending_progress)
        ):
            return "wait_window", candle_age

        return "ready", candle_age

    def _resolve_short_pending_price(
        self,
        symbol: str,
        pending: Dict[str, Any],
    ) -> Optional[float]:
        """
        Pending giriş için öncelikle canlı cache fiyatını çözer.

        Cache boşsa pending kaydındaki fiyat fallback olarak kullanılır.
        """
        sym_u = str(symbol or "").upper().strip()

        pending_price = self._get_cached_mid_price(sym_u)

        if pending_price is None or pending_price <= 0.0:
            pending_price = self._resolve_price(
                symbol=sym_u,
                price=pending.get("price"),
            )

        if pending_price is None or pending_price <= 0.0:
            return None

        return float(pending_price)

    def _handle_short_pending_open_result(
        self,
        *,
        symbol: str,
        result: Any,
    ) -> None:
        """
        Pending open sonucuna göre kaydı temizler veya yeniden denemeye bırakır.
        """
        sym_u = str(symbol or "").upper().strip()

        result_status = (
            str(result.get("status") or "").lower().strip()
            if isinstance(result, dict)
            else ""
        )

        result_reason = (
            str(result.get("reason") or "").lower().strip()
            if isinstance(result, dict)
            else ""
        )

        opened_ok = result_status in (
            "opened",
            "dry_run",
        )

        retryable_pre_order_reasons = {
            "open_lock",
            "symbol_cooldown",
            "max_open_positions",
            "missing_price",
        }

        retry_pending = (
            result_status == "skip"
            and result_reason in retryable_pre_order_reasons
        )

        if opened_ok:
            self._pop_short_pending(sym_u)

        elif retry_pending:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-RETRY] "
                        "symbol=%s status=%s reason=%s",
                        sym_u,
                        result_status,
                        result_reason,
                    )
            except Exception:
                pass

        else:
            # Bilinmeyen veya kalıcı sonuçlarda yeniden emir
            # gönderme riskini önlemek için pending temizlenir.
            self._pop_short_pending(sym_u)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][PENDING-RESULT] "
                    "symbol=%s status=%s reason=%s "
                    "opened=%s retry=%s",
                    sym_u,
                    result_status or "unknown",
                    result_reason,
                    opened_ok,
                    retry_pending,
                )
        except Exception:
            pass

    async def _process_single_short_pending_entry(
        self,
        *,
        symbol: str,
        pending: Dict[str, Any],
        current_candle_id: int,
        progress: float,
        max_pending_progress: float,
    ) -> None:
        """
        Tek bir SHORT pending kaydını değerlendirir.
        """
        sym_u = str(symbol or "").upper().strip()

        created_candle_id = int(
            pending.get("candle_id")
            or 0
        )

        window_state, candle_age = (
            self._classify_short_pending_window(
                created_candle_id=created_candle_id,
                current_candle_id=current_candle_id,
                progress=progress,
                max_pending_progress=max_pending_progress,
            )
        )

        if window_state == "invalid":
            self._pop_short_pending(sym_u)

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-CANCEL] "
                        "symbol=%s reason=invalid_candle_id",
                        sym_u,
                    )
            except Exception:
                pass

            return

        if window_state == "expired":
            self._pop_short_pending(sym_u)

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-EXPIRE] "
                        "symbol=%s created_candle=%s "
                        "current_candle=%s age=%s",
                        sym_u,
                        created_candle_id,
                        current_candle_id,
                        candle_age,
                    )
            except Exception:
                pass

            return

        if window_state != "ready":
            return

        pending_score = float(
            pending.get("score")
            or 0.0
        )

        pending_min_score = float(
            pending.get("min_score")
            or 0.0
        )

        pending_price = self._resolve_short_pending_price(
            symbol=sym_u,
            pending=pending,
        )

        if pending_price is None:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-WAIT] "
                        "symbol=%s reason=missing_price",
                        sym_u,
                    )
            except Exception:
                pass

            return

        candidate = self._evaluate_short_candidate(
            symbol=sym_u,
            price=float(pending_price),
            signal_score=pending_score,
            final_min_score=pending_min_score,
            allow_pending=False,
        )

        if not candidate.get("allow", False):
            reason = str(
                candidate.get("reason")
                or "candidate_block"
            ).strip().lower()

            retryable = bool(
                candidate.get("retryable", False)
            )

            if not retryable:
                self._cancel_short_pending(
                    symbol=sym_u,
                    reason=reason,
                    signal_score=pending_score,
                    min_score=pending_min_score,
                )

            else:
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC-SHORT][PENDING-WAIT] "
                            "symbol=%s reason=%s retryable=true",
                            sym_u,
                            reason,
                        )
                except Exception:
                    pass

            return

        pending["price"] = float(pending_price)
        pending["signal_score"] = float(pending_score)
        pending["open_min_score"] = float(pending_min_score)
        pending["entry_interval"] = str(self.entry_interval)
        pending["pending_revalidated"] = True

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][PENDING-OPEN] "
                    "symbol=%s progress=%.3f "
                    "created_candle=%s "
                    "current_candle=%s "
                    "score=%.4f min=%.4f "
                    "reason=%s",
                    sym_u,
                    progress,
                    created_candle_id,
                    current_candle_id,
                    pending_score,
                    pending_min_score,
                    candidate.get("reason"),
                )
        except Exception:
            pass

        result = await asyncio.to_thread(
            self.open_position_from_signal,
            sym_u,
            "short",
            self.entry_interval,
            pending,
        )

        self._handle_short_pending_open_result(
            symbol=sym_u,
            result=result,
        )

    async def _process_short_pending_entries(
        self,
    ) -> None:
        """
        Bekleyen SHORT girişlerini mum kimliğine göre işler.

        Pending oluşturulduğu mumda bekler ve yalnızca
        bir sonraki mumun erken giriş penceresinde açılmayı dener.
        """
        current_candle_id = self._get_candle_id(
            self.sniper_entry_interval
        )

        progress = self._calc_candle_progress(
            self.sniper_entry_interval
        )

        max_pending_progress = min(
            self.sniper_entry_max_candle_progress,
            0.20,
        )

        for psym, pending in list(
            self._pending_short_entries.items()
        ):
            if not isinstance(pending, dict):
                self._pop_short_pending(psym)
                continue

            await self._process_single_short_pending_entry(
                symbol=psym,
                pending=pending,
                current_candle_id=current_candle_id,
                progress=progress,
                max_pending_progress=max_pending_progress,
            )

    def _make_short_entry_decision(
        self,
        *,
        allow: bool,
        action: str,
        reason: str,
        signal_score: Optional[float] = None,
        required_score: Optional[float] = None,
        progress: Optional[float] = None,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        """
        SHORT giriş değerlendirmeleri için standart karar sözlüğü üretir.

        Bu helper state değiştirmez.
        """
        decision: Dict[str, Any] = {
            "allow": bool(allow),
            "action": str(action or "block").strip().lower(),
            "reason": str(reason or "unknown").strip().lower(),
            "retryable": bool(retryable),
        }

        if signal_score is not None:
            decision["score"] = float(signal_score)

        if required_score is not None:
            decision["required_score"] = float(required_score)

        if progress is not None:
            decision["progress"] = float(progress)

        return decision

    def _evaluate_short_candidate(
        self,
        symbol: str,
        price: float,
        signal_score: float,
        final_min_score: float,
        allow_pending: bool = True,
    ) -> Dict[str, Any]:
        """
        SHORT adayının emir aşamasına geçip geçemeyeceğini değerlendirir.

        Bu fonksiyon emir açmaz ve pending state değiştirmez.

        allow_pending=True:
            Geç kalınmış sniper penceresinde action="pending"
            kararı döndürebilir.

        allow_pending=False:
            Pending yöneticisi tarafından yapılan yeniden
            değerlendirmede yeni pending kararı üretmez.

        Kararın uygulanması çağıran katmanın sorumluluğundadır.

        action:
            block   -> giriş reddedildi
            pending -> sonraki uygun 1m pencere beklenecek
            open    -> giriş doğrulandı, emir açılabilir
        """
        sym_u = str(symbol or "").upper().strip()

        if not sym_u:
            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="empty_symbol",
                retryable=False,
            )

        if signal_score < final_min_score:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK][LOW-SCORE] "
                        "symbol=%s score=%.4f min=%.4f",
                        sym_u,
                        signal_score,
                        final_min_score,
                    )
            except Exception:
                pass

            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="low_score",
                signal_score=signal_score,
                required_score=final_min_score,
                retryable=False,
            )

        if (
            self.block_same_symbol_reentry
            and self._get_position(sym_u) is not None
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][OPEN-BLOCK]"
                        "[POSITION-EXISTS] symbol=%s",
                        sym_u,
                    )
            except Exception:
                pass

            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="position_exists",
                signal_score=signal_score,
                required_score=final_min_score,
                retryable=False,
            )

        progress = self._calc_candle_progress(
            self.sniper_entry_interval
        )

        if progress > self.sniper_entry_max_candle_progress:
            if allow_pending:
                return self._make_short_entry_decision(
                    allow=False,
                    action="pending",
                    reason="sniper_window_passed",
                    signal_score=signal_score,
                    required_score=final_min_score,
                    progress=progress,
                    retryable=True,
                )

            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="pending_window_passed",
                signal_score=signal_score,
                required_score=final_min_score,
                progress=progress,
                retryable=False,
            )

        if not self._check_sniper_entry(
            symbol=sym_u,
            signal_score=signal_score,
            min_open_score=final_min_score,
        ):
            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="sniper_window",
                signal_score=signal_score,
                required_score=final_min_score,
                progress=progress,
                retryable=True,
            )

        if not self._check_short_entry_pattern(
            symbol=sym_u,
        ):
            return self._make_short_entry_decision(
                allow=False,
                action="block",
                reason="entry_pattern",
                signal_score=signal_score,
                required_score=final_min_score,
                progress=progress,
                retryable=True,
            )

        return self._make_short_entry_decision(
            allow=True,
            action="open",
            reason="entry_confirmed",
            signal_score=signal_score,
            required_score=final_min_score,
            progress=progress,
            retryable=False,
        )

    def _resolve_short_open_context(
        self,
        *,
        symbol: str,
        probs: Optional[Dict[str, float]],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        SHORT giriş değerlendirmesinde kullanılacak skor bağlamını
        hazırlar.

        Bu helper:
            - signal score hesaplar
            - coin minimum score çözer
            - dynamic minimum score hesaplar
            - kopyalanmış extra sözlüğünü döndürür

        Pending oluşturmaz ve emir açmaz.
        """
        sym_u = str(symbol or "").upper().strip()

        extra0: Dict[str, Any] = dict(
            extra
            if isinstance(extra, dict)
            else {}
        )

        probs0: Dict[str, float] = (
            probs
            if isinstance(probs, dict)
            else {}
        )

        side_norm = "short"

        signal_score = (
            self._resolve_signal_score_for_open(
                side=side_norm,
                extra=extra0,
                probs=probs0,
            )
        )

        coin_min_score = (
            self._get_coin_side_min_open_score(
                symbol=sym_u,
                side=side_norm,
                default_score=self.open_min_score,
            )
        )

        final_min_score = (
            self._resolve_dynamic_open_score(
                base_score=coin_min_score,
                extra=extra0,
            )
        )

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][DECISION] "
                    "symbol=%s score=%.4f "
                    "coin_min=%.4f final_min=%.4f",
                    sym_u,
                    signal_score,
                    coin_min_score,
                    final_min_score,
                )
        except Exception:
            pass

        return {
            "extra": extra0,
            "signal_score": float(signal_score),
            "coin_min_score": float(coin_min_score),
            "final_min_score": float(final_min_score),
        }

    def _apply_short_candidate_decision(
        self,
        candidate: Optional[Dict[str, Any]],
        *,
        symbol: str,
        price: float,
        signal_score: float,
        final_min_score: float,
    ) -> bool:
        """
        SHORT evaluator kararını uygular.

        True:
            Emir açma akışı devam edebilir.

        False:
            Karar pending veya block olduğu için akış durmalıdır.

        Evaluator state değiştirmez; pending state değişiklikleri
        yalnızca bu uygulama katmanında yapılır.
        """
        decision = (
            candidate
            if isinstance(candidate, dict)
            else {}
        )

        sym_u = str(symbol or "").upper().strip()

        action = str(
            decision.get("action")
            or "block"
        ).strip().lower()

        reason = str(
            decision.get("reason")
            or "candidate_block"
        ).strip().lower()

        allow = bool(
            decision.get("allow", False)
        )

        if allow and action == "open":
            return True

        if action == "pending":
            self._create_short_pending(
                symbol=sym_u,
                signal_score=signal_score,
                min_score=final_min_score,
                price=price,
            )
            return False

        if reason == "low_score":
            self._cancel_short_pending(
                symbol=sym_u,
                reason=reason,
                signal_score=signal_score,
                min_score=final_min_score,
            )

        elif reason == "position_exists":
            self._pop_short_pending(sym_u)

        return False

    def _prepare_short_open_payload(
        self,
        extra: Optional[Dict[str, Any]],
        *,
        price: float,
        signal_score: float,
        final_min_score: float,
    ) -> Dict[str, Any]:
        """
        SHORT emir açılışı için kullanılacak payload'u hazırlar.

        Gelen sözlüğü doğrudan değiştirmemek için yeni dict üretir.
        Bu helper emir açmaz ve state değiştirmez.
        """
        payload: Dict[str, Any] = dict(
            extra
            if isinstance(extra, dict)
            else {}
        )

        try:
            price_f = float(price or 0.0)
        except Exception:
            price_f = 0.0

        payload["price"] = price_f
        payload["signal_score"] = float(signal_score)
        payload["open_min_score"] = float(final_min_score)
        payload["entry_interval"] = str(self.entry_interval)

        return payload

    async def execute_decision(
        self,
        signal: str,
        symbol: str,
        price: float,
        size: Optional[float],
        interval: str,
        training_mode: bool,
        hybrid_mode: bool,
        probs: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        del size
        del interval
        del training_mode
        del hybrid_mode

        started_ts = time.perf_counter()

        sym_u = str(symbol or "").upper().strip()
        signal_u = str(signal or "").lower().strip()

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][ENTER] symbol=%s signal=%s",
                    symbol,
                    signal,
                )
        except Exception:
            pass

        if not sym_u:
            return

        # SHORT-only executor.
        if signal_u not in (
            "short",
            "sell",
            "bear",
            "down",
        ):
            return

        open_context = self._resolve_short_open_context(
            symbol=sym_u,
            probs=probs,
            extra=extra,
        )

        extra0 = open_context["extra"]
        signal_score = float(
            open_context["signal_score"]
        )
        final_min_score = float(
            open_context["final_min_score"]
        )

        candidate = self._evaluate_short_candidate(
            symbol=sym_u,
            price=price,
            signal_score=signal_score,
            final_min_score=final_min_score,
        )

        continue_open = (
            self._apply_short_candidate_decision(
                candidate=candidate,
                symbol=sym_u,
                price=price,
                signal_score=signal_score,
                final_min_score=final_min_score,
            )
        )

        if not continue_open:
            return

        extra0 = self._prepare_short_open_payload(
            extra=extra0,
            price=price,
            signal_score=signal_score,
            final_min_score=final_min_score,
        )

        decision_ms = (
            time.perf_counter() - started_ts
        ) * 1000.0

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][OPEN-NOW] "
                    "symbol=%s score=%.4f min=%.4f "
                    "decision_ms=%.3f",
                    sym_u,
                    signal_score,
                    final_min_score,
                    decision_ms,
                )
        except Exception:
            pass

        await asyncio.to_thread(
            self.open_position_from_signal,
            sym_u,
            "short",
            self.entry_interval,
            extra0,
        )


    def _get_exchange_short_positions(self):
        out = {}
        rows = self.client.futures_position_information()
        for row in rows or []:
            try:
                amt = float(row.get("positionAmt") or 0.0)
                sym = str(row.get("symbol") or "").upper()
                if amt >= 0 or not sym:
                    continue
                out[sym] = {
                    "symbol": sym,
                    "side": "short",
                    "qty": abs(amt),
                    "entry_price": float(row.get("entryPrice") or 0.0),
                    "mark_price": float(row.get("markPrice") or 0.0),
                    "leverage": int(float(row.get("leverage") or self.default_order_leverage)),
                }
            except Exception:
                continue
        return out


    def sync_positions_with_exchange(self) -> Dict[str, Any]:
        ex = self._get_exchange_short_positions()
        local = self._get_all_local_positions()
        added = []
        removed = []
        updated = []

        for sym, pos in ex.items():
            cur = local.get(sym)
            if cur is None:
                self._set_position(sym, pos)
                added.append(sym)
                continue

            cur["qty"] = float(pos.get("qty") or 0.0)
            cur["entry_price"] = float(pos.get("entry_price") or 0.0)
            cur["mark_price"] = float(pos.get("mark_price") or 0.0)
            self._set_position(sym, cur)
            updated.append(sym)

        for sym in list(local):
            if sym not in ex:
                self._del_position(sym)
                removed.append(sym)

        out = {
            "exchange_open": len(ex),
            "local_open": len(self._get_all_local_positions()),
            "added_local": added,
            "removed_local": removed,
            "updated_local": updated,
        }

        if self.logger:
            self.logger.info("[EXEC-SHORT][SYNC] %s", out)

        return out

    async def _position_sync_loop(self) -> None:
        while True:
            try:
                if self.position_sync_enabled:
                    await asyncio.to_thread(
                        self.sync_positions_with_exchange
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][SYNC-LOOP][WARN] %s",
                        str(exc)[:200],
                    )

            await asyncio.sleep(
                max(1, self.position_sync_interval_sec)
            )

    def _calc_short_roi(self, pos, price):
        entry = float(pos.get("entry_price") or 0.0)
        lev = float(pos.get("leverage") or 1.0)
        if entry <= 0 or price <= 0:
            return 0.0
        return ((entry - float(price)) / entry) * lev

    async def _position_lifecycle_loop(self) -> None:
        while True:
            try:
                if self.position_lifecycle_enabled:
                    await self._process_short_pending_entries()

                    for sym, pos in self._get_all_local_positions().items():
                        price = self._resolve_order_price(
                            sym,
                            pos.get("mark_price"),
                        )
                        if price is None or price <= 0:
                            continue

                        roi = self._calc_short_roi(pos, float(price))

                        if roi <= self.roi_hard_stop_pct:
                            await asyncio.to_thread(
                                self.close_position,
                                sym,
                                float(price),
                                "roi_hard_stop",
                                "lifecycle",
                            )
                            continue

                        old_best_roi = float(
                            pos.get("best_roi_pct") or 0.0
                        )
                        best_roi = max(
                            old_best_roi,
                            float(roi),
                        )

                        if best_roi > old_best_roi:
                            pos["best_roi_pct"] = float(best_roi)
                            self._set_position(sym, pos)

                        if (
                            self.fixed_roi_tp_enable
                            and best_roi >= self.fixed_roi_tp_pct
                        ):
                            giveback = max(
                                0.0,
                                float(self.fixed_roi_tp_giveback),
                            )
                            trigger_roi = (
                                float(best_roi) - giveback
                            )

                            if (
                                old_best_roi
                                < self.fixed_roi_tp_pct
                                <= best_roi
                                and self.logger
                            ):
                                self.logger.info(
                                    "[EXEC-SHORT][TP-ARMED] "
                                    "symbol=%s roi=%.6f "
                                    "best_roi=%.6f target=%.6f "
                                    "giveback=%.6f trigger=%.6f",
                                    sym,
                                    roi,
                                    best_roi,
                                    self.fixed_roi_tp_pct,
                                    giveback,
                                    trigger_roi,
                                )

                            if roi <= trigger_roi:
                                if self.logger:
                                    self.logger.info(
                                        "[EXEC-SHORT][TP-GIVEBACK] "
                                        "symbol=%s roi=%.6f "
                                        "best_roi=%.6f "
                                        "giveback=%.6f "
                                        "trigger=%.6f",
                                        sym,
                                        roi,
                                        best_roi,
                                        giveback,
                                        trigger_roi,
                                    )

                                await asyncio.to_thread(
                                    self.close_position,
                                    sym,
                                    float(price),
                                    "fixed_roi_tp_giveback",
                                    "lifecycle",
                                )
                                continue

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][LIFECYCLE][WARN] %s",
                        str(exc)[:200],
                    )

            await asyncio.sleep(
                max(1, self.position_lifecycle_interval_sec)
            )
