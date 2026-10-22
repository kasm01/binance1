# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import math
from decimal import Decimal, ROUND_DOWN
import os
import time
from typing import Any, Dict, Optional, Tuple

from core.short_entry_candle_service import ShortEntryCandleService
from orchestration.context.entry_context_snapshot import build_entry_context_from_rows
from orchestration.risk.risk_master import (
    evaluate_position_risk,
    resolve_market_context_multiplier,
)
from orchestration.telemetry.market_context_shadow import (
    build_shadow_event,
    publish_shadow_event,
)


class TradeExecutorShort:
    """
    SHORT-only futures executor with one authority per architecture stage.

    Decision flow:
        MODEL
        -> SIGNAL NORMALIZATION
        -> MARKET CONTEXT
        -> MODEL SCORE GATE
        -> SNIPER
        -> ENTRY CONFIRMATION (READY / PENDING / REJECT)
        -> PENDING REVALIDATION
        -> EXECUTION RISK
        -> FINAL SAFETY
        -> ORDER
        -> POSITION LIFECYCLE
        -> TELEMETRY / OUTCOME FEEDBACK

    Core rule:
        A behavior is decided in one stage only. Helper functions may compute
        data for a stage but must not introduce parallel veto authorities.
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
        # position_manager / risk_manager / telegram_bot remain constructor-compatible
        # but are intentionally not retained: SHORT execution owns no parallel managers.
        del position_manager, risk_manager, telegram_bot
        kwargs.pop("tg_bot", None)

        self.redis_price_cache = kwargs.get("redis_price_cache")
        self.hedge_mode_enabled = True

        # Binance Futures hedge mode compatibility.

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

        # -----------------------------------------------------
        # order sizing authority
        # -----------------------------------------------------
        self.short_sizing_mode = str(
            os.getenv("SHORT_SIZING_MODE", "fixed") or "fixed"
        ).strip().lower()
        if self.short_sizing_mode not in ("fixed", "balance_pct", "equity_single"):
            self.short_sizing_mode = "fixed"

        self.per_position_balance_pct = max(
            0.01,
            min(
                1.0,
                self._env_float("PER_POSITION_BALANCE_PCT", 0.30),
            ),
        )

        self.use_available_balance_for_sizing = self._env_bool(
            "USE_AVAILABLE_BALANCE_FOR_SIZING",
            True,
        )

        # SHORT execution operational knobs: single .env authority.
        self.short_min_order_notional_usdt = max(
            0.0, self._env_float("SHORT_MIN_ORDER_NOTIONAL_USDT", 10.0)
        )
        self.short_price_cache_max_age_sec = max(
            0.1, self._env_float("SHORT_PRICE_CACHE_MAX_AGE_SEC", 2.0)
        )
        self.short_open_lock_sec = max(
            0.0, self._env_float("SHORT_OPEN_LOCK_SEC", 5.0)
        )
        self.short_pending_max_candle_age = max(
            1, self._env_int("SHORT_PENDING_MAX_CANDLE_AGE", 1)
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
        # sniper entry timing
        # -----------------------------------------------------
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
        self.entry_candle_dir_interval = str(
            os.getenv(
                "ENTRY_CANDLE_DIR_INTERVAL",
                "1m",
            )
            or "1m"
        ).strip()

        self.short_current_reversal_min_pct = max(
            0.0, self._env_float("SHORT_CURRENT_REVERSAL_MIN_PCT", 0.0001)
        )

        # -----------------------------------------------------
        # duplicate / position limits
        # -----------------------------------------------------
        self.max_open_positions = self._env_int(
            "MAX_OPEN_POSITIONS",
            3,
        )

        self.same_symbol_open_cooldown_sec = self._env_float(
            "SAME_SYMBOL_OPEN_COOLDOWN_SEC",
            300.0,
        )
        # Dedicated cooldown after a stop close. Separate from generic last-open
        # cooldown so TP/manual exits keep their existing behavior. OFF by default.
        self.short_stop_reentry_cooldown_sec = max(
            0.0, self._env_float("SHORT_STOP_REENTRY_COOLDOWN_SEC", 0.0)
        )

        # single-position sizing + drawdown governor
        # One SHORT engine owns at most one open position for its symbol.
        self.short_normal_position_equity_pct = max(
            0.0, min(1.0, self._env_float("SHORT_NORMAL_POSITION_EQUITY_PCT", 1.0))
        )
        self.short_reduced_position_equity_pct = max(
            0.0, min(1.0, self._env_float("SHORT_REDUCED_POSITION_EQUITY_PCT", 0.50))
        )
        self.short_dd_reduce_trigger_pct = max(0.0, self._env_float("SHORT_DD_REDUCE_TRIGGER_PCT", 0.08))
        self.short_dd_recover_trigger_pct = max(0.0, self._env_float("SHORT_DD_RECOVER_TRIGGER_PCT", 0.04))
        self.risk_oi_confirmed_build_enable = self._env_bool(
            "RISK_OI_CONFIRMED_BUILD_ENABLE", False
        )
        self.risk_oi_confirmed_build_multiplier = max(
            0.0, min(1.0, self._env_float("RISK_OI_CONFIRMED_BUILD_MULTIPLIER", 0.50))
        )
        # Entry-time MARKET CONTEXT shadow telemetry. This never changes the
        # order decision and is OFF by default. The snapshot is fetched only
        # after the order attempt, using a frozen confirmation-minute cutoff.
        self.short_entry_context_shadow_enable = self._env_bool(
            "SHORT_ENTRY_CONTEXT_SHADOW_ENABLE", False
        )
        self.short_entry_context_shadow_stream = str(
            os.getenv("SHORT_ENTRY_CONTEXT_SHADOW_STREAM", "telemetry:market_context_shadow")
            or "telemetry:market_context_shadow"
        ).strip()
        self.short_entry_context_shadow_maxlen = max(
            1000, self._env_int("SHORT_ENTRY_CONTEXT_SHADOW_MAXLEN", 20000)
        )
        self._short_equity_peak = 0.0
        self._short_risk_mode = "NORMAL"

        self._symbol_open_locks: Dict[str, float] = {}
        self._short_stop_reentry_until: Dict[str, float] = {}
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
            -0.0900,
        )

        # -----------------------------------------------------
        # SHORT dynamic-stop confirmation
        # -----------------------------------------------------
        self.short_dynamic_stop_enable = self._env_bool(
            "SHORT_DYNAMIC_STOP_ENABLE",
            True,
        )
        self.short_dynamic_stop_confirm_sec = max(
            0.0,
            self._env_float("SHORT_DYNAMIC_STOP_CONFIRM_SEC", 60.0),
        )
        self.short_dynamic_stop_catastrophic_pct = max(
            0.0,
            self._env_float("SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT", 0.045),
        )

        # -----------------------------------------------------
        # telemetry / outcome feedback
        # -----------------------------------------------------
        self.short_outcome_feedback_enable = self._env_bool(
            "SHORT_OUTCOME_FEEDBACK_ENABLE",
            True,
        )
        self.short_outcome_feedback_stream = str(
            os.getenv("SHORT_OUTCOME_FEEDBACK_STREAM", "telemetry:short_outcomes")
            or "telemetry:short_outcomes"
        ).strip()
        self.short_outcome_feedback_maxlen = max(100, self._env_int(
            "SHORT_OUTCOME_FEEDBACK_MAXLEN",
            5000,
        ))
        # Telemetry-only fee estimate. Lifecycle decisions remain gross until
        # fee-aware exit behavior is explicitly validated offline.
        self.short_estimated_round_trip_fee_pct = max(
            0.0,
            self._env_float("SHORT_ESTIMATED_ROUND_TRIP_FEE_PCT", 0.0008),
        )

        self.short_entry_candle_service = (
            ShortEntryCandleService(
                client=self.client,
                logger=self.logger,
            )
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
                    "sizing=%s pct=%.4f "
                    "base_notional=%.4f max_notional=%.4f "
                    "max_leverage=%.2f default_leverage=%s "
                    "model_gate=upstream_master_only "
                    "sniper_interval=%s sniper_window=%.3f..%.3f "
                    "confirmation_interval=%s reversal_min=%.6f "
                    "fixed_roi=%s tp=%.6f giveback=%.6f "
                    "hard_stop=%.6f",
                    self.dry_run,
                    self.short_sizing_mode,
                    self.per_position_balance_pct,
                    self.base_order_notional,
                    self.max_position_notional,
                    self.max_leverage,
                    self.default_order_leverage,
                    self.sniper_entry_interval,
                    self.sniper_entry_min_candle_progress,
                    self.sniper_entry_max_candle_progress,
                    self.entry_candle_dir_interval,
                    self.short_current_reversal_min_pct,
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

    # FAST ENTRY KLINE
    # =========================================================







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
        return (
            self.short_entry_candle_service
            .get_pair(
                symbol,
                interval,
            )
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

        min_reversal_pct = float(self.short_current_reversal_min_pct)

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
                    max_age_sec=float(self.short_price_cache_max_age_sec),
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
                    max_age_sec=float(self.short_price_cache_max_age_sec),
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

        # Prefer the shared live price caches. This avoids a Binance REST call on
        # every lifecycle tick when the central market-data path is healthy.
        cached = self._get_cached_mid_price(sym_u)
        if cached is not None and cached > 0.0:
            return float(cached)

        try:
            fn = getattr(self.client, "futures_mark_price", None)
            if callable(fn):
                out = fn(symbol=sym_u)
                if isinstance(out, dict):
                    px = self._clip_float(out.get("markPrice"), None)
                    if px is not None and px > 0.0:
                        return float(px)
        except Exception:
            pass

        px = self._clip_float(fallback_price, None)
        return float(px) if px is not None and px > 0.0 else None

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
        market_lot = None
        lot = None

        if isinstance(filters, list):
            for item in filters:
                if not isinstance(item, dict):
                    continue

                ftype = str(item.get("filterType") or "").upper()
                if ftype == "MARKET_LOT_SIZE":
                    market_lot = item
                elif ftype == "LOT_SIZE":
                    lot = item
                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    try:
                        min_notional = max(
                            min_notional,
                            float(item.get("notional") or item.get("minNotional") or 0.0),
                        )
                    except Exception:
                        pass

        # MARKET orders use MARKET_LOT_SIZE when Binance provides it;
        # LOT_SIZE is only the fallback. Never merge two distinct filters.
        qty_filter = market_lot if isinstance(market_lot, dict) else lot
        if isinstance(qty_filter, dict):
            try:
                step = float(qty_filter.get("stepSize") or 0.0)
            except Exception:
                step = 0.0
            try:
                min_qty = float(qty_filter.get("minQty") or 0.0)
            except Exception:
                min_qty = 0.0

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

    def _set_short_stop_reentry_lock(self, *, symbol: str, now_ts: Optional[float] = None) -> float:
        """Arm stop-only same-symbol re-entry lock; return expiry epoch."""
        sym_u = str(symbol or "").upper().strip()
        sec = max(0.0, float(self.short_stop_reentry_cooldown_sec or 0.0))
        if not sym_u or sec <= 0.0:
            return 0.0
        now_f = float(now_ts if now_ts is not None else time.time())
        until = now_f + sec
        self._short_stop_reentry_until[sym_u] = until
        try:
            if self.redis is not None:
                ttl = max(60, int(sec) + 60)
                self.redis.setex(f"risk:short:stop_reentry_until:{sym_u}", ttl, str(until))
        except Exception:
            pass
        return until

    def _get_short_stop_reentry_until(self, symbol: str) -> float:
        sym_u = str(symbol or "").upper().strip()
        local = float(self._short_stop_reentry_until.get(sym_u) or 0.0)
        remote = 0.0
        try:
            if self.redis is not None:
                remote = float(self.redis.get(f"risk:short:stop_reentry_until:{sym_u}") or 0.0)
        except Exception:
            remote = 0.0
        return max(local, remote)

    def _resolve_short_open_risk_state(self, *, symbol: str, now_ts: float, signal_score: float = 0.0, model_threshold: float = 0.0) -> Optional[Dict[str, Any]]:
        del signal_score
        del model_threshold
        sym_u = str(symbol or "").upper().strip()

        # Single-position authority: never add to an already-open symbol.
        # A new entry is allowed only after the previous position is fully closed.
        existing = self._get_position(sym_u)
        if isinstance(existing, dict):
            return {"status": "skip", "reason": "position_exists", "symbol": sym_u}

        stop_reentry_until = self._get_short_stop_reentry_until(sym_u)
        if stop_reentry_until > float(now_ts):
            return {
                "status": "skip",
                "reason": "stop_reentry_cooldown",
                "symbol": sym_u,
                "retry_after_sec": max(0.0, stop_reentry_until - float(now_ts)),
            }

        last_open_ts = 0.0
        try:
            if self.redis is not None:
                last_open_ts = float(self.redis.get(f"bot:last_open_ts:{sym_u}") or 0.0)
        except Exception:
            last_open_ts = 0.0
        if last_open_ts > 0.0 and (float(now_ts) - last_open_ts) < self.same_symbol_open_cooldown_sec:
            return {"status": "skip", "reason": "symbol_cooldown", "symbol": sym_u}

        if self._count_open_positions() >= self.max_open_positions:
            return {"status": "skip", "reason": "max_open_positions", "symbol": sym_u}
        return None

    def _get_short_equity_usdt_sync(self) -> float:
        client = getattr(self, "client", None)
        if client is None:
            return 0.0
        try:
            fn = getattr(client, "futures_account_balance", None)
            if callable(fn):
                rows = fn()
                if isinstance(rows, list):
                    for row in rows:
                        if str(row.get("asset") or "").upper() == "USDT":
                            return max(0.0, float(row.get("balance") or 0.0))
        except Exception:
            pass
        return 0.0

    def _get_short_available_balance_usdt_sync(self) -> float:
        client = getattr(self, "client", None)
        if client is None:
            return 0.0
        try:
            fn = getattr(client, "futures_account", None)
            if callable(fn):
                acc = fn()
                if isinstance(acc, dict):
                    return max(0.0, float(acc.get("availableBalance") or 0.0))
        except Exception:
            pass
        return 0.0

    def _resolve_short_drawdown_mode(self, equity: float) -> Dict[str, Any]:
        equity=max(0.0,float(equity))
        peak=max(float(self._short_equity_peak or 0.0),equity)
        mode=str(self._short_risk_mode or "NORMAL").upper()
        try:
            if self.redis is not None:
                peak=max(peak,float(self.redis.get("risk:short:equity_peak") or 0.0))
                raw=self.redis.get("risk:short:dd_mode")
                if raw is not None:
                    m=raw.decode() if isinstance(raw,(bytes,bytearray)) else str(raw)
                    if m.upper() in ("NORMAL","REDUCED"): mode=m.upper()
        except Exception:
            pass
        dd=max(0.0,(peak-equity)/peak) if peak>0 else 0.0
        if mode=="NORMAL" and dd>=self.short_dd_reduce_trigger_pct: mode="REDUCED"
        elif mode=="REDUCED" and dd<=self.short_dd_recover_trigger_pct: mode="NORMAL"
        self._short_equity_peak=peak; self._short_risk_mode=mode
        try:
            if self.redis is not None:
                self.redis.set("risk:short:equity_peak",str(peak)); self.redis.set("risk:short:dd_mode",mode)
        except Exception: pass
        return {"mode":mode,"peak":peak,"dd_pct":dd}

    def _resolve_short_position_risk(
        self,
        *,
        equity: float,
        market_context_multiplier: float,
    ) -> Dict[str, Any]:
        """Resolve SHORT position risk through the shared Risk Master contract.

        Drawdown state is still locally sourced during the current migration phase,
        but the combination rule is centralized in risk_decision.v1.
        """
        gov = self._resolve_short_drawdown_mode(float(equity))
        decision = evaluate_position_risk(
            market_context_multiplier=market_context_multiplier,
            base_mode=str(gov.get("mode") or "NORMAL"),
            normal_multiplier=self.short_normal_position_equity_pct,
            reduced_multiplier=self.short_reduced_position_equity_pct,
        ).to_dict()
        decision["drawdown_pct"] = float(gov.get("dd_pct") or 0.0)
        decision["equity_peak"] = float(gov.get("peak") or 0.0)
        return decision

    def _resolve_short_order_notional(
        self,
        *,
        symbol: str,
        risk_multiplier: float = 1.0,
    ) -> Optional[float]:
        del symbol
        min_n = float(self.short_min_order_notional_usdt)
        try:
            market_mult = max(0.0, min(1.0, float(risk_multiplier)))
        except Exception:
            market_mult = 1.0
        if market_mult <= 0.0:
            return None

        if self.short_sizing_mode != "equity_single":
            max_n = float(self.max_position_notional)
            if max_n < min_n:
                return None
            if self.short_sizing_mode == "fixed":
                base_notional = max(min_n, min(float(self.base_order_notional), max_n))
            else:
                available = self._get_short_available_balance_usdt_sync()
                equity = self._get_short_equity_usdt_sync()
                base = available if self.use_available_balance_for_sizing and available > 0 else equity
                if base <= 0:
                    return None
                base_notional = max(min_n, min(base * self.per_position_balance_pct, max_n))
            decision = evaluate_position_risk(
                market_context_multiplier=market_mult,
                base_mode="NORMAL",
                normal_multiplier=1.0,
                reduced_multiplier=1.0,
            )
            notional = float(base_notional) * float(decision.final_size_multiplier)
            return None if notional < min_n else float(notional)

        equity = self._get_short_equity_usdt_sync()
        if equity <= 0:
            return None
        decision = self._resolve_short_position_risk(
            equity=equity,
            market_context_multiplier=market_mult,
        )
        if str(decision.get("state") or "") != "READY":
            return None
        notional = float(equity) * float(decision.get("final_size_multiplier") or 0.0)
        if notional < min_n:
            return None
        return float(notional)

    def _prepare_short_open_order(
        self,
        *,
        symbol: str,
        fallback_price: Any = None,
        risk_multiplier: float = 1.0,
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

        notional = self._resolve_short_order_notional(symbol=sym_u, risk_multiplier=risk_multiplier)

        if notional is None or float(notional) <= 0.0:
            return {
                "ok": False,
                "result": {
                    "status": "skip",
                    "reason": "sizing_unavailable",
                    "symbol": sym_u,
                },
            }

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
            "risk_multiplier": max(0.0, min(1.0, float(risk_multiplier))),
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
        entry_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Açılan SHORT pozisyonu yerel store ve Redis'e kaydeder.

        Bu helper market emri göndermez.
        """
        sym_u = str(symbol or "").upper().strip()

        meta0 = dict(entry_meta) if isinstance(entry_meta, dict) else {}
        existing = self._get_position(sym_u)
        if isinstance(existing, dict):
            # Final-safety should prevent this path. Never merge/average positions.
            raise RuntimeError(f"single_position_violation:{sym_u}")

        pos = {
            "symbol": sym_u,
            "side": "short",
            "qty": float(qty),
            "entry_price": float(order_price),
            "notional": float(qty) * float(order_price),
            "interval": str(interval or self.entry_interval),
            "leverage": int(leverage),
            "created_ts": float(now_ts),
            "best_roi_pct": 0.0,
            "position_mode": "single",
        }
        pos["model_provider"] = str(meta0.get("model_provider") or "short_coin_specific")
        pos["signal_score"] = float(self._clip_float(meta0.get("signal_score"), 0.0) or 0.0)
        pos["open_min_score"] = float(self._clip_float(meta0.get("open_min_score"), 0.0) or 0.0)

        self._set_position(
            sym_u,
            pos,
        )

        try:
            if self.redis is not None:
                cooldown_ttl = max(3600, int(self.same_symbol_open_cooldown_sec) + 60)
                self.redis.setex(
                    f"bot:last_open_ts:{sym_u}",
                    cooldown_ttl,
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
            "position_mode": "single",
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

        if now_ts - lock_ts < float(self.short_open_lock_sec):
            return {
                "status": "skip",
                "reason": "open_lock",
                "symbol": sym_u,
            }

        self._symbol_open_locks[sym_u] = now_ts

        try:
            final_safety = self._evaluate_short_final_safety_stage(
                symbol=sym_u,
                now_ts=now_ts,
                signal_score=float(meta0.get("signal_score") or 0.0),
                model_threshold=float(meta0.get("open_min_score") or meta0.get("model_score_gate_threshold") or 0.0),
            )
            if final_safety["state"] != "READY":
                result = final_safety.get("result")
                if isinstance(result, dict):
                    return result
                return {
                    "status": "skip",
                    "reason": str(final_safety.get("reason") or "final_safety"),
                    "symbol": sym_u,
                }

            market_context0, risk_context_source = self._select_short_risk_market_context(meta0)
            meta0["risk_context_source"] = str(risk_context_source)
            market_risk_multiplier, market_risk_reasons = resolve_market_context_multiplier(
                market_context=market_context0,
                oi_confirmed_build_enable=self.risk_oi_confirmed_build_enable,
                oi_confirmed_build_multiplier=self.risk_oi_confirmed_build_multiplier,
            )
            meta0["market_risk_multiplier"] = float(market_risk_multiplier)
            meta0["market_risk_reasons"] = list(market_risk_reasons)

            order_plan = self._prepare_short_open_order(
                symbol=sym_u,
                fallback_price=meta0.get("price"),
                risk_multiplier=float(market_risk_multiplier),
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
                entry_meta=meta0,
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

    def _emit_short_outcome_feedback(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
        exit_price: float,
        pnl_pct: float,
        roi_pct: float,
        reason: str,
    ) -> None:
        """Bounded structured outcome record for later analysis/retraining."""
        if not self.short_outcome_feedback_enable:
            return

        now_ts = time.time()
        created_ts = self._clip_float(position.get("created_ts"), now_ts) or now_ts
        fee_pct = max(0.0, float(self.short_estimated_round_trip_fee_pct))
        leverage = max(1.0, float(self._clip_float(position.get("leverage"), self.default_order_leverage) or 1.0))
        estimated_net_pnl_pct = float(pnl_pct) - fee_pct
        estimated_net_roi_pct = float(roi_pct) - (fee_pct * leverage)
        record = {
            "ts": float(now_ts),
            "symbol": str(symbol or "").upper().strip(),
            "side": "short",
            "model_provider": str(position.get("model_provider") or "hybrid_coin_specific"),
            "signal_score": float(self._clip_float(position.get("signal_score"), 0.0) or 0.0),
            "open_min_score": float(self._clip_float(position.get("open_min_score"), 0.0) or 0.0),
            "entry_price": float(self._clip_float(position.get("entry_price"), 0.0) or 0.0),
            "exit_price": float(exit_price),
            # Backward-compatible gross fields.
            "pnl_pct": float(pnl_pct),
            "roi_pct": float(roi_pct),
            "gross_pnl_pct": float(pnl_pct),
            "gross_roi_pct": float(roi_pct),
            "estimated_round_trip_fee_pct": float(fee_pct),
            "estimated_net_pnl_pct": float(estimated_net_pnl_pct),
            "estimated_net_roi_pct": float(estimated_net_roi_pct),
            "estimated_net_pnl_bps": float(estimated_net_pnl_pct * 10000.0),
            "duration_sec": max(0.0, float(now_ts) - float(created_ts)),
            "reason": str(reason or "unknown"),
        }

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][OUTCOME] %s",
                    json.dumps(record, separators=(",", ":"), sort_keys=True),
                )
        except Exception:
            pass

        try:
            if self.redis is not None and self.short_outcome_feedback_stream:
                self.redis.xadd(
                    self.short_outcome_feedback_stream,
                    {
                        "symbol": record["symbol"],
                        "json": json.dumps(record, separators=(",", ":"), sort_keys=True),
                    },
                    maxlen=int(self.short_outcome_feedback_maxlen),
                    approximate=True,
                )
        except Exception as exc:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][OUTCOME][WARN] symbol=%s err=%s",
                        record["symbol"], str(exc)[:160],
                    )
            except Exception:
                pass

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

        self._emit_short_outcome_feedback(
            symbol=sym_u,
            position=position,
            exit_price=float(exit_price),
            pnl_pct=float(pnl_pct),
            roi_pct=float(roi_pct),
            reason=str(reason),
        )

        self._del_position(sym_u)

        if str(reason or "").strip().lower() in {
            "dynamic_stop_confirmed",
            "dynamic_stop_catastrophic",
            "roi_hard_stop",
        }:
            self._set_short_stop_reentry_lock(symbol=sym_u)

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
        if exit_price is None or exit_price <= 0.0:
            exit_price = self._resolve_order_price(
                symbol=sym_u,
                fallback_price=pos.get("mark_price"),
            )
        if exit_price is None or exit_price <= 0.0:
            return {
                "status": "skip",
                "reason": "missing_close_price",
                "symbol": sym_u,
            }

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
        market_context: Optional[Dict[str, Any]] = None,
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
            "market_context": dict(market_context) if isinstance(market_context, dict) else {},
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

    def _store_short_pending_from_stage(
        self,
        *,
        symbol: str,
        signal_score: float,
        min_score: float,
        price: float,
        market_context: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Single pending-write path for SNIPER and ENTRY CONFIRMATION."""
        return self._create_short_pending(
            symbol=symbol,
            signal_score=signal_score,
            min_score=min_score,
            price=price,
            market_context={
                "provider": market_context.get("provider"),
                "probs": dict(market_context.get("probs") or {}),
                "extra": dict(extra or {}),
            },
        )

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

        if candle_age > int(self.short_pending_max_candle_age):
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
            "stop_reentry_cooldown",
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

    def _handle_short_candidate_result(
        self,
        *,
        symbol: str,
        candidate: Dict[str, Any],
        signal_score: float,
        min_score: float,
    ) -> bool:
        """
        Pending SHORT aday sonucu allow değilse
        cancel veya retry davranışını uygular.

        True dönerse caller işlemi sonlandırmalıdır.
        """
        if candidate.get("allow", False):
            return False

        reason = str(
            candidate.get("reason")
            or "candidate_block"
        ).strip().lower()

        retryable = bool(
            candidate.get("retryable", False)
        )

        if not retryable:
            self._cancel_short_pending(
                symbol=symbol,
                reason=reason,
                signal_score=signal_score,
                min_score=min_score,
            )

        else:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-WAIT] "
                        "symbol=%s reason=%s "
                        "retryable=true",
                        symbol,
                        reason,
                    )
            except Exception:
                pass

        return True

    def _check_short_pending_window(
        self,
        *,
        symbol: str,
        pending: Dict[str, Any],
        current_candle_id: int,
        progress: float,
        max_pending_progress: float,
    ) -> bool:
        """
        Pending SHORT kaydının mevcut mum
        penceresinde işlenebilir olup olmadığını
        kontrol eder.
        """
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
            self._pop_short_pending(symbol)

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-CANCEL] "
                        "symbol=%s "
                        "reason=invalid_candle_id",
                        symbol,
                    )
            except Exception:
                pass

            return False

        if window_state == "expired":
            self._pop_short_pending(symbol)

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-EXPIRE] "
                        "symbol=%s created_candle=%s "
                        "current_candle=%s age=%s",
                        symbol,
                        created_candle_id,
                        current_candle_id,
                        candle_age,
                    )
            except Exception:
                pass

            return False

        return window_state == "ready"

    def _prepare_short_pending_candidate(
        self,
        *,
        symbol: str,
        pending: Dict[str, Any],
    ):
        """Revalidate one pending SHORT through the same active gate authorities."""
        pending_score = float(pending.get("score") or 0.0)
        pending_min_score = float(pending.get("min_score") or 0.0)
        stored_context = pending.get("market_context") if isinstance(pending.get("market_context"), dict) else {}
        provider = str(stored_context.get("provider") or "hybrid_coin_specific").strip().lower()
        stored_extra = dict(stored_context.get("extra") or {})
        stored_extra.setdefault("model_provider", provider)
        revalidated_context = self._build_short_market_context_stage(
            probs=stored_context.get("probs") if isinstance(stored_context.get("probs"), dict) else {},
            extra=stored_extra,
        )
        if revalidated_context["state"] != "READY":
            candidate = self._make_short_entry_decision(
                allow=False, action="block",
                reason=str(revalidated_context.get("reason") or "market_context"),
                signal_score=pending_score, required_score=pending_min_score, retryable=False,
            )
            return float(pending.get("price") or 0.0), pending_score, pending_min_score, candidate

        pending_price = self._resolve_short_pending_price(
            symbol=symbol,
            pending=pending,
        )
        if pending_price is None:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][PENDING-WAIT] symbol=%s reason=missing_price",
                        symbol,
                    )
            except Exception:
                pass
            return None

        score_gate = self._evaluate_short_model_score_gate_stage(
            symbol=symbol,
            market_context=revalidated_context,
        )
        pending_score = float(score_gate.get("signal_score") or 0.0)
        pending_min_score = float(score_gate.get("final_min_score") or 0.0)
        if score_gate["state"] not in ("READY", "PENDING_CONTEXT"):
            candidate = self._make_short_entry_decision(
                allow=False,
                action="block",
                reason=str(score_gate.get("reason") or "model_score_gate"),
                signal_score=pending_score,
                required_score=pending_min_score,
                retryable=False,
            )
            return float(pending_price), pending_score, pending_min_score, candidate

        sniper = self._evaluate_short_sniper_stage(
            symbol=symbol,
            signal_score=pending_score,
            final_min_score=pending_min_score,
            allow_pending=False,
        )
        if sniper["state"] != "READY":
            candidate = sniper.get("decision") or self._make_short_entry_decision(
                allow=False,
                action="block",
                reason=str(sniper.get("reason") or "pending_sniper"),
                signal_score=pending_score,
                required_score=pending_min_score,
                progress=sniper.get("progress"),
                retryable=False,
            )
            return float(pending_price), pending_score, pending_min_score, candidate

        confirmation = self._evaluate_short_entry_confirmation_stage(
            symbol=symbol,
            signal_score=pending_score,
            final_min_score=pending_min_score,
            progress=float(sniper["progress"]),
        )
        if confirmation["state"] != "READY":
            candidate = confirmation.get("decision") or self._make_short_entry_decision(
                allow=False,
                action="block",
                reason=str(confirmation.get("reason") or "entry_confirmation"),
                signal_score=pending_score,
                required_score=pending_min_score,
                progress=float(sniper["progress"]),
                retryable=True,
            )
            return float(pending_price), pending_score, pending_min_score, candidate

        risk = self._evaluate_short_execution_risk_stage(
            symbol=symbol,
            signal_score=pending_score,
            model_threshold=pending_min_score,
        )
        if risk["state"] != "READY":
            reason = str(risk.get("reason") or "execution_risk")
            retryable = reason in {"symbol_cooldown", "stop_reentry_cooldown", "max_open_positions", "open_lock"}
            candidate = self._make_short_entry_decision(
                allow=False,
                action="block",
                reason=reason,
                signal_score=pending_score,
                required_score=pending_min_score,
                progress=float(sniper["progress"]),
                retryable=retryable,
            )
            return float(pending_price), pending_score, pending_min_score, candidate

        candidate = self._make_short_entry_decision(
            allow=True,
            action="open",
            reason="pending_revalidated",
            signal_score=pending_score,
            required_score=pending_min_score,
            progress=float(sniper["progress"]),
            retryable=False,
        )
        return float(pending_price), pending_score, pending_min_score, candidate

    def _prepare_short_pending_open_payload(
        self,
        *,
        symbol: str,
        pending: Dict[str, Any],
        pending_price: float,
        pending_score: float,
        pending_min_score: float,
        candidate: Dict[str, Any],
        progress: float,
        created_candle_id: int,
        current_candle_id: int,
    ) -> None:
        """
        Yeniden doğrulanan SHORT pending kaydını
        pozisyon açılışına hazırlar.
        """
        pending["price"] = float(
            pending_price
        )
        pending["signal_score"] = float(
            pending_score
        )
        pending["open_min_score"] = float(
            pending_min_score
        )
        pending["entry_interval"] = str(
            self.entry_interval
        )
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
                    symbol,
                    progress,
                    created_candle_id,
                    current_candle_id,
                    pending_score,
                    pending_min_score,
                    candidate.get("reason"),
                )
        except Exception:
            pass

    async def _execute_short_pending_open(
        self,
        *,
        symbol: str,
        pending: Dict[str, Any],
    ) -> None:
        """
        SHORT pending kaydını worker thread içinde
        açar ve sonucu pending yöneticisine iletir.
        """
        shadow_kline_rows = self.short_entry_candle_service.get_last_rows(
            symbol, self.entry_candle_dir_interval
        )

        result = await asyncio.to_thread(
            self.open_position_from_signal,
            symbol,
            "short",
            self.entry_interval,
            pending,
        )

        self._schedule_short_context_shadow(
            symbol=symbol,
            payload=pending,
            confirmation_open_ms=int(pending.get("shadow_confirmation_open_ms") or self._short_confirmation_open_ms()),
            entry_path="pending_revalidated",
            kline_rows=shadow_kline_rows,
        )

        self._handle_short_pending_open_result(
            symbol=symbol,
            result=result,
        )

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

        if not self._check_short_pending_window(
            symbol=sym_u,
            pending=pending,
            current_candle_id=current_candle_id,
            progress=progress,
            max_pending_progress=max_pending_progress,
        ):
            return

        prepared_candidate = (
            self._prepare_short_pending_candidate(
                symbol=sym_u,
                pending=pending,
            )
        )

        if prepared_candidate is None:
            return

        (
            pending_price,
            pending_score,
            pending_min_score,
            candidate,
        ) = prepared_candidate

        if self._handle_short_candidate_result(
            symbol=sym_u,
            candidate=candidate,
            signal_score=pending_score,
            min_score=pending_min_score,
        ):
            return

        pending["shadow_confirmation_open_ms"] = self._short_confirmation_open_ms()
        pending_rows = self.short_entry_candle_service.get_last_rows(
            sym_u, self.entry_candle_dir_interval
        )
        pending["entry_market_context"] = self._build_short_entry_market_context(
            symbol=sym_u,
            confirmation_open_ms=int(pending["shadow_confirmation_open_ms"]),
            kline_rows=pending_rows,
        )
        soft_context_gate = self._evaluate_short_soft_context_revalidation_stage(
            payload=pending,
            entry_market_context=pending.get("entry_market_context"),
        )
        if soft_context_gate["state"] != "READY":
            self._cancel_short_pending(
                symbol=sym_u,
                reason=str(soft_context_gate.get("reason") or "soft_context"),
                signal_score=pending_score,
                min_score=pending_min_score,
            )
            return
        score_contract = self._extract_short_score_contract(pending)
        if str(score_contract.get("model_score_gate_state") or "").strip().upper() == "PENDING_CONTEXT":
            pending["model_score_gate_promoted"] = True
            pending["model_score_gate_promotion_reason"] = str(soft_context_gate.get("reason") or "")

        self._prepare_short_pending_open_payload(
            symbol=sym_u,
            pending=pending,
            pending_price=pending_price,
            pending_score=pending_score,
            pending_min_score=pending_min_score,
            candidate=candidate,
            progress=progress,
            created_candle_id=int(
                pending.get("candle_id")
                or 0
            ),
            current_candle_id=current_candle_id,
        )

        await self._execute_short_pending_open(
            symbol=sym_u,
            pending=pending,
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

        # PENDING revalidation uses the same SNIPER window authority as direct entry.
        max_pending_progress = float(self.sniper_entry_max_candle_progress)

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

    # =========================================================
    # TARGET SHORT ENTRY PIPELINE
    # MODEL -> SIGNAL NORMALIZATION -> MARKET CONTEXT
    # -> MODEL SCORE GATE -> SNIPER -> ENTRY CONFIRMATION
    # -> EXECUTION RISK -> FINAL SAFETY -> ORDER
    # =========================================================

    def _normalize_short_signal_stage(
        self,
        *,
        symbol: str,
        signal: str,
    ) -> Dict[str, Any]:
        """Normalize incoming signal into the SHORT-only execution domain."""
        sym_u = str(symbol or "").upper().strip()
        signal_u = str(signal or "").lower().strip()

        if not sym_u:
            return {
                "state": "REJECT",
                "reason": "empty_symbol",
                "symbol": "",
                "side": "",
            }

        if signal_u not in ("short", "sell", "bear", "down"):
            return {
                "state": "REJECT",
                "reason": "not_short_signal",
                "symbol": sym_u,
                "side": signal_u,
            }

        return {
            "state": "READY",
            "reason": "short_signal",
            "symbol": sym_u,
            "side": "short",
        }

    def _build_short_market_context_stage(
        self,
        *,
        probs: Optional[Dict[str, float]],
        extra: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Carry the canonical MARKET CONTEXT contract into execution.

        In orchestrated production, the hard MARKET CONTEXT decision belongs to
        MasterExecutor. This stage does not invent a second context veto.
        """
        extra0 = dict(extra) if isinstance(extra, dict) else {}
        provider = str(extra0.get("model_provider") or "hybrid_coin_specific").strip().lower()
        source = str(extra0.get("execution_source") or "local").strip().lower()
        if source == "orchestrated" and not bool(extra0.get("market_context_passed", False)):
            return {
                "state": "REJECT",
                "reason": "upstream_market_context_gate_missing",
                "provider": provider,
                "probs": dict(probs) if isinstance(probs, dict) else {},
                "extra": extra0,
            }
        return {
            "state": "READY",
            "reason": "upstream_market_context_pass" if source == "orchestrated" else "local_context_pass",
            "provider": provider,
            "probs": dict(probs) if isinstance(probs, dict) else {},
            "extra": extra0,
        }

    def _evaluate_short_model_score_gate_stage(
        self,
        *,
        symbol: str,
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Consume the upstream MODEL SCORE GATE without creating a second threshold.

        READY is a normal hard-score pass. PENDING_CONTEXT is an explicit
        upstream provisional SHORT band that must be revalidated with the exact
        entry_market_context.v1 after ENTRY CONFIRMATION.
        """
        del symbol
        provider = str(market_context.get("provider") or "hybrid_coin_specific").strip().lower()
        extra0 = dict(market_context.get("extra") or {})
        execution_source = str(extra0.get("execution_source") or "local").strip().lower()

        try:
            signal_score = float(extra0.get("model_score"))
        except Exception:
            signal_score = 0.0
        try:
            threshold = float(extra0.get("model_score_gate_threshold") or 0.0)
        except Exception:
            threshold = 0.0
        try:
            soft_threshold = float(extra0.get("model_score_gate_soft_threshold") or 0.0)
        except Exception:
            soft_threshold = 0.0
        gate_state = str(extra0.get("model_score_gate_state") or "").strip().upper()
        if not gate_state:
            gate_state = "READY" if bool(extra0.get("model_score_gate_passed", False)) else "REJECT"

        base = {
            "provider": provider,
            "extra": extra0,
            "signal_score": signal_score,
            "coin_min_score": threshold,
            "final_min_score": threshold,
            "soft_min_score": soft_threshold,
        }

        if execution_source != "orchestrated":
            return {"state": "REJECT", "reason": "upstream_model_score_gate_required", **base}

        if not (0.0 <= signal_score <= 1.0):
            return {"state": "REJECT", "reason": "invalid_upstream_model_score", **base}

        if gate_state == "PENDING_CONTEXT" and bool(extra0.get("soft_context_required", False)):
            if soft_threshold <= 0.0 or not (soft_threshold <= signal_score < threshold):
                return {"state": "REJECT", "reason": "invalid_soft_score_contract", **base}
            base["final_min_score"] = soft_threshold
            return {
                "state": "PENDING_CONTEXT",
                "reason": "upstream_soft_score_pending_context",
                **base,
            }

        if gate_state != "READY" or not bool(extra0.get("model_score_gate_passed", False)):
            return {"state": "REJECT", "reason": "upstream_model_score_gate_missing", **base}

        return {
            "state": "READY",
            "reason": "upstream_model_score_gate_pass",
            **base,
        }

    def _extract_short_score_contract(self, payload: Any) -> Dict[str, Any]:
        """Return score-gate metadata from direct or pending payload shapes."""
        obj = payload if isinstance(payload, dict) else {}
        if "model_score_gate_state" in obj or "soft_context_required" in obj:
            return obj
        wrapper = obj.get("market_context") if isinstance(obj.get("market_context"), dict) else {}
        extra = wrapper.get("extra") if isinstance(wrapper.get("extra"), dict) else {}
        return extra

    def _evaluate_short_soft_context_revalidation_stage(
        self,
        *,
        payload: Any,
        entry_market_context: Any,
    ) -> Dict[str, Any]:
        """Promote only an upstream PENDING_CONTEXT whose entry context is SHORT build."""
        score_contract = self._extract_short_score_contract(payload)
        gate_state = str(score_contract.get("model_score_gate_state") or "READY").strip().upper()
        soft_required = bool(score_contract.get("soft_context_required", False))

        if gate_state != "PENDING_CONTEXT" and not soft_required:
            return {"state": "READY", "reason": "hard_score_context_not_required"}
        if gate_state != "PENDING_CONTEXT" or not soft_required:
            return {"state": "REJECT", "reason": "invalid_soft_score_state"}

        ctx = entry_market_context if isinstance(entry_market_context, dict) else {}
        if not bool(ctx.get("available", False)):
            return {"state": "REJECT", "reason": "soft_entry_context_unavailable"}
        regime = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
        if (
            str(regime.get("name") or "").strip().lower() == "confirmed_build"
            and str(regime.get("build_direction") or "").strip().lower() == "short"
        ):
            return {"state": "READY", "reason": "soft_short_build_confirmed"}
        return {"state": "REJECT", "reason": "soft_short_build_not_confirmed"}

    def _evaluate_short_sniper_stage(
        self,
        *,
        symbol: str,
        signal_score: float,
        final_min_score: float,
        allow_pending: bool = True,
    ) -> Dict[str, Any]:
        """Single SNIPER authority; maps timing directly to READY/PENDING/REJECT."""
        progress = self._calc_candle_progress(self.sniper_entry_interval)

        if progress > self.sniper_entry_max_candle_progress:
            action = "pending" if allow_pending else "block"
            reason = "sniper_window_passed" if allow_pending else "pending_window_passed"
            decision = self._make_short_entry_decision(
                allow=False, action=action, reason=reason,
                signal_score=signal_score, required_score=final_min_score,
                progress=progress, retryable=bool(allow_pending),
            )
            return {
                "state": "PENDING" if allow_pending else "REJECT",
                "reason": reason, "progress": float(progress),
                "decision": decision,
            }

        if progress < float(self.sniper_entry_min_candle_progress):
            decision = self._make_short_entry_decision(
                allow=False, action="block", reason="sniper_window_early",
                signal_score=signal_score, required_score=final_min_score,
                progress=progress, retryable=True,
            )
            return {
                "state": "REJECT", "reason": "sniper_window_early",
                "progress": float(progress), "decision": decision,
            }

        return {"state": "READY", "reason": "sniper_ready", "progress": float(progress)}

    def _evaluate_short_entry_confirmation_stage(
        self,
        *,
        symbol: str,
        signal_score: float,
        final_min_score: float,
        progress: float,
    ) -> Dict[str, Any]:
        """Single ENTRY CONFIRMATION authority for the SHORT reversal candle."""
        if self._check_short_candle_direction(symbol=symbol):
            return {
                "state": "READY",
                "reason": "entry_confirmed",
                "progress": float(progress),
            }

        return {
            "state": "PENDING",
            "reason": "entry_confirmation_wait",
            "progress": float(progress),
            "decision": self._make_short_entry_decision(
                allow=False,
                action="pending",
                reason="entry_confirmation_wait",
                signal_score=signal_score,
                required_score=final_min_score,
                progress=progress,
                retryable=True,
            ),
        }

    def _build_short_entry_market_context(
        self,
        *,
        symbol: str,
        confirmation_open_ms: int,
        kline_rows: Any,
    ) -> Dict[str, Any]:
        """Build entry-time context from confirmation rows; no Binance I/O."""
        try:
            return build_entry_context_from_rows(
                redis_client=self.redis,
                symbol=symbol,
                rows=kline_rows,
                confirmation_open_ms=int(confirmation_open_ms),
            )
        except Exception as exc:
            return {
                "available": False,
                "schema_version": "entry_market_context.v1",
                "reason": "entry_context_exception",
                "error": str(exc)[:200],
                "symbol": str(symbol or "").upper().strip(),
            }

    def _extract_short_canonical_market_context(self, payload: Any) -> Dict[str, Any]:
        """Extract Master market_context.v1 from direct or pending payload shapes."""
        obj = payload if isinstance(payload, dict) else {}
        direct = obj.get("market_context")
        if isinstance(direct, dict) and str(direct.get("schema_version") or "").startswith("market_context."):
            return dict(direct)
        wrapper = obj.get("market_context") if isinstance(obj.get("market_context"), dict) else {}
        extra = wrapper.get("extra") if isinstance(wrapper.get("extra"), dict) else {}
        nested = extra.get("market_context")
        if isinstance(nested, dict):
            return dict(nested)
        return {}

    def _select_short_risk_market_context(
        self, payload: Any
    ) -> Tuple[Dict[str, Any], str]:
        """Prefer entry_market_context.v1; fall back to canonical Master context."""
        obj = payload if isinstance(payload, dict) else {}
        entry = obj.get("entry_market_context") if isinstance(obj.get("entry_market_context"), dict) else {}
        if bool(entry.get("available")):
            return dict(entry), "entry"
        return self._extract_short_canonical_market_context(obj), "master_fallback"

    def _short_confirmation_open_ms(self) -> int:
        interval = str(self.entry_candle_dir_interval or "1m")
        sec = max(1, int(self._interval_to_seconds(interval)))
        return int(self._get_candle_id(interval) * sec * 1000)

    def _run_short_context_shadow_sync(
        self,
        *,
        symbol: str,
        payload: Dict[str, Any],
        confirmation_open_ms: int,
        entry_path: str,
        kline_rows: Any = None,
    ) -> None:
        if not self.short_entry_context_shadow_enable:
            return
        try:
            master_context = self._extract_short_canonical_market_context(payload)
            event = build_shadow_event(
                client=self.client,
                redis_client=self.redis,
                symbol=symbol,
                master_context=master_context,
                confirmation_open_ms=int(confirmation_open_ms),
                model_score=float(payload.get("model_score", payload.get("signal_score", 0.0)) or 0.0),
                intent_id=str(payload.get("intent_id") or ""),
                entry_path=str(entry_path or "direct"),
                oi_policy_multiplier=float(self.risk_oi_confirmed_build_multiplier),
                kline_rows=kline_rows,
                entry_context_override=(payload.get("entry_market_context") if isinstance(payload.get("entry_market_context"), dict) else None),
            )
            sid = publish_shadow_event(
                redis_client=self.redis,
                event=event,
                stream=self.short_entry_context_shadow_stream,
                maxlen=self.short_entry_context_shadow_maxlen,
            )
            try:
                if self.logger:
                    cmp0 = event.get("comparison") if isinstance(event.get("comparison"), dict) else {}
                    self.logger.info(
                        "[EXEC-SHORT][CTX-SHADOW] symbol=%s path=%s transition=%s changed=%s stream_id=%s",
                        symbol, entry_path, cmp0.get("transition"), cmp0.get("changed"), sid,
                    )
            except Exception:
                pass
        except Exception as exc:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC-SHORT][CTX-SHADOW][WARN] symbol=%s err=%s",
                        symbol, str(exc)[:220],
                    )
            except Exception:
                pass

    def _schedule_short_context_shadow(
        self,
        *,
        symbol: str,
        payload: Dict[str, Any],
        confirmation_open_ms: int,
        entry_path: str,
        kline_rows: Any = None,
    ) -> None:
        """Fire-and-forget telemetry only; must never delay or veto ORDER."""
        if not self.short_entry_context_shadow_enable:
            return
        try:
            asyncio.create_task(
                asyncio.to_thread(
                    self._run_short_context_shadow_sync,
                    symbol=symbol,
                    payload=dict(payload or {}),
                    confirmation_open_ms=int(confirmation_open_ms),
                    entry_path=str(entry_path or "direct"),
                    kline_rows=list(kline_rows) if isinstance(kline_rows, list) else None,
                )
            )
        except Exception:
            return

    def _evaluate_short_execution_risk_stage(
        self,
        *,
        symbol: str,
        signal_score: float = 0.0,
        model_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Pre-order risk authority. Final safety rechecks atomically at order time."""
        result = self._resolve_short_open_risk_state(
            symbol=symbol,
            now_ts=time.time(),
            signal_score=signal_score,
            model_threshold=model_threshold,
        )
        if result is None:
            return {"state": "READY", "reason": "execution_risk_pass"}

        reason = str(result.get("reason") or "execution_risk")
        return {
            "state": "REJECT",
            "reason": reason,
            "result": result,
        }

    def _evaluate_short_final_safety_stage(
        self,
        *,
        symbol: str,
        now_ts: Optional[float] = None,
        signal_score: float = 0.0,
        model_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """FINAL SAFETY recheck immediately before ORDER submission."""
        result = self._resolve_short_open_risk_state(
            symbol=symbol,
            now_ts=float(time.time() if now_ts is None else now_ts),
            signal_score=signal_score,
            model_threshold=model_threshold,
        )
        if result is None:
            return {"state": "READY", "reason": "final_safety_pass"}
        return {
            "state": "REJECT",
            "reason": str(result.get("reason") or "final_safety"),
            "result": result,
        }

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

        normalized = self._normalize_short_signal_stage(
            symbol=symbol,
            signal=signal,
        )
        if normalized["state"] != "READY":
            return

        sym_u = str(normalized["symbol"])

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][ENTER] symbol=%s signal=%s",
                    sym_u,
                    signal,
                )
        except Exception:
            pass

        market_context = self._build_short_market_context_stage(
            probs=probs,
            extra=extra,
        )

        if market_context.get("state") != "READY":
            self._pop_short_pending(sym_u)
            return

        score_gate = self._evaluate_short_model_score_gate_stage(
            symbol=sym_u,
            market_context=market_context,
        )
        if score_gate["state"] not in ("READY", "PENDING_CONTEXT"):
            self._cancel_short_pending(
                symbol=sym_u,
                reason=str(score_gate.get("reason") or "model_score_gate"),
                signal_score=score_gate["signal_score"],
                min_score=score_gate["final_min_score"],
            )
            return

        signal_score = float(score_gate["signal_score"])
        final_min_score = float(score_gate["final_min_score"])
        extra0 = score_gate["extra"]

        sniper = self._evaluate_short_sniper_stage(
            symbol=sym_u,
            signal_score=signal_score,
            final_min_score=final_min_score,
            allow_pending=True,
        )

        if sniper["state"] == "PENDING":
            self._store_short_pending_from_stage(
                symbol=sym_u,
                signal_score=signal_score,
                min_score=final_min_score,
                price=price,
                market_context=market_context,
                extra=extra0,
            )
            return

        if sniper["state"] != "READY":
            return

        confirmation = self._evaluate_short_entry_confirmation_stage(
            symbol=sym_u,
            signal_score=signal_score,
            final_min_score=final_min_score,
            progress=float(sniper["progress"]),
        )
        if confirmation["state"] == "PENDING":
            self._store_short_pending_from_stage(
                symbol=sym_u,
                signal_score=signal_score,
                min_score=final_min_score,
                price=price,
                market_context=market_context,
                extra=extra0,
            )
            return

        if confirmation["state"] != "READY":
            return

        shadow_confirmation_open_ms = self._short_confirmation_open_ms()
        shadow_kline_rows = self.short_entry_candle_service.get_last_rows(
            sym_u, self.entry_candle_dir_interval
        )
        entry_market_context = self._build_short_entry_market_context(
            symbol=sym_u,
            confirmation_open_ms=shadow_confirmation_open_ms,
            kline_rows=shadow_kline_rows,
        )
        extra0["entry_market_context"] = dict(entry_market_context)

        soft_context_gate = self._evaluate_short_soft_context_revalidation_stage(
            payload=extra0,
            entry_market_context=entry_market_context,
        )
        if soft_context_gate["state"] != "READY":
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC-SHORT][SOFT-CONTEXT-REJECT] symbol=%s reason=%s",
                        sym_u, str(soft_context_gate.get("reason") or "soft_context"),
                    )
            except Exception:
                pass
            return
        if str(score_gate.get("state") or "") == "PENDING_CONTEXT":
            extra0["model_score_gate_promoted"] = True
            extra0["model_score_gate_promotion_reason"] = str(soft_context_gate.get("reason") or "")

        execution_risk = self._evaluate_short_execution_risk_stage(
            symbol=sym_u,
            signal_score=signal_score,
            model_threshold=final_min_score,
        )
        if execution_risk["state"] != "READY":
            return

        extra0 = self._prepare_short_open_payload(
            extra=extra0,
            price=price,
            signal_score=signal_score,
            final_min_score=final_min_score,
        )

        decision_ms = (time.perf_counter() - started_ts) * 1000.0

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

        # FINAL SAFETY is intentionally rechecked inside
        # open_position_from_signal() immediately before ORDER.
        await asyncio.to_thread(
            self.open_position_from_signal,
            sym_u,
            "short",
            self.entry_interval,
            extra0,
        )
        self._schedule_short_context_shadow(
            symbol=sym_u,
            payload=extra0,
            confirmation_open_ms=shadow_confirmation_open_ms,
            entry_path="direct",
            kline_rows=shadow_kline_rows,
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
                # Exchange can provide aggregate qty/entry but cannot reconstruct
                # our virtual layer history after a process restart. Until layer
                # state is persisted in the Stage-4 isolated state store, adopt
                # fail-closed and block further scale-ins for this position.
                pos["position_mode"] = "single"
                pos["adopted_from_exchange"] = True
                pos["layer_state"] = "exchange_adopted_unknown"
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

    def _resolve_short_lifecycle_position_state(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
    ):
        """
        Lifecycle kontrolü için geçerli fiyatı
        ve SHORT ROI değerini hazırlar.
        """
        price = self._resolve_order_price(
            symbol,
            position.get("mark_price"),
        )

        if price is None or price <= 0:
            return None

        roi = self._calc_short_roi(
            position,
            float(price),
        )

        return (
            float(price),
            float(roi),
        )

    def _update_short_best_roi(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
        roi: float,
    ):
        """
        SHORT pozisyonun en iyi ROI değerini
        günceller ve eski/yeni değeri döndürür.
        """
        old_best_roi = float(
            position.get("best_roi_pct")
            or 0.0
        )

        best_roi = max(
            old_best_roi,
            float(roi),
        )

        if best_roi > old_best_roi:
            position["best_roi_pct"] = float(
                best_roi
            )
            self._set_position(
                symbol,
                position,
            )

        return (
            old_best_roi,
            best_roi,
        )

    def _resolve_short_fixed_roi_tp_state(
        self,
        *,
        old_best_roi: float,
        best_roi: float,
    ):
        """
        Fixed ROI TP için arm, giveback ve
        trigger durumunu hazırlar.
        """
        if (
            not self.fixed_roi_tp_enable
            or float(best_roi)
            < float(self.fixed_roi_tp_pct)
        ):
            return None

        giveback = max(
            0.0,
            float(self.fixed_roi_tp_giveback),
        )

        trigger_roi = (
            float(best_roi)
            - giveback
        )

        armed_now = (
            float(old_best_roi)
            < float(self.fixed_roi_tp_pct)
            <= float(best_roi)
        )

        return {
            "armed_now": bool(armed_now),
            "giveback": float(giveback),
            "trigger_roi": float(trigger_roi),
        }

    def _evaluate_short_dynamic_stop_state(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
        price: float,
        roi: float,
    ) -> Optional[Dict[str, Any]]:
        """LONG-style confirmed stop adapted for SHORT direction.

        ROI_HARD_STOP_PCT is the soft trigger. Once breached, the position must
        remain beyond the trigger for SHORT_DYNAMIC_STOP_CONFIRM_SEC before
        closing. A larger adverse price move closes immediately via the
        catastrophic guard. Recovery above the soft trigger clears pending state.
        """
        sym_u = str(symbol or "").upper().strip()
        trigger = float(self.roi_hard_stop_pct)
        pending_ts = float(position.get("dynamic_stop_pending_ts") or 0.0)

        if float(roi) > trigger:
            if pending_ts > 0.0:
                position["dynamic_stop_pending_ts"] = 0.0
                position["dynamic_stop_pending_price"] = 0.0
                self._set_position(sym_u, position)
            return None

        if not self.short_dynamic_stop_enable:
            return {
                "state": "CLOSE",
                "reason": "roi_hard_stop",
                "price": float(price),
                "roi": float(roi),
            }

        entry = float(position.get("entry_price") or 0.0)
        adverse_price_pct = (float(price) - entry) / entry if entry > 0.0 else 0.0
        catastrophic = float(self.short_dynamic_stop_catastrophic_pct)
        if catastrophic > 0.0 and adverse_price_pct >= catastrophic:
            return {
                "state": "CLOSE",
                "reason": "dynamic_stop_catastrophic",
                "price": float(price),
                "roi": float(roi),
                "adverse_price_pct": float(adverse_price_pct),
            }

        now_ts = time.time()
        if pending_ts <= 0.0:
            position["dynamic_stop_pending_ts"] = float(now_ts)
            position["dynamic_stop_pending_price"] = float(price)
            self._set_position(sym_u, position)
            return {
                "state": "HOLD",
                "reason": "dynamic_stop_pending",
                "price": float(price),
                "roi": float(roi),
            }

        if (now_ts - pending_ts) >= float(self.short_dynamic_stop_confirm_sec):
            return {
                "state": "CLOSE",
                "reason": "dynamic_stop_confirmed",
                "price": float(price),
                "roi": float(roi),
            }

        return {
            "state": "HOLD",
            "reason": "dynamic_stop_pending",
            "price": float(price),
            "roi": float(roi),
        }

    def _evaluate_short_position_lifecycle_stage(
        self,
        *,
        symbol: str,
        position: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Single POSITION LIFECYCLE decision authority for an open SHORT."""
        lifecycle_state = self._resolve_short_lifecycle_position_state(
            symbol=symbol, position=position,
        )
        if lifecycle_state is None:
            return {"state": "HOLD", "reason": "missing_price"}

        price, roi = lifecycle_state
        stop_state = self._evaluate_short_dynamic_stop_state(
            symbol=symbol,
            position=position,
            price=float(price),
            roi=float(roi),
        )
        if isinstance(stop_state, dict):
            return stop_state

        old_best_roi, best_roi = self._update_short_best_roi(
            symbol=symbol, position=position, roi=roi,
        )
        tp_state = self._resolve_short_fixed_roi_tp_state(
            old_best_roi=old_best_roi, best_roi=best_roi,
        )

        if tp_state is None:
            return {
                "state": "HOLD", "reason": "lifecycle_hold",
                "price": float(price), "roi": float(roi),
                "best_roi": float(best_roi),
            }

        giveback = float(tp_state["giveback"])
        trigger_roi = float(tp_state["trigger_roi"])
        if tp_state["armed_now"] and self.logger:
            self.logger.info(
                "[EXEC-SHORT][TP-ARMED] symbol=%s roi=%.6f best_roi=%.6f target=%.6f giveback=%.6f trigger=%.6f",
                symbol, roi, best_roi, self.fixed_roi_tp_pct, giveback, trigger_roi,
            )

        if roi <= trigger_roi:
            if self.logger:
                self.logger.info(
                    "[EXEC-SHORT][TP-GIVEBACK] symbol=%s roi=%.6f best_roi=%.6f giveback=%.6f trigger=%.6f",
                    symbol, roi, best_roi, giveback, trigger_roi,
                )
            return {
                "state": "CLOSE", "reason": "fixed_roi_tp_giveback",
                "price": float(price), "roi": float(roi),
                "best_roi": float(best_roi),
            }

        return {
            "state": "HOLD", "reason": "tp_armed_hold",
            "price": float(price), "roi": float(roi),
            "best_roi": float(best_roi),
        }

    async def _close_short_from_lifecycle(
        self,
        *,
        symbol: str,
        price: float,
        reason: str,
    ):
        """
        Lifecycle kaynaklı SHORT kapanışını
        executor thread'inde gerçekleştirir.
        """
        await asyncio.to_thread(
            self.close_position,
            symbol,
            float(price),
            reason,
            "lifecycle",
        )

    async def _position_lifecycle_loop(self) -> None:
        while True:
            try:
                # PENDING REVALIDATION is an independent architecture stage.
                await self._process_short_pending_entries()

                if self.position_lifecycle_enabled:
                    for sym, pos in self._get_all_local_positions().items():
                        decision = self._evaluate_short_position_lifecycle_stage(
                            symbol=sym,
                            position=pos,
                        )
                        if decision.get("state") != "CLOSE":
                            continue

                        await self._close_short_from_lifecycle(
                            symbol=sym,
                            price=float(decision.get("price") or 0.0),
                            reason=str(decision.get("reason") or "lifecycle_close"),
                        )

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
