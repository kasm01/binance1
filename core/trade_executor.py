# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set

import redis


class TradeExecutor:
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
        self.client = client
        self.redis = redis_client
        self.logger = logger
        self.price_cache = price_cache
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.telegram_bot = telegram_bot

        self.dry_run = bool(dry_run)
        self.base_order_notional = float(base_order_notional or 120.0)
        self.max_position_notional = float(max_position_notional or 500.0)
        self.max_leverage = float(max_leverage or 3.0)

        self.default_order_leverage = int(
            float(os.getenv("DEFAULT_ORDER_LEVERAGE", str(int(self.max_leverage or 3))))
        )
        self.enable_dynamic_leverage = str(
            os.getenv("ENABLE_DYNAMIC_LEVERAGE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.exchange_info_ttl_sec = float(
            os.getenv("EXCHANGE_INFO_TTL_SEC", "300") or 300.0
        )
        self._exchange_info_cache: Optional[Dict[str, Any]] = None
        self._exchange_info_cache_ts: float = 0.0

        self.order_poll_status = str(
            os.getenv("ORDER_POLL_STATUS", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self.order_poll_wait_s = float(os.getenv("ORDER_POLL_WAIT_S", "2.0") or 2.0)

        self.order_verify_position = str(
            os.getenv("ORDER_VERIFY_POSITION", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.hedge_mode_enabled = True

        self.whale_block_actions = {"block", "avoid", "hard_block"}
        self.whale_reduce_actions = {"reduce", "scale_down", "soft_reduce"}

        self.whale_hard_block_min_score = float(
            os.getenv("WHALE_HARD_BLOCK_MIN_SCORE", "0.85") or 0.85
        )
        self.whale_reduce_min_score = float(
            os.getenv("WHALE_REDUCE_MIN_SCORE", "0.60") or 0.60
        )
        self.whale_reduce_factor = float(
            os.getenv("WHALE_REDUCE_FACTOR", "0.50") or 0.50
        )

        self.max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS", "3") or 3)
        self.block_same_symbol_reentry = str(
            os.getenv("BLOCK_SAME_SYMBOL_REENTRY", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.use_available_balance_for_sizing = str(
            os.getenv("USE_AVAILABLE_BALANCE_FOR_SIZING", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.per_position_balance_pct = float(
            os.getenv("PER_POSITION_BALANCE_PCT", "0.30") or 0.30
        )

        self.last_snapshot: Dict[str, Any] = {}
        self.last_snapshot_by_symbol: Dict[str, Dict[str, Any]] = {}

        self.position_sync_enabled = str(
            os.getenv("POSITION_SYNC_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_sync_interval_sec = int(
            float(os.getenv("POSITION_SYNC_INTERVAL_SEC", "300") or 300)
        )

        self.position_sync_remove_orphans = str(
            os.getenv("POSITION_SYNC_REMOVE_ORPHANS", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_lifecycle_enabled = str(
            os.getenv("POSITION_LIFECYCLE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_lifecycle_interval_sec = int(
            float(os.getenv("POSITION_LIFECYCLE_INTERVAL_SEC", "15") or 15)
        )
        # === ULTRA FAST SCALP EXIT ENGINE ===
        self.scalp_exit_enable = str(
            os.getenv("SCALP_EXIT_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.scalp_grace_sec = int(
            float(os.getenv("SCALP_GRACE_SEC", "20") or 20)
        )

        self.scalp_hard_stop_pct = float(
            os.getenv("SCALP_HARD_STOP_PCT", "0.04") or 0.04
        )

        self.scalp_profit_arm_pct = float(
            os.getenv("SCALP_PROFIT_ARM_PCT", "0.008") or 0.007
        )

        self.scalp_profit_lock_pct = float(
            os.getenv("SCALP_PROFIT_LOCK_PCT", "0.0035") or 0.0035
        )

        self.scalp_micro_pullback_pct = float(
            os.getenv("SCALP_MICRO_PULLBACK_PCT", "0.0025") or 0.0040
        )

        self.scalp_deep_pullback_pct = float(
            os.getenv("SCALP_DEEP_PULLBACK_PCT", "0.0040") or 0.0045
        )

        self.scalp_reverse_min_score = float(
            os.getenv("SCALP_REVERSE_MIN_SCORE", "0.54") or 0.52
        )

        self.scalp_fast_exit_profit_pct = float(
            os.getenv("SCALP_FAST_EXIT_PROFIT_PCT", "0.02") or 0.02
        )

        self.scalp_fast_exit_pullback_pct = float(
            os.getenv("SCALP_FAST_EXIT_PULLBACK_PCT", "0.0035") or 0.0035
        )

        self.scalp_stall_after_profit_sec = int(
            float(os.getenv("SCALP_STALL_AFTER_PROFIT_SEC", "180") or 120)
        )

        self.scalp_stall_min_profit_pct = float(
            os.getenv("SCALP_STALL_MIN_PROFIT_PCT", "0.01") or 0.08
        )

        # === ROI / USDT BASED EXIT CONTROL ===
        self.usdt_loss_kill_enable = str(
            os.getenv("USDT_LOSS_KILL_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.max_loss_usdt_per_trade = float(
            os.getenv("MAX_LOSS_USDT_PER_TRADE", "4.0") or 4.0
        )

        self.roi_hard_stop_pct = float(
            os.getenv("ROI_HARD_STOP_PCT", "0.08") or 0.08
        )  # %8 ROI zarar

        self.profit_lock_arm_roi_pct = float(
            os.getenv("PROFIT_LOCK_ARM_ROI_PCT", "0.0175") or 0.0175
        )  # %1.75 ROI kâr görünce arm

        self.profit_lock_retrace_roi_pct = float(
            os.getenv("PROFIT_LOCK_RETRACE_ROI_PCT", "0.0045") or 0.0045
        )  # %0.45 ROI geri dönüşte kilitle

        self.profit_floor_roi_pct = float(
            os.getenv("PROFIT_FLOOR_ROI_PCT", "0.0040") or 0.0040
        )  # kâr gördükten sonra min korunacak ROI

        self.scalp_profit_arm_roi_pct = float(
            os.getenv("SCALP_PROFIT_ARM_ROI_PCT", "0.0175") or 0.0175
        )

        self.scalp_fast_exit_profit_roi_pct = float(
            os.getenv("SCALP_FAST_EXIT_PROFIT_ROI_PCT", "0.0200") or 0.0200
        )

        self.scalp_fast_exit_pullback_roi_pct = float(
            os.getenv("SCALP_FAST_EXIT_PULLBACK_ROI_PCT", "0.0040") or 0.0040
        )

        self.scalp_retrace_roi_pct = float(
            os.getenv("SCALP_RETRACE_ROI_PCT", "0.0045") or 0.0045
        )

        self.scalp_micro_pullback_roi_pct = float(
            os.getenv("SCALP_MICRO_PULLBACK_ROI_PCT", "0.0035") or 0.0035
        )

        self.scalp_deep_pullback_roi_pct = float(
            os.getenv("SCALP_DEEP_PULLBACK_ROI_PCT", "0.0060") or 0.0060
        )

        # === STRONG PROFIT LOCK ===
        self.profit_lock_enable = str(
            os.getenv("PROFIT_LOCK_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.profit_lock_min_hold_sec = int(
            float(os.getenv("PROFIT_LOCK_MIN_HOLD_SEC", "10") or 10)
        )

        # erken kâr koruma
        self.profit_guard_arm_pct = float(
            os.getenv("PROFIT_GUARD_ARM_PCT", "0.008") or 0.008
        )
        self.profit_guard_retrace_pct = float(
            os.getenv("PROFIT_GUARD_RETRACE_PCT", "0.0035") or 0.0035
        )

        # sert kâr kilidi
        self.profit_lock_arm_pct = float(
            os.getenv("PROFIT_LOCK_ARM_PCT", "0.02") or 0.02
        )
        self.profit_lock_retrace_pct = float(
            os.getenv("PROFIT_LOCK_RETRACE_PCT", "0.0018") or 0.0018
        )

        # ters sinyal kâr koruma
        self.profit_reverse_arm_pct = float(
            os.getenv("PROFIT_REVERSE_ARM_PCT", "0.012") or 0.012
        )
        self.profit_reverse_min_score = float(
            os.getenv("PROFIT_REVERSE_MIN_SCORE", "0.50") or 0.50
        )

        # minimum korunacak kâr tabanı
        self.profit_floor_pct = float(
            os.getenv("PROFIT_FLOOR_PCT", "0.004") or 0.004
        )
        # === ENTRY CONFIRMATION (2-BAR) ===
        self.entry_confirm_enable = str(
            os.getenv("ENTRY_CONFIRM_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.entry_confirm_bars = int(
            float(os.getenv("ENTRY_CONFIRM_BARS", "2") or 2)
        )

        self.entry_confirm_max_age_sec = int(
            float(os.getenv("ENTRY_CONFIRM_MAX_AGE_SEC", "120") or 120)
        )
        # === DEAD TRADE EXIT ===
        self.dead_trade_exit_enable = str(
            os.getenv("DEAD_TRADE_EXIT_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.dead_trade_max_sec = int(
            float(os.getenv("DEAD_TRADE_MAX_SEC", "90") or 90)
        )

        self.dead_trade_min_best_pnl_pct = float(
            os.getenv("DEAD_TRADE_MIN_BEST_PNL_PCT", "0.002") or 0.002
        )
        # === NO-TRADE ZONE ===
        self.no_trade_zone_enable = str(
            os.getenv("NO_TRADE_ZONE_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.no_trade_min_score = float(
            os.getenv("NO_TRADE_MIN_SCORE", "0.63") or 0.63
        )

        self.no_trade_min_whale_score = float(
            os.getenv("NO_TRADE_MIN_WHALE_SCORE", "0.45") or 0.45
        )

        self.no_trade_min_price_move_pct = float(
            os.getenv("NO_TRADE_MIN_PRICE_MOVE_PCT", "0.0010") or 0.0010
        )

        self.no_trade_require_whale_align = str(
            os.getenv("NO_TRADE_REQUIRE_WHALE_ALIGN", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self.default_sl_pct = float(os.getenv("SL_PCT", "0.01") or 0.01)
        self.default_tp_pct = float(os.getenv("TP_PCT", "0.02") or 0.02)
        self.default_trailing_pct = float(os.getenv("TRAILING_PCT", "0.03") or 0.03)

        self.trailing_activation_pct = float(
            os.getenv("TRAILING_ACTIVATION_PCT", "0.003") or 0.003
        )

        self.stall_close_enabled = str(
            os.getenv("STALL_CLOSE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.stall_min_pnl_pct = float(
            os.getenv("STALL_MIN_PNL_PCT", "0.002") or 0.002
        )

        self.stall_close_after_profit_sec = int(
            float(os.getenv("STALL_CLOSE_AFTER_PROFIT_SEC", "900") or 900)
        )

        self.weak_signal_enabled = str(
            os.getenv("WEAK_SIGNAL_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.weak_signal_mode = str(
            os.getenv("WEAK_SIGNAL_MODE", "protect")
        ).strip().lower()

        self.weak_signal_action = str(
            os.getenv("WEAK_SIGNAL_ACTION", "protect")
        ).strip().lower()

        self.weak_signal_close_thr = float(
            os.getenv("WEAK_SIGNAL_CLOSE_THR", "0.45") or 0.45
        )
        self.weak_signal_reduce_thr = float(
            os.getenv("WEAK_SIGNAL_REDUCE_THR", "0.50") or 0.50
        )
        self.weak_signal_protect_thr = float(
            os.getenv("WEAK_SIGNAL_PROTECT_THR", "0.55") or 0.55
        )

        self.weak_signal_min_pnl_pct = float(
            os.getenv("WEAK_SIGNAL_MIN_PNL_PCT", "0.0025") or 0.0025
        )
        self.weak_signal_reduce_frac = float(
            os.getenv("WEAK_SIGNAL_REDUCE_FRAC", "0.50") or 0.50
        )

        self.weak_signal_cooldown_sec = int(
            float(os.getenv("WEAK_SIGNAL_COOLDOWN_SEC", "300") or 300)
        )
        self.weak_signal_grace_sec = int(
            float(os.getenv("WEAK_SIGNAL_GRACE_SEC", "120") or 120)
        )
        self.weak_signal_fresh_sec = int(
            float(os.getenv("WEAK_SIGNAL_FRESH_SEC", "180") or 180)
        )

        self.weak_signal_hold_below = float(
            os.getenv("WEAK_SIGNAL_HOLD_BELOW", "0.55") or 0.55
        )
        self.weak_signal_reverse_above = float(
            os.getenv("WEAK_SIGNAL_REVERSE_ABOVE", "0.60") or 0.60
        )

        self.hold_close_enabled = str(
            os.getenv("HOLD_CLOSE_ENABLED", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.hold_close_min_pnl_pct = float(
            os.getenv("HOLD_CLOSE_MIN_PNL_PCT", "0.008") or 0.008
        )

        self.reverse_close_enabled = str(
            os.getenv("REVERSE_CLOSE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.opposite_signal_close_enable = str(
            os.getenv("OPPOSITE_SIGNAL_CLOSE_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.tp_runner_enable = str(
            os.getenv("TP_RUNNER_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.tp_arm_pnl_pct = float(
            os.getenv("TP_ARM_PNL_PCT", "0.020") or 0.020
        )

        self.tp_runner_pullback_pct = float(
            os.getenv("TP_RUNNER_PULLBACK_PCT", "0.006") or 0.006
        )

        self.stall_peak_gate_pct = float(
            os.getenv("STALL_PEAK_GATE_PCT", "0.040") or 0.040
        )

        self.manual_close_detect_enable = str(
            os.getenv("MANUAL_CLOSE_DETECT_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self.hard_stop_enable = str(
            os.getenv("HARD_STOP_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.hard_stop_loss_pct = float(
            os.getenv("HARD_STOP_LOSS_PCT", "0.040") or 0.040
        )
        # fallback / compatibility
        self.scalp_retrace_pct = float(
            os.getenv("SCALP_RETRACE_PCT", "0.0010") or 0.0010
        )
        self.manual_close_reconcile_on_lifecycle = str(
            os.getenv("MANUAL_CLOSE_RECONCILE_ON_LIFECYCLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        try:
            self.opposite_signal_min_score = float(
                os.getenv("OPPOSITE_SIGNAL_MIN_SCORE", "0.55") or 0.55
            )
        except Exception:
            self.opposite_signal_min_score = 0.55

        try:
            self.opposite_signal_max_age_sec = float(
                os.getenv("OPPOSITE_SIGNAL_MAX_AGE_SEC", "900") or 900.0
            )
        except Exception:
            self.opposite_signal_max_age_sec = 900.0

        try:
            self.opposite_signal_confirm_count = int(
                float(os.getenv("OPPOSITE_SIGNAL_CONFIRM_COUNT", "1") or 1)
            )
        except Exception:
            self.opposite_signal_confirm_count = 1

        self.tp_runner_enable = str(
            os.getenv("TP_RUNNER_ENABLE", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.tp_arm_pnl_pct = float(
            os.getenv("TP_ARM_PNL_PCT", "0.020") or 0.020
        )

    # ---------------------------------------------------------
    # basic helpers
    # ---------------------------------------------------------
    def _env_csv_set(self, key: str) -> Set[str]:
        raw = str(os.getenv(key, "") or "")
        return {
            item.strip().upper()
            for item in raw.split(",")
            if item and item.strip()
        }

    def _get_symbol_exit_profile(self, symbol: str) -> str:
        sym = str(symbol or "").upper().strip()
        aggressive = self._env_csv_set("COIN_EXIT_MODE_AGGRESSIVE")
        soft = self._env_csv_set("COIN_EXIT_MODE_SOFT")

        if sym in aggressive:
            return "aggressive"
        if sym in soft:
            return "soft"
        return "normal"

    def _apply_exit_profile(
        self,
        symbol: str,
        scalp_fast_exit_pullback_roi_pct: float,
        scalp_retrace_roi_pct: float,
        scalp_micro_pullback_roi_pct: float,
        scalp_deep_pullback_roi_pct: float,
        scalp_stall_after_profit_sec: float,
        profit_lock_retrace_roi_pct: float,
    ) -> Dict[str, float]:
        profile = self._get_symbol_exit_profile(symbol)

        if profile == "aggressive":
            scalp_fast_exit_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_AGGR_FAST_EXIT_PULLBACK_MULT", "0.80"))
            scalp_retrace_roi_pct *= float(os.getenv("EXIT_PROFILE_AGGR_RETRACE_MULT", "0.80"))
            scalp_micro_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_AGGR_MICRO_PULLBACK_MULT", "0.80"))
            scalp_deep_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_AGGR_DEEP_PULLBACK_MULT", "0.85"))
            scalp_stall_after_profit_sec *= float(os.getenv("EXIT_PROFILE_AGGR_STALL_AFTER_PROFIT_MULT", "0.75"))
            profit_lock_retrace_roi_pct *= float(os.getenv("EXIT_PROFILE_AGGR_PROFIT_LOCK_RETRACE_MULT", "0.85"))

        elif profile == "soft":
            scalp_fast_exit_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_SOFT_FAST_EXIT_PULLBACK_MULT", "1.20"))
            scalp_retrace_roi_pct *= float(os.getenv("EXIT_PROFILE_SOFT_RETRACE_MULT", "1.20"))
            scalp_micro_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_SOFT_MICRO_PULLBACK_MULT", "1.20"))
            scalp_deep_pullback_roi_pct *= float(os.getenv("EXIT_PROFILE_SOFT_DEEP_PULLBACK_MULT", "1.15"))
            scalp_stall_after_profit_sec *= float(os.getenv("EXIT_PROFILE_SOFT_STALL_AFTER_PROFIT_MULT", "1.25"))
            profit_lock_retrace_roi_pct *= float(os.getenv("EXIT_PROFILE_SOFT_PROFIT_LOCK_RETRACE_MULT", "1.15"))

        return {
            "profile": profile,
            "scalp_fast_exit_pullback_roi_pct": scalp_fast_exit_pullback_roi_pct,
            "scalp_retrace_roi_pct": scalp_retrace_roi_pct,
            "scalp_micro_pullback_roi_pct": scalp_micro_pullback_roi_pct,
            "scalp_deep_pullback_roi_pct": scalp_deep_pullback_roi_pct,
            "scalp_stall_after_profit_sec": scalp_stall_after_profit_sec,
            "profit_lock_retrace_roi_pct": profit_lock_retrace_roi_pct,
        }

    def _is_reduceonly_retryable_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        needles = [
            "reduceonly",
            "parameter 'reduceonly' sent when not required",
            "parameter reduceonly sent when not required",
            "-1106",
        ]
        return any(n in text for n in needles)

    def _build_close_order_kwargs(
        self,
        symbol: str,
        side: str,
        qty: float,
        position_side: Optional[str] = None,
        reduce_only: bool = True,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": qty,
        }

        if position_side:
            kwargs["positionSide"] = position_side

        if reduce_only:
            kwargs["reduceOnly"] = True

        return kwargs

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            return int(str(os.getenv(key, str(default))).strip())
        except Exception:
            return int(default)

    @staticmethod
    def _env_float(key: str, default: float) -> float:
        try:
            return float(str(os.getenv(key, str(default))).strip())
        except Exception:
            return float(default)

    @staticmethod
    def _truthy_env(key: str, default: str = "0") -> bool:
        val = str(os.getenv(key, default)).strip().lower()
        return val in ("1", "true", "yes", "on", "y")

    @staticmethod
    def _clip_float(val: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            return default

    @staticmethod
    def _safe_json(x: Any, limit: int = 1200) -> str:
        try:
            s = json.dumps(x, ensure_ascii=False, default=str)
        except Exception:
            s = str(x)
        if len(s) > limit:
            return s[:limit] + "...(trunc)"
        return s

    @staticmethod
    def _signal_u_from_any(signal: str) -> str:
        s = str(signal or "").strip().upper()
        if s in ("BUY", "LONG"):
            return "BUY"
        if s in ("SELL", "SHORT"):
            return "SELL"
        return "HOLD"

    @staticmethod
    def _normalize_side(signal: str) -> str:
        s = str(signal or "").strip().upper()
        if s == "BUY":
            return "long"
        if s == "SELL":
            return "short"
        return "hold"

    @staticmethod
    def _side_to_position_side(side: str) -> str:
        s = str(side or "").strip().lower()
        if s == "long":
            return "LONG"
        if s == "short":
            return "SHORT"
        return "BOTH"

    @staticmethod
    def _make_client_order_id(symbol: str, tag: str) -> str:
        sym = str(symbol).upper()
        rid = uuid.uuid4().hex[:12]
        return f"b1_{tag}_{sym}_{rid}"[:36]

    @staticmethod
    def _summarize_order(resp: Any) -> Dict[str, Any]:
        if not isinstance(resp, dict):
            return {"resp": str(resp)}

        keys = [
            "symbol",
            "side",
            "type",
            "orderId",
            "clientOrderId",
            "status",
            "price",
            "avgPrice",
            "origQty",
            "executedQty",
            "cumQty",
            "cumQuote",
            "reduceOnly",
            "positionSide",
            "timeInForce",
            "updateTime",
            "transactTime",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if k in resp:
                out[k] = resp.get(k)

        for k in ("code", "msg"):
            if k in resp:
                out[k] = resp.get(k)

        return out

    def _get_all_local_positions(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        # 1) Önce PositionManager/list_symbols üzerinden sembolleri bul,
        #    ama pozisyonu çalışan wrapper olan self._get_position ile oku.
        try:
            pm = getattr(self, "position_manager", None)
            list_symbols_fn = getattr(pm, "list_symbols", None) if pm is not None else None

            if callable(list_symbols_fn):
                raw_symbols = list_symbols_fn() or []

                norm_symbols = []
                for s in raw_symbols:
                    try:
                        s_text = str(s).strip()
                        if ":" in s_text:
                            s_text = s_text.split(":")[-1]
                        s_text = s_text.upper().strip()
                        if s_text:
                            norm_symbols.append(s_text)
                    except Exception:
                        pass

                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][STATE] position_manager symbols | count=%s symbols=%s",
                            len(norm_symbols),
                            norm_symbols,
                        )
                except Exception:
                    pass

                for sym_u in norm_symbols:
                    try:
                        pos = self._get_position(sym_u)
                        if isinstance(pos, dict) and pos:
                            cp = dict(pos)
                            cp["symbol"] = sym_u
                            out[sym_u] = cp
                    except Exception:
                        if self.logger:
                            self.logger.exception(
                                "[EXEC][STATE] _get_position failed | symbol=%s",
                                sym_u,
                            )
        except Exception:
            if self.logger:
                self.logger.exception("[EXEC][STATE] position_manager list failed")

        # 2) Redis fallback
        if not out:
            try:
                r = getattr(self, "redis", None)
                if r is None:
                    r = getattr(self, "redis_client", None)

                if r is not None:
                    keys = r.keys("bot:positions:*") or []

                    key_list = []
                    for k in keys:
                        if isinstance(k, (bytes, bytearray)):
                            key_list.append(k.decode("utf-8", errors="ignore"))
                        else:
                            key_list.append(str(k))

                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][STATE] redis keys scan | count=%s keys=%s",
                                len(key_list),
                                key_list,
                            )
                    except Exception:
                        pass

                    for key_s in key_list:
                        try:
                            raw = r.get(key_s)
                            if not raw:
                                continue

                            if isinstance(raw, (bytes, bytearray)):
                                raw = raw.decode("utf-8", errors="ignore")

                            pos = json.loads(raw)
                            if not isinstance(pos, dict):
                                continue

                            sym_u = str(
                                pos.get("symbol") or key_s.split(":")[-1]
                            ).upper().strip()
                            if not sym_u:
                                continue

                            cp = dict(pos)
                            cp["symbol"] = sym_u
                            out[sym_u] = cp

                        except Exception:
                            if self.logger:
                                self.logger.exception(
                                    "[EXEC][STATE] redis position parse failed | key=%s",
                                    key_s,
                                )
            except Exception:
                if self.logger:
                    self.logger.exception("[EXEC][STATE] redis fallback list failed")

        # 3) In-memory fallback
        if not out:
            try:
                mem = getattr(self, "_positions", None)
                if isinstance(mem, dict):
                    for sym, pos in mem.items():
                        try:
                            sym_u = str(sym).upper().strip()
                            if not sym_u:
                                continue
                            if isinstance(pos, dict) and pos:
                                cp = dict(pos)
                                cp["symbol"] = sym_u
                                out[sym_u] = cp
                        except Exception:
                            pass
            except Exception:
                pass

        # 4) normalize + sadece geçerli açık pozisyonlar
        norm: Dict[str, Dict[str, Any]] = {}

        for sym, pos in out.items():
            try:
                if not isinstance(pos, dict):
                    continue

                sym_u = str(sym or pos.get("symbol") or "").upper().strip()
                if not sym_u:
                    continue

                side = str(pos.get("side") or "").strip().lower()

                try:
                    qty = float(pos.get("qty") or 0.0)
                except Exception:
                    qty = 0.0

                try:
                    entry_price = float(pos.get("entry_price") or 0.0)
                except Exception:
                    entry_price = 0.0

                if side not in ("long", "short"):
                    continue
                if qty <= 0:
                    continue
                if entry_price <= 0:
                    continue

                cp = dict(pos)
                cp["symbol"] = sym_u
                cp["side"] = side
                cp["qty"] = float(qty)
                cp["entry_price"] = float(entry_price)
                cp["interval"] = str(cp.get("interval") or "5m").strip() or "5m"
                cp["notional"] = float(cp.get("notional") or (qty * entry_price))
                cp["sl_price"] = float(cp.get("sl_price") or 0.0)
                cp["tp_price"] = float(cp.get("tp_price") or 0.0)
                cp["trailing_pct"] = float(
                    cp.get("trailing_pct")
                    or getattr(self, "default_trailing_pct", 0.03)
                    or 0.03
                )
                cp["stall_ttl_sec"] = int(
                    cp.get("stall_ttl_sec")
                    or getattr(self, "default_stall_ttl_sec", 7200)
                    or 7200
                )
                cp["best_pnl_pct"] = float(cp.get("best_pnl_pct") or 0.0)
                cp["last_best_ts"] = float(cp.get("last_best_ts") or time.time())
                cp["atr_value"] = float(cp.get("atr_value") or 0.0)
                cp["highest_price"] = float(cp.get("highest_price") or entry_price)
                cp["lowest_price"] = float(cp.get("lowest_price") or entry_price)

                meta = cp.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                if not isinstance(meta.get("probs"), dict):
                    meta["probs"] = {}
                if not isinstance(meta.get("extra"), dict):
                    meta["extra"] = {}
                cp["meta"] = meta

                norm[sym_u] = cp

            except Exception:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] normalize failed | symbol=%s",
                        sym,
                    )

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][STATE] local positions resolved | count=%s symbols=%s",
                    len(norm),
                    list(norm.keys()),
                )
        except Exception:
            pass

        return norm

    def _get_exchange_open_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        client = getattr(self, "client", None)
        if client is None:
            return out

        def _append_from_rows(rows: Any) -> None:
            if not isinstance(rows, list):
                return

            for row in rows:
                if not isinstance(row, dict):
                    continue

                try:
                    amt = float(row.get("positionAmt") or 0.0)
                except Exception:
                    amt = 0.0

                if abs(amt) <= 1e-12:
                    continue

                sym = str(row.get("symbol") or "").upper().strip()
                if not sym:
                    continue

                side = "long" if amt > 0 else "short"

                entry_price = self._clip_float(row.get("entryPrice"), 0.0) or 0.0
                mark_price = self._clip_float(row.get("markPrice"), 0.0) or 0.0

                px = float(mark_price if mark_price > 0 else entry_price)
                notional = float(abs(amt) * px) if px > 0 else 0.0

                out.append(
                    {
                        "symbol": sym,
                        "side": side,
                        "qty": abs(float(amt)),
                        "entry_price": float(entry_price),
                        "mark_price": float(mark_price),
                        "notional": float(notional),
                        "raw": row,
                    }
                )

        # 1) Önce futures_account() -> positions
        try:
            fn_acc = getattr(client, "futures_account", None)
            if callable(fn_acc):
                acc = fn_acc()
                if isinstance(acc, dict):
                    _append_from_rows(acc.get("positions", []) or [])
                    if out:
                        return out
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][EXCHANGE-POS] futures_account positions read failed | err=%s",
                        str(e)[:300],
                    )
            except Exception:
                pass

        # 2) Fallback: futures_position_information()
        try:
            fn = getattr(client, "futures_position_information", None)
            if callable(fn):
                rows = fn()
                _append_from_rows(rows)
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][EXCHANGE-POS] futures_position_information read failed | err=%s",
                        str(e)[:300],
                    )
            except Exception:
                pass

        return out

    def _get_exchange_open_positions_map(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}

        try:
            rows = self._get_exchange_open_positions()
            for row in rows:
                if not isinstance(row, dict):
                    continue

                sym = str(row.get("symbol") or "").upper().strip()
                side = str(row.get("side") or "").lower().strip()
                qty = float(row.get("qty") or 0.0)

                if not sym or side not in ("long", "short") or qty <= 0:
                    continue

                out[sym] = row
        except Exception:
            try:
                if self.logger:
                    self.logger.exception("[EXEC][EXCHANGE-POS] map build failed")
            except Exception:
                pass

        return out

    def _count_open_positions(self) -> int:
        seen: Dict[str, Dict[str, Any]] = {}

        # 1) Exchange
        try:
            ex_rows = self._get_exchange_open_positions()
            for row in ex_rows:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol") or "").upper().strip()
                side = str(row.get("side") or "").lower().strip()
                qty = float(row.get("qty") or 0.0)
                if sym and side in ("long", "short") and qty > 0:
                    seen[sym] = row
        except Exception:
            pass

        # 2) Redis fallback / tamamlayıcı kaynak
        try:
            r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
            if r is not None:
                keys = r.keys("bot:positions:*") or []
                for key in keys:
                    try:
                        raw = r.get(key)
                        if not raw:
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            raw = raw.decode("utf-8", errors="ignore")

                        obj = json.loads(raw)
                        if not isinstance(obj, dict):
                            continue

                        sym = str(obj.get("symbol") or str(key).split(":")[-1]).upper().strip()
                        side = str(obj.get("side") or "").lower().strip()
                        qty = float(obj.get("qty") or 0.0)

                        if sym and side in ("long", "short") and qty > 0:
                            seen.setdefault(
                                sym,
                                {
                                    "symbol": sym,
                                    "side": side,
                                    "qty": qty,
                                },
                            )
                    except Exception:
                        continue
        except Exception:
            pass

        return int(len(seen))

    def _has_open_position_on_symbol(self, symbol: str) -> bool:
        sym_u = str(symbol).upper().strip()
        if not sym_u:
            return False

        try:
            for row in self._get_exchange_open_positions():
                if str(row.get("symbol") or "").upper().strip() == sym_u:
                    return True
        except Exception:
            pass

        try:
            pos = self._get_position(sym_u)
            if isinstance(pos, dict):
                side = str(pos.get("side") or "").lower().strip()
                qty = float(pos.get("qty") or 0.0)
                if side in ("long", "short") and qty > 0:
                    return True
        except Exception:
            pass

        return False
    def _get_available_balance_usdt_sync(self) -> float:
        client = getattr(self, "client", None)
        if client is None:
            return 0.0

        try:
            fn = getattr(client, "futures_account", None)
            if not callable(fn):
                return 0.0

            acc = fn()
            if not isinstance(acc, dict):
                return 0.0

            available = float(acc.get("availableBalance", 0.0) or 0.0)
            if available > 0:
                return available
        except Exception:
            pass

        return 0.0

    async def _get_available_balance_usdt(self) -> float:
        try:
            return float(self._get_available_balance_usdt_sync())
        except Exception:
            return 0.0

    def _compute_balance_based_notional(
        self,
        symbol: str,
        side: str,
        price: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> float:
        extra0 = extra if isinstance(extra, dict) else {}

        available_balance = self._clip_float(
            extra0.get("available_balance_usdt"),
            None,
        )
        equity_usdt = self._clip_float(
            extra0.get("equity_usdt"),
            None,
        )

        if self.use_available_balance_for_sizing and available_balance is not None and available_balance > 0:
            base_balance = float(available_balance)
            balance_source = "available_balance"
        elif equity_usdt is not None and equity_usdt > 0:
            base_balance = float(equity_usdt)
            balance_source = "equity"
        else:
            base_balance = self._clip_float(
                os.getenv("DEFAULT_EQUITY_USDT", "1000"),
                1000.0,
            ) or 1000.0
            balance_source = "fallback"

        pct = float(self.per_position_balance_pct)
        pct = max(0.01, min(pct, 1.0))

        notional = float(base_balance) * float(pct)
        notional = float(min(notional, float(self.max_position_notional)))
        notional = float(max(10.0, notional))

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][NOTIONAL] symbol=%s side=%s price=%.6f balance_source=%s "
                    "base_balance=%.2f pct=%.4f notional=%.2f",
                    str(symbol).upper(),
                    str(side).lower(),
                    float(price),
                    balance_source,
                    float(base_balance),
                    float(pct),
                    float(notional),
                )
        except Exception:
            pass

        return float(notional)

        self.tp_runner_enable = str(
            os.getenv("TP_RUNNER_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.tp_arm_pnl_pct = float(
            os.getenv("TP_ARM_PNL_PCT", "0.020") or 0.020
        )

        self.tp_runner_pullback_pct = float(
            os.getenv("TP_RUNNER_PULLBACK_PCT", "0.006") or 0.006
        )

        self.stall_peak_gate_pct = float(
            os.getenv("STALL_PEAK_GATE_PCT", "0.040") or 0.040
        )

        self.manual_close_detect_enable = str(
            os.getenv("MANUAL_CLOSE_DETECT_ENABLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.manual_close_reconcile_on_lifecycle = str(
            os.getenv("MANUAL_CLOSE_RECONCILE_ON_LIFECYCLE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

    # ---------------------------------------------------------
    # coroutine / retry helpers
    # ---------------------------------------------------------
    def _fire_and_forget(self, maybe: Any, *, label: str = "task") -> None:
        try:
            if inspect.isawaitable(maybe):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(maybe)
                except RuntimeError:
                    try:
                        if self.logger:
                            self.logger.warning("[EXEC] %s coroutine returned but no running loop; dropped", label)
                    except Exception:
                        pass
        except Exception:
            pass

    async def _await_if_coro(self, maybe: Any, *, label: str = "awaitable") -> Any:
        try:
            if inspect.isawaitable(maybe):
                return await maybe
        except Exception:
            try:
                if self.logger:
                    self.logger.exception(f"[EXEC] await failed | label={label}")
            except Exception:
                pass
        return maybe

    def _call_with_retry(
        self,
        fn: Any,
        payload: Dict[str, Any],
        attempts: int = 3,
        base_sleep: float = 0.6,
    ) -> Any:
        last_err: Optional[Exception] = None
        attempts = max(1, int(attempts))

        for i in range(attempts):
            try:
                return fn(**payload)
            except Exception as e:
                last_err = e
                if i >= attempts - 1:
                    break
                sleep_s = float(base_sleep) * (2 ** i)
                time.sleep(sleep_s)

        if last_err is not None:
            raise last_err
        raise RuntimeError("call_with_retry failed without exception")

    # ---------------------------------------------------------
    # whale helpers
    # ---------------------------------------------------------
    def _extract_whale_context(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        if isinstance(extra, dict):
            out.update(extra)

        def _merge_from_obj(obj: Any) -> None:
            if not isinstance(obj, dict):
                return

            for k in (
                "whale_score",
                "whale_dir",
                "whale_action",
                "whale_bias",
                "signal_score",
                "score",
                "score_total",
                "_score_total_final",
                "_score_selected",
                "trail_pct",
                "stall_ttl_sec",
                "price",
                "intent_id",
                "recommended_leverage",
                "recommended_notional_pct",
                "risk_tags",
                "reasons",
                "reason_codes",
                "w_min",
            ):
                v = obj.get(k)
                if v is not None and (k not in out or out.get(k) in (None, "", 0, 0.0, [], {})):
                    out[k] = v

            raw2 = obj.get("raw")
            if isinstance(raw2, dict):
                for k in (
                    "whale_score",
                    "whale_dir",
                    "whale_action",
                    "whale_bias",
                    "price",
                    "trail_pct",
                    "stall_ttl_sec",
                ):
                    v = raw2.get(k)
                    if v is not None and (k not in out or out.get(k) in (None, "", 0, 0.0)):
                        out[k] = v

        _merge_from_obj(extra)

        raw = out.get("raw")

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = None

        _merge_from_obj(raw)

        if isinstance(raw, dict):
            raw_nested = raw.get("raw")
            if isinstance(raw_nested, str):
                try:
                    raw_nested = json.loads(raw_nested)
                except Exception:
                    raw_nested = None
            _merge_from_obj(raw_nested)

            if isinstance(raw_nested, dict):
                raw_nested2 = raw_nested.get("raw")
                if isinstance(raw_nested2, str):
                    try:
                        raw_nested2 = json.loads(raw_nested2)
                    except Exception:
                        raw_nested2 = None
                _merge_from_obj(raw_nested2)

        try:
            ws = float(out.get("whale_score") or 0.0)
        except Exception:
            ws = 0.0
        out["whale_score"] = ws

        wd = str(out.get("whale_dir") or "none").strip().lower()
        if wd in ("buy", "bull", "up"):
            wd = "long"
        elif wd in ("sell", "bear", "down"):
            wd = "short"
        elif wd not in ("long", "short"):
            wd = "none"
        out["whale_dir"] = wd

        wb = str(out.get("whale_bias") or "").strip().lower()
        if not wb:
            if wd in ("long", "short") and ws > 0:
                wb = wd
            else:
                wb = "hold"
        out["whale_bias"] = wb

        return out

    @staticmethod
    def _whale_action(extra: Dict[str, Any]) -> str:
        try:
            return str(extra.get("whale_action", "") or "").strip().lower()
        except Exception:
            return ""

    @staticmethod
    def _whale_bias(side: str, extra: Dict[str, Any]) -> str:
        try:
            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            if side in ("long", "short") and wdir in ("long", "short"):
                if side == wdir:
                    return "align"
                return "oppose"
        except Exception:
            pass
        return "hold"

    def _should_block_open_by_whale(self, side: str, extra: Dict[str, Any]) -> bool:
        try:
            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)

            if action in self.whale_block_actions and ws >= float(self.whale_hard_block_min_score):
                return True

            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            if side in ("long", "short") and wdir in ("long", "short"):
                if wdir != side and ws >= float(self.whale_hard_block_min_score):
                    return True
        except Exception:
            pass
        return False

    def _apply_whale_open_adjustments(self, side: str, notional: float, extra: Dict[str, Any]) -> float:
        out = float(notional)
        try:
            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)

            if action in self.whale_reduce_actions and ws >= float(self.whale_reduce_min_score):
                out *= 0.5

            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            if wdir in ("long", "short") and wdir != str(side).strip().lower() and ws >= 0.58:
                out *= 0.75
        except Exception:
            pass

        return max(0.0, float(out))

    # ---------------------------------------------------------
    # price helpers
    # ---------------------------------------------------------
    def _get_cached_mid_price(self, symbol: str) -> Optional[float]:
        sym = str(symbol).upper()
        try:
            pc = getattr(self, "price_cache", None)
            if pc is not None:
                for method_name in ("get_mid_price", "get_price", "get_last_price"):
                    fn = getattr(pc, method_name, None)
                    if callable(fn):
                        v = fn(sym)
                        fv = self._clip_float(v, None)
                        if fv is not None and fv > 0:
                            return fv
        except Exception:
            pass

        try:
            if self.redis is not None:
                raw = self.redis.get(f"price:{sym}")
                if raw:
                    fv = self._clip_float(raw, None)
                    if fv is not None and fv > 0:
                        return fv
        except Exception:
            pass

        return None

    def _resolve_price(
        self,
        symbol: str,
        price: Any = None,
        mark_price: Any = None,
        last_price: Any = None,
    ) -> Optional[float]:
        for v in (mark_price, price, last_price):
            fv = self._clip_float(v, None)
            if fv is not None and fv > 0:
                return fv

        cached = self._get_cached_mid_price(symbol)
        if cached is not None and cached > 0:
            return float(cached)

        return None
    # ---------------------------------------------------------
    # exchange info cache
    # ---------------------------------------------------------
    def _get_exchange_info_cached(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()

        if (
            not force_refresh
            and isinstance(self._exchange_info_cache, dict)
            and (now - float(self._exchange_info_cache_ts)) < float(self.exchange_info_cache_ttl_sec)
        ):
            return self._exchange_info_cache

        client = getattr(self, "client", None)
        if client is None:
            return {}

        try:
            fn = getattr(client, "futures_exchange_info", None)
            if not callable(fn):
                return {}

            data = fn()
            if isinstance(data, dict):
                self._exchange_info_cache = data
                self._exchange_info_cache_ts = now
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][EXCHANGE_INFO] refreshed cache ttl=%.1fs symbols=%s",
                            float(self.exchange_info_cache_ttl_sec),
                            len(data.get("symbols", []) or []),
                        )
                except Exception:
                    pass
                return data
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning("[EXEC][EXCHANGE_INFO] refresh failed err=%s", str(e))
            except Exception:
                pass

        return self._exchange_info_cache or {}

    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()

        def _extract(symbol_row: Dict[str, Any]) -> Dict[str, Any]:
            filters = symbol_row.get("filters") or []
            if not isinstance(filters, list):
                filters = []

            tick_size = 0.0
            step_size = 0.0
            min_qty = 0.0
            min_notional = 0.0

            for f in filters:
                if not isinstance(f, dict):
                    continue

                ftype = str(f.get("filterType") or "").strip().upper()

                if ftype == "PRICE_FILTER":
                    try:
                        tick_size = float(f.get("tickSize") or 0.0)
                    except Exception:
                        pass

                elif ftype == "LOT_SIZE":
                    try:
                        step_size = float(f.get("stepSize") or 0.0)
                    except Exception:
                        pass
                    try:
                        min_qty = float(f.get("minQty") or 0.0)
                    except Exception:
                        pass

                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    try:
                        min_notional = float(
                            f.get("notional")
                            or f.get("minNotional")
                            or 0.0
                        )
                    except Exception:
                        pass

            out = dict(symbol_row)
            out["symbol"] = sym
            out["tick_size"] = float(tick_size)
            out["step"] = float(step_size)
            out["step_size"] = float(step_size)
            out["min_qty"] = float(min_qty)
            out["min_notional"] = float(min_notional)
            return out

        try:
            exch = self._get_exchange_info_cached(force_refresh=False)
            symbols = exch.get("symbols", []) if isinstance(exch, dict) else []
            for s in symbols:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper().strip() == sym:
                    info = _extract(s)
                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYMBOL_INFO] resolved | symbol=%s tick_size=%.10f step=%.10f min_qty=%.10f min_notional=%.10f",
                                sym,
                                float(info.get("tick_size") or 0.0),
                                float(info.get("step") or 0.0),
                                float(info.get("min_qty") or 0.0),
                                float(info.get("min_notional") or 0.0),
                            )
                    except Exception:
                        pass
                    return info
        except Exception:
            pass

        try:
            exch = self._get_exchange_info_cached(force_refresh=True)
            symbols = exch.get("symbols", []) if isinstance(exch, dict) else []
            for s in symbols:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper().strip() == sym:
                    info = _extract(s)
                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYMBOL_INFO] resolved(refresh) | symbol=%s tick_size=%.10f step=%.10f min_qty=%.10f min_notional=%.10f",
                                sym,
                                float(info.get("tick_size") or 0.0),
                                float(info.get("step") or 0.0),
                                float(info.get("min_qty") or 0.0),
                                float(info.get("min_notional") or 0.0),
                            )
                    except Exception:
                        pass
                    return info
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][SYMBOL_INFO] fetch failed | symbol=%s err=%s",
                        sym,
                        str(e),
                    )
            except Exception:
                pass

        return {
            "symbol": sym,
            "tick_size": 0.0,
            "step": 0.0,
            "step_size": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
        }

    def _fmt_qty(self, symbol: str, qty: float) -> str:
        sym = str(symbol).upper().strip()
        q = float(qty or 0.0)

        if q <= 0:
            return "0"

        step = 0.0
        try:
            s_info = self._get_symbol_info(sym)
            if isinstance(s_info, dict):
                step = float(s_info.get("step_size") or s_info.get("step") or 0.0)
        except Exception:
            step = 0.0

        if step <= 0:
            try:
                if abs(q - round(q)) < 1e-12:
                    return str(int(round(q)))
            except Exception:
                pass
            return ("%.10f" % q).rstrip("0").rstrip(".")

        step_s = ("%.16f" % step).rstrip("0").rstrip(".")
        if "." in step_s:
            decimals = len(step_s.split(".")[1])
        else:
            decimals = 0

        fmt = "{:0." + str(decimals) + "f}"
        return fmt.format(q)

    def _normalize_close_quantity(
        self,
        symbol: str,
        qty: float,
        price: float,
    ) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()

        out: Dict[str, Any] = {
            "symbol": sym,
            "raw_qty": float(qty or 0.0),
            "norm_qty": 0.0,
            "price": float(price or 0.0),
            "step": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
            "reject_reason": "-",
        }

        try:
            raw_qty = abs(float(qty or 0.0))
        except Exception:
            raw_qty = 0.0

        try:
            px = float(price or 0.0)
        except Exception:
            px = 0.0

        out["raw_qty"] = float(raw_qty)
        out["price"] = float(px)

        if not sym or raw_qty <= 0.0:
            out["reject_reason"] = "raw_qty_invalid"
            return out

        info = {}
        try:
            info = self._get_symbol_info(sym)
        except Exception:
            info = {}

        def _f(x: Any, default: float = 0.0) -> float:
            try:
                v = float(x)
                if v < 0:
                    return float(default)
                return v
            except Exception:
                return float(default)

        step = _f(
            (info or {}).get("step")
            or (info or {}).get("step_size")
            or 0.0,
            0.0,
        )
        min_qty = _f((info or {}).get("min_qty") or 0.0, 0.0)
        min_notional = _f((info or {}).get("min_notional") or 0.0, 0.0)

        out["step"] = float(step)
        out["min_qty"] = float(min_qty)
        out["min_notional"] = float(min_notional)

        norm_qty = float(raw_qty)

        # step varsa aşağı yuvarla
        if step > 0.0:
            try:
                steps = int(raw_qty / step)
                norm_qty = float(steps * step)
            except Exception:
                norm_qty = 0.0

            # kayan nokta artıklarını temizle
            try:
                step_str = f"{step:.16f}".rstrip("0")
                if "." in step_str:
                    decimals = len(step_str.split(".")[1])
                    norm_qty = round(norm_qty, decimals)
                else:
                    norm_qty = round(norm_qty, 0)
            except Exception:
                pass

        # step yoksa ham qty ile devam et
        else:
            norm_qty = float(raw_qty)

        # min_qty kontrolü
        if min_qty > 0.0 and norm_qty < min_qty:
            out["norm_qty"] = 0.0
            out["reject_reason"] = "below_min_qty"
            return out

        # min_notional kontrolü
        if px > 0.0 and min_notional > 0.0:
            if (norm_qty * px) < min_notional:
                # close'ta mümkünse min_notional yüzünden pozisyonu sıfırlamaya çalış:
                # önce raw qty yeterliyse ham qty'yi tekrar deneyelim
                retry_qty = float(raw_qty)

                if step > 0.0:
                    try:
                        steps = int(retry_qty / step)
                        retry_qty = float(steps * step)
                    except Exception:
                        retry_qty = 0.0

                    try:
                        step_str = f"{step:.16f}".rstrip("0")
                        if "." in step_str:
                            decimals = len(step_str.split(".")[1])
                            retry_qty = round(retry_qty, decimals)
                        else:
                            retry_qty = round(retry_qty, 0)
                    except Exception:
                        pass

                if retry_qty > 0.0 and retry_qty >= norm_qty and (retry_qty * px) >= min_notional:
                    norm_qty = float(retry_qty)
                else:
                    out["norm_qty"] = 0.0
                    out["reject_reason"] = "below_min_notional"
                    return out

        if norm_qty <= 0.0:
            out["norm_qty"] = 0.0
            out["reject_reason"] = "normalized_zero"
            return out

        out["norm_qty"] = float(norm_qty)
        out["reject_reason"] = "-"

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][CLOSE][QTY] symbol=%s raw_qty=%.10f norm_qty=%.10f price=%.6f step=%.10f min_qty=%.10f min_notional=%.6f reject=%s",
                    sym,
                    float(out["raw_qty"]),
                    float(out["norm_qty"]),
                    float(out["price"]),
                    float(out["step"]),
                    float(out["min_qty"]),
                    float(out["min_notional"]),
                    str(out["reject_reason"]),
                )
        except Exception:
            pass

        return out

    def _extract_symbol_filters(self, symbol_info: Dict[str, Any]) -> Dict[str, float]:
        out = {
            "step_size": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
            "tick_size": 0.0,
        }

        try:
            filters = symbol_info.get("filters", []) or []
            for f in filters:
                if not isinstance(f, dict):
                    continue

                ftype = str(f.get("filterType", "")).upper()

                if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                    step_size = float(f.get("stepSize", 0.0) or 0.0)
                    min_qty = float(f.get("minQty", 0.0) or 0.0)

                    if step_size > 0:
                        out["step_size"] = max(out["step_size"], step_size)
                    if min_qty > 0:
                        out["min_qty"] = max(out["min_qty"], min_qty)

                elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = f.get("notional") or f.get("minNotional") or 0.0
                    out["min_notional"] = float(min_notional or 0.0)

                elif ftype == "PRICE_FILTER":
                    tick_size = float(f.get("tickSize", 0.0) or 0.0)
                    if tick_size > 0:
                        out["tick_size"] = tick_size
        except Exception:
            pass

        return out

    @staticmethod
    def _round_qty(qty: float, digits: int = 12) -> float:
        try:
            return float(round(float(qty), int(digits)))
        except Exception:
            return 0.0

    @staticmethod
    def _floor_to_step(qty: float, step: float) -> float:
        if step <= 0:
            return float(qty)
        try:
            return math.floor(float(qty) / float(step)) * float(step)
        except Exception:
            return 0.0

    def _normalize_order_qty(
        self,
        symbol: str,
        raw_qty: float,
        price: float,
        symbol_info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        sym = str(symbol).upper()
        px = float(price or 0.0)
        rq = float(raw_qty or 0.0)

        filters = self._extract_symbol_filters(symbol_info or {})
        step = float(filters.get("step_size", 0.0) or 0.0)
        min_qty = float(filters.get("min_qty", 0.0) or 0.0)
        min_notional = float(filters.get("min_notional", 0.0) or 0.0)

        q = float(rq)
        if step > 0:
            q = self._floor_to_step(q, step)
        q = self._round_qty(q)

        reject_reason = ""
        if q <= 0:
            reject_reason = "qty_le_zero"
        elif min_qty > 0 and q < min_qty:
            reject_reason = "below_min_qty"
        elif px > 0 and min_notional > 0 and (q * px) < min_notional:
            reject_reason = "below_min_notional"

        meta = {
            "symbol": sym,
            "raw_qty": float(rq),
            "price": float(px),
            "step_size": float(step),
            "min_qty": float(min_qty),
            "min_notional": float(min_notional),
            "normalized_qty": float(q),
            "normalized_notional": float(q * px),
            "reject_reason": str(reject_reason),
        }

        if reject_reason:
            return 0.0, meta
        return float(q), meta
    # ---------------------------------------------------------
    # leverage helpers
    # ---------------------------------------------------------
    def _resolve_target_leverage(self, extra: Optional[Dict[str, Any]] = None) -> int:
        extra0 = extra if isinstance(extra, dict) else {}

        lev_min = int(self._env_int("LEV_MIN", 3))
        lev_max_env = int(self._env_int("LEV_MAX", max(lev_min, self.max_leverage)))
        lev_max = max(lev_min, min(int(lev_max_env), int(self.max_leverage)))

        rec = extra0.get("recommended_leverage")
        rec_f = self._clip_float(rec, None)

        if rec_f is None:
            target = lev_min
        else:
            target = int(round(float(rec_f)))

        target = max(lev_min, min(target, lev_max))
        return int(target)

    def _set_symbol_leverage(self, symbol: str, leverage: int) -> int:
        sym = str(symbol).upper()
        lev = int(max(1, leverage))

        client = getattr(self, "client", None)
        if client is None:
            return lev

        fn = getattr(client, "futures_change_leverage", None)
        if not callable(fn):
            return lev

        resp = fn(symbol=sym, leverage=lev)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][LEV][SET] symbol=%s requested=%s response=%s",
                    sym,
                    int(lev),
                    self._safe_json(resp, limit=900),
                )
        except Exception:
            pass

        try:
            if isinstance(resp, dict):
                return int(resp.get("leverage") or lev)
        except Exception:
            pass

        return lev

    # ---------------------------------------------------------
    # equity / notional
    # ---------------------------------------------------------
    def _get_futures_equity_usdt_sync(self) -> float:
        client = getattr(self, "client", None)
        if client is None:
            return 0.0

        try:
            fn = getattr(client, "futures_account_balance", None)
            if callable(fn):
                rows = fn()
                if isinstance(rows, list):
                    for r in rows:
                        if str(r.get("asset", "")).upper() == "USDT":
                            bal = self._clip_float(r.get("balance"), 0.0) or 0.0
                            cw = self._clip_float(r.get("crossWalletBalance"), None)
                            if cw is not None and cw > 0:
                                return float(cw)
                            return float(bal)
        except Exception:
            pass

        return 0.0

    async def _get_futures_equity_usdt(self) -> float:
        return float(self._get_futures_equity_usdt_sync())

    def _compute_notional(self, symbol: str, side: str, price: float, extra: Dict[str, Any]) -> float:
        base = float(self.base_order_notional)

        liq_use_notional = self._truthy_env("LIQ_USE_NOTIONAL", "1")
        min_pct = self._env_float("NOTIONAL_MIN_PCT", 0.02)
        max_pct = self._env_float("NOTIONAL_MAX_PCT", 0.25)

        eq = self._clip_float(extra.get("equity_usdt"), None)
        if eq is None or eq <= 0:
            eq = self._clip_float(os.getenv("DEFAULT_EQUITY_USDT", "1000"), 1000.0) or 1000.0

        npct = self._clip_float(extra.get("recommended_notional_pct"), None)
        if npct is not None:
            npct = max(float(min_pct), min(float(npct), float(max_pct)))

        if liq_use_notional and npct is not None and npct > 0:
            notional = float(eq) * float(npct)
        else:
            notional = float(base)

        notional = min(float(notional), float(self.max_position_notional))
        notional = max(10.0, float(notional))

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][NOTIONAL] symbol=%s side=%s base=%.2f equity=%.2f npct=%s notional=%.2f",
                    str(symbol).upper(),
                    str(side),
                    float(base),
                    float(eq),
                    ("-" if npct is None else f"{float(npct):.4f}"),
                    float(notional),
                )
        except Exception:
            pass

        return float(notional)

    # ---------------------------------------------------------
    # position state helpers
    # ---------------------------------------------------------
    def _pos_key(self, symbol: str) -> str:
        return f"bot:positions:{str(symbol).upper()}"

    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()

        try:
            pm = getattr(self, "position_manager", None)
            if pm is not None and hasattr(pm, "get_position"):
                pos = pm.get_position(sym)
                if isinstance(pos, dict):
                    return pos
        except Exception:
            pass

        try:
            if self.redis is not None:
                raw = self.redis.get(self._pos_key(sym))
                if raw:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        return obj
        except Exception:
            pass

        return None

    def _store_latest_signal(
        self,
        symbol: str,
        side: str,
        interval: str,
        score: float,
        raw: Optional[Dict[str, Any]] = None,
    ) -> None:
        sym = str(symbol).upper().strip()
        if not sym:
            return

        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
        if r is None:
            return

        try:
            payload = {
                "symbol": sym,
                "side": str(side or "").strip().lower(),
                "interval": str(interval or "").strip(),
                "score": float(score or 0.0),
                "ts": float(time.time()),
                "raw": raw if isinstance(raw, dict) else {},
            }

            serialized = json.dumps(payload, ensure_ascii=False, default=str)

            try:
                r.set(f"bot:last_signal:{sym}", serialized, ex=3600)
            except TypeError:
                r.set(f"bot:last_signal:{sym}", serialized)
                try:
                    r.expire(f"bot:last_signal:{sym}", 3600)
                except Exception:
                    pass

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][WEAK] latest signal stored | symbol=%s side=%s interval=%s score=%.6f",
                        sym,
                        str(side or "").strip().lower(),
                        str(interval or "").strip(),
                        float(score or 0.0),
                    )
            except Exception:
                pass

        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][WEAK] latest signal store failed | symbol=%s",
                        sym,
                    )
            except Exception:
                pass
    def _confirm_entry_signal(
        self,
        symbol: str,
        side: str,
        interval: str,
        score: float,
    ) -> bool:
        if not bool(getattr(self, "entry_confirm_enable", True)):
            return True

        sym = str(symbol).upper().strip()
        side0 = str(side).lower().strip()
        interval0 = str(interval or "5m").strip() or "5m"

        if side0 not in ("long", "short"):
            return False

        need = max(1, int(getattr(self, "entry_confirm_bars", 2) or 2))
        max_age = max(30, int(getattr(self, "entry_confirm_max_age_sec", 120) or 120))

        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
        if r is None:
            return True

        key = f"bot:entry_confirm:{sym}:{interval0}:{side0}"

        try:
            raw = r.get(key)
            state = json.loads(raw) if raw else {}
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}

        now_ts = time.time()
        last_side = str(state.get("side") or "").lower()
        last_ts = float(state.get("ts") or 0.0)
        count = int(state.get("count") or 0)

        if last_side == side0 and (now_ts - last_ts) <= max_age:
            count += 1
        else:
            count = 1

        new_state = {
            "side": side0,
            "count": count,
            "ts": now_ts,
        }

        try:
            r.set(key, json.dumps(new_state))
            r.expire(key, max_age)
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][ENTRY-CONFIRM] symbol=%s side=%s count=%s/%s score=%.4f",
                    sym,
                    side0,
                    count,
                    need,
                    float(score),
                )
        except Exception:
            pass

        return count >= need

    def _in_no_trade_zone(
        self,
        symbol: str,
        side: str,
        signal_score: float,
        whale_score: float,
        whale_dir: str,
        order_price: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:

        if not bool(getattr(self, "no_trade_zone_enable", True)):
            return None

        sym = str(symbol).upper().strip()
        side0 = str(side).lower().strip()
        whale_dir0 = str(whale_dir or "none").lower().strip()
        meta0 = meta if isinstance(meta, dict) else {}

        min_score = float(getattr(self, "no_trade_min_score", 0.63) or 0.63)
        min_whale = float(getattr(self, "no_trade_min_whale_score", 0.45) or 0.45)
        min_move = float(getattr(self, "no_trade_min_price_move_pct", 0.0010) or 0.0010)
        require_align = bool(getattr(self, "no_trade_require_whale_align", True))

        # 1) borderline score
        if float(signal_score) < float(min_score):
            return "no_trade_low_score"

        # 2) whale zayıf
        if float(whale_score) < float(min_whale):
            return "no_trade_weak_whale"

        # 3) whale yön uyumsuz
        if require_align and whale_dir0 in ("long", "short") and whale_dir0 != side0:
            return "no_trade_whale_misaligned"

        # 4) hareket yok (cansız piyasa)
        ref_price = float(meta0.get("price") or order_price or 0.0)

        try:
            live_price = self._resolve_price(symbol=sym, price=float(order_price or 0.0))
        except Exception:
            live_price = None

        if live_price and ref_price > 0:
            move_pct = abs(float(live_price) - float(ref_price)) / ref_price
            if float(move_pct) < float(min_move):
                return "no_trade_low_volatility"

        return None
    def _evaluate_position_signal_exit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        price: float,
    ) -> Optional[Dict[str, Any]]:
        import json
        import time

        sym = str(symbol).upper().strip()
        if not sym or not isinstance(pos, dict):
            return None

        side = str(pos.get("side") or "").strip().lower()
        entry_price = float(pos.get("entry_price") or 0.0)

        if side not in ("long", "short") or entry_price <= 0:
            return None

        # ---------------------------------
        # PNL hesap
        # ---------------------------------
        if side == "long":
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price

        # ---------------------------------
        # reverse / weak enable
        # ---------------------------------
        if not bool(getattr(self, "reverse_close_enabled", True)):
            return None

        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
        if r is None:
            return None

        # ---------------------------------
        # son sinyal oku
        # ---------------------------------
        try:
            raw = r.get(f"bot:last_signal:{sym}")
            if not raw:
                return None

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            sig = json.loads(raw)
            if not isinstance(sig, dict):
                return None
        except Exception:
            return None

        sig_side = str(sig.get("side") or "").strip().lower()
        sig_score = float(sig.get("score") or 0.0)
        sig_ts = float(sig.get("ts") or 0.0)

        now_ts = time.time()

        # ---------------------------------
        # thresholdlar
        # ---------------------------------
        min_score = float(getattr(self, "opposite_signal_min_score", 0.58))
        max_age = float(getattr(self, "opposite_signal_max_age_sec", 600))
        confirm_need = int(getattr(self, "opposite_signal_confirm_count", 1))

        tp_arm = float(getattr(self, "tp_arm_pnl_pct", 0.02))  # %2

        if sig_side not in ("long", "short"):
            return None

        if not sig_ts or (now_ts - sig_ts) > max_age:
            return None

        if sig_score < min_score:
            return None

        opposite = (
            (side == "long" and sig_side == "short")
            or (side == "short" and sig_side == "long")
        )

        if not opposite:
            return None
        # ---------------------------------
        # CONFIRMATION (anti noise)
        # ---------------------------------
        confirm_key = f"bot:reverse_confirm:{sym}:{side}:{sig_side}"

        try:
            confirm_count = int(r.incr(confirm_key))
            r.expire(confirm_key, int(max(30, max_age)))
        except Exception:
            confirm_count = 1

        if confirm_count < confirm_need:
            return None

        try:
            r.delete(confirm_key)
        except Exception:
            pass

        # ---------------------------------
        # 🔥 ANA MANTIK
        # ---------------------------------

        # 1️⃣ %2+ KÂR VAR → TERS SİNYALDE ANINDA ÇIK
        if pnl_pct >= tp_arm:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][PROFIT-REVERSE] close | symbol=%s side=%s pnl=%.4f sig=%s score=%.4f",
                        sym,
                        side,
                        pnl_pct,
                        sig_side,
                        sig_score,
                    )
            except Exception:
                pass

            return {
                "action": "close",
                "reason": "profit_reverse",
                "pnl_pct": float(pnl_pct),
                "signal_score": float(sig_score),
            }

        # 2️⃣ düşük kâr / zarar → klasik reverse close
        if pnl_pct < tp_arm:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][REVERSE] close | symbol=%s side=%s pnl=%.4f sig=%s",
                        sym,
                        side,
                        pnl_pct,
                        sig_side,
                    )
            except Exception:
                pass

            return {
                "action": "close",
                "reason": "reverse_signal",
                "pnl_pct": float(pnl_pct),
                "signal_score": float(sig_score),
            }

        return None
    def _resolve_position_leverage(self, pos: Dict[str, Any]) -> float:
        extra = pos.get("extra") if isinstance(pos, dict) else None
        extra = extra if isinstance(extra, dict) else {}

        candidates = [
            pos.get("leverage") if isinstance(pos, dict) else None,
            pos.get("target_leverage") if isinstance(pos, dict) else None,
            extra.get("target_leverage"),
            extra.get("recommended_leverage"),
            getattr(self, "default_order_leverage", None),
            getattr(self, "default_leverage", None),
            1,
        ]

        for v in candidates:
            try:
                f = float(v)
                if f > 0:
                    return f
            except Exception:
                pass

        return 1.0

    def _calc_unrealized_pnl_usdt(
        self,
        side: str,
        entry_price: float,
        price: float,
        qty: float,
    ) -> float:
        try:
            entry = float(entry_price)
            px = float(price)
            q = float(qty)
        except Exception:
            return 0.0

        if entry <= 0 or px <= 0 or q <= 0:
            return 0.0

        if str(side).lower().strip() == "long":
            return (px - entry) * q
        return (entry - px) * q

    def _evaluate_scalp_exit_pro(
        self,
        symbol: str,
        pos: Dict[str, Any],
        price: float,
        side: str,
        entry_price: float,
        highest_price: float,
        lowest_price: float,
        best_pnl_pct: float,
        last_best_ts: float,
        now_ts: float,
    ) -> Optional[str]:
        scalp_enabled = bool(
            getattr(
                self,
                "scalp_exit_enable",
                getattr(self, "scalp_engine_enable", True),
            )
        )
        if not scalp_enabled:
            return None

        try:
            opened_at_raw = str(pos.get("opened_at") or "").strip()
            opened_ts = 0.0
            if opened_at_raw:
                try:
                    opened_ts = datetime.fromisoformat(
                        opened_at_raw.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    opened_ts = 0.0

            grace_sec = float(getattr(self, "scalp_grace_sec", 15) or 15)
            if opened_ts > 0 and (float(now_ts) - float(opened_ts)) < grace_sec:
                return None
        except Exception:
            pass

        if side == "long":
            pnl_pct = (float(price) - float(entry_price)) / max(float(entry_price), 1e-12)
            retrace_pct = (
                (float(highest_price) - float(price)) / max(float(highest_price), 1e-12)
                if float(highest_price) > 0
                else 0.0
            )
        else:
            pnl_pct = (float(entry_price) - float(price)) / max(float(entry_price), 1e-12)
            retrace_pct = (
                (float(price) - float(lowest_price)) / max(float(lowest_price), 1e-12)
                if float(lowest_price) > 0
                else 0.0
            )

        lev = float(self._resolve_position_leverage(pos))
        roi_pct = float(pnl_pct) * lev
        best_roi_pct = float(best_pnl_pct) * lev
        retrace_roi_pct = float(retrace_pct) * lev

        roi_hard_stop_pct = abs(float(getattr(self, "roi_hard_stop_pct", 0.08) or 0.08))
        scalp_profit_arm_roi_pct = float(
            getattr(self, "scalp_profit_arm_roi_pct", 0.0175) or 0.0175
        )
        scalp_fast_exit_profit_roi_pct = float(
            getattr(self, "scalp_fast_exit_profit_roi_pct", 0.0200) or 0.0200
        )
        scalp_fast_exit_pullback_roi_pct = float(
            getattr(self, "scalp_fast_exit_pullback_roi_pct", 0.0040) or 0.0040
        )
        scalp_retrace_roi_pct = float(
            getattr(self, "scalp_retrace_roi_pct", 0.0045) or 0.0045
        )
        scalp_micro_pullback_roi_pct = float(
            getattr(self, "scalp_micro_pullback_roi_pct", 0.0035) or 0.0035
        )
        scalp_deep_pullback_roi_pct = float(
            getattr(self, "scalp_deep_pullback_roi_pct", 0.0060) or 0.0060
        )
        scalp_reverse_min_score = float(
            getattr(self, "scalp_reverse_min_score", 0.50) or 0.50
        )
        scalp_stall_min_profit_roi_pct = float(
            getattr(self, "scalp_stall_min_profit_roi_pct", 0.0100) or 0.0100
        )
        scalp_stall_after_profit_sec = float(
            getattr(self, "scalp_stall_after_profit_sec", 45) or 45
        )

        profile_cfg = self._apply_exit_profile(
            symbol=symbol,
            scalp_fast_exit_pullback_roi_pct=scalp_fast_exit_pullback_roi_pct,
            scalp_retrace_roi_pct=scalp_retrace_roi_pct,
            scalp_micro_pullback_roi_pct=scalp_micro_pullback_roi_pct,
            scalp_deep_pullback_roi_pct=scalp_deep_pullback_roi_pct,
            scalp_stall_after_profit_sec=scalp_stall_after_profit_sec,
            profit_lock_retrace_roi_pct=scalp_retrace_roi_pct,
        )

        profile = profile_cfg["profile"]
        scalp_fast_exit_pullback_roi_pct = profile_cfg["scalp_fast_exit_pullback_roi_pct"]
        scalp_retrace_roi_pct = profile_cfg["scalp_retrace_roi_pct"]
        scalp_micro_pullback_roi_pct = profile_cfg["scalp_micro_pullback_roi_pct"]
        scalp_deep_pullback_roi_pct = profile_cfg["scalp_deep_pullback_roi_pct"]
        scalp_stall_after_profit_sec = profile_cfg["scalp_stall_after_profit_sec"]

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][EXIT-PROFILE] symbol=%s profile=%s fast_pb=%.6f retrace=%.6f micro_pb=%.6f deep_pb=%.6f stall_after_profit=%.2f",
                    symbol,
                    profile,
                    scalp_fast_exit_pullback_roi_pct,
                    scalp_retrace_roi_pct,
                    scalp_micro_pullback_roi_pct,
                    scalp_deep_pullback_roi_pct,
                    scalp_stall_after_profit_sec,
                )
        except Exception:
            pass

        if float(roi_pct) <= -roi_hard_stop_pct:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_hard_stop_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_hard_stop_roi"

        if (
            float(best_roi_pct) >= scalp_fast_exit_profit_roi_pct
            and float(retrace_roi_pct) >= scalp_fast_exit_pullback_roi_pct
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_fast_exit_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_fast_exit_roi"

        if (
            float(best_roi_pct) >= scalp_profit_arm_roi_pct
            and float(retrace_roi_pct) >= scalp_retrace_roi_pct
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_profit_retrace_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_profit_retrace_roi"

        if (
            float(best_roi_pct) >= scalp_profit_arm_roi_pct
            and float(retrace_roi_pct) >= scalp_micro_pullback_roi_pct
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_micro_pullback_exit_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_micro_pullback_exit_roi"

        if (
            float(best_roi_pct) >= max(scalp_profit_arm_roi_pct, scalp_micro_pullback_roi_pct)
            and float(retrace_roi_pct) >= scalp_deep_pullback_roi_pct
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_deep_pullback_exit_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_deep_pullback_exit_roi"

        try:
            reverse_meta = self._evaluate_position_signal_exit(
                symbol=symbol,
                pos=pos,
                price=float(price),
            ) or {}
        except Exception:
            reverse_meta = {}

        if isinstance(reverse_meta, dict):
            try:
                reverse_score = float(reverse_meta.get("signal_score") or 0.0)
            except Exception:
                reverse_score = 0.0

            if (
                str(reverse_meta.get("reason") or "") == "reverse_signal"
                and float(reverse_score) >= scalp_reverse_min_score
                and float(best_roi_pct) >= scalp_profit_arm_roi_pct
            ):
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                            symbol,
                            side,
                            profile,
                            "scalp_reverse_kill_roi",
                            float(price),
                            float(entry_price),
                            float(pnl_pct),
                            float(best_pnl_pct),
                        )
                except Exception:
                    pass
                return "scalp_reverse_kill_roi"

        stalled_for = float(now_ts) - float(last_best_ts or now_ts)
        if (
            float(best_roi_pct) >= scalp_stall_min_profit_roi_pct
            and float(stalled_for) >= scalp_stall_after_profit_sec
        ):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s profile=%s reason=%s price=%.8f entry=%.8f pnl_pct=%.6f best_pnl_pct=%.6f",
                        symbol,
                        side,
                        profile,
                        "scalp_stall_exit_roi",
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass
            return "scalp_stall_exit_roi"

        return None

    def _evaluate_scalp_exit(
        self,
        symbol: str,
        pos: Dict[str, Any],
        price: float,
        side: str,
        entry_price: float,
        best_pnl_pct: float,
        highest_price: float,
        lowest_price: float,
        now_ts: float,
        last_best_ts: float,
    ) -> Optional[str]:
        return self._evaluate_scalp_exit_pro(
            symbol=symbol,
            pos=pos,
            price=price,
            side=side,
            entry_price=entry_price,
            best_pnl_pct=best_pnl_pct,
            highest_price=highest_price,
            lowest_price=lowest_price,
            now_ts=now_ts,
            last_best_ts=last_best_ts,
        )
    def _get_latest_signal(self, symbol: str) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()
        if not sym:
            return {}

        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
        if r is None:
            return {}

        try:
            raw = r.get(f"bot:last_signal:{sym}")
            if not raw:
                return {}

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][WEAK] latest signal read failed | symbol=%s",
                        sym,
                    )
            except Exception:
                pass
            return {}

    def _check_opposite_signal_close(
        self,
        symbol: str,
        side: str,
        interval: str,
        price: float,
    ) -> Optional[str]:
        sym = str(symbol).upper().strip()
        pos_side = str(side or "").strip().lower()
        itv = str(interval or "").strip()

        if not sym or pos_side not in ("long", "short"):
            return None

        sig = self._get_latest_signal(sym)
        if not isinstance(sig, dict) or not sig:
            return None

        sig_side = str(sig.get("side") or "").strip().lower()
        sig_interval = str(sig.get("interval") or "").strip()
        sig_score = float(sig.get("score") or 0.0)
        sig_ts = float(sig.get("ts") or 0.0)

        now_ts = time.time()

        try:
            max_age_sec = float(getattr(self, "opposite_signal_max_age_sec", 900) or 900.0)
        except Exception:
            max_age_sec = 900.0

        try:
            min_score = float(getattr(self, "opposite_signal_min_score", 0.55) or 0.55)
        except Exception:
            min_score = 0.55

        try:
            confirm_count_req = int(getattr(self, "opposite_signal_confirm_count", 1) or 1)
        except Exception:
            confirm_count_req = 1

        if sig_side not in ("long", "short"):
            return None

        if sig_interval and itv and sig_interval != itv:
            return None

        if sig_ts <= 0 or (now_ts - sig_ts) > max_age_sec:
            return None

        if sig_score < min_score:
            return None

        is_opposite = (
            (pos_side == "long" and sig_side == "short")
            or (pos_side == "short" and sig_side == "long")
        )
        if not is_opposite:
            return None

        confirm_key = f"bot:opposite_counter:{sym}"
        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)

        confirm_count = 1
        if r is not None:
            try:
                new_v = r.incr(confirm_key)
                confirm_count = int(new_v or 1)
                try:
                    r.expire(confirm_key, int(max(30, max_age_sec)))
                except Exception:
                    pass
            except Exception:
                confirm_count = 1

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][WEAK] opposite signal seen | symbol=%s pos_side=%s sig_side=%s score=%.6f confirm=%s/%s interval=%s price=%.6f",
                    sym,
                    pos_side,
                    sig_side,
                    float(sig_score),
                    int(confirm_count),
                    int(confirm_count_req),
                    itv,
                    float(price),
                )
        except Exception:
            pass

        if confirm_count < confirm_count_req:
            return None

        if r is not None:
            try:
                r.delete(confirm_key)
            except Exception:
                pass

        return "opposite_signal"

    def _get_exchange_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        if not sym:
            return None

        try:
            rows = self._get_exchange_open_positions()
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol") or "").upper().strip() == sym:
                    return row
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][EXCHANGE-POS] fetch skipped | symbol=%s err=%s",
                        sym,
                        str(e)[:300],
                    )
            except Exception:
                pass

        return None
    def _get_effective_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        if not sym:
            return None

        local_pos = self._get_position(sym)
        if isinstance(local_pos, dict):
            try:
                side = str(local_pos.get("side") or "").strip().lower()
                qty = float(local_pos.get("qty") or 0.0)
                if side in ("long", "short") and qty > 0:
                    return local_pos
            except Exception:
                pass

        ex_pos = self._get_exchange_position(sym)
        if isinstance(ex_pos, dict):
            return ex_pos

        return None

    def _set_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        sym = str(symbol).upper()

        try:
            pm = getattr(self, "position_manager", None)
            if pm is not None and hasattr(pm, "set_position"):
                pm.set_position(sym, pos)
                return
        except Exception:
            pass

        try:
            if self.redis is not None:
                self.redis.set(self._pos_key(sym), json.dumps(pos, ensure_ascii=False, default=str))
        except Exception:
            pass

    def _del_position(self, symbol: str) -> None:
        sym = str(symbol).upper().strip()
        if not sym:
            return

        pm = getattr(self, "position_manager", None)

        # position_manager tarafını olabildiğince temizle
        try:
            if pm is not None:
                for method_name in (
                    "clear_position",
                    "delete_position",
                    "del_position",
                    "remove_position",
                    "close_position_state",
                ):
                    fn = getattr(pm, method_name, None)
                    if callable(fn):
                        try:
                            fn(sym)
                            break
                        except TypeError:
                            try:
                                fn(symbol=sym)
                                break
                            except Exception:
                                pass
        except Exception:
            pass

        # Redis tarafını zorla temizle
        try:
            if self.redis is not None:
                self.redis.delete(self._pos_key(sym))
        except Exception:
            try:
                if self.logger:
                    self.logger.exception("[EXEC][STATE] redis delete failed | symbol=%s", sym)
            except Exception:
                pass

        # doğrulama logu
        try:
            if self.redis is not None:
                still_exists = self.redis.get(self._pos_key(sym))
                if still_exists:
                    if self.logger:
                        self.logger.warning(
                            "[EXEC][STATE] local position delete attempted but key still exists | symbol=%s",
                            sym,
                        )
                else:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][STATE] local position removed | symbol=%s",
                            sym,
                        )
        except Exception:
            pass

    def _finalize_close_state(
        self,
        symbol: str,
        side: str,
        pos: Optional[Dict[str, Any]],
        interval: str,
        realized_pnl: float,
        close_price: float,
        reason: str,
        exchange_pos_after: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()
        pos = pos if isinstance(pos, dict) else {}

        residual_qty = 0.0
        residual_side = str(side or "").strip().lower() or str(pos.get("side") or "").strip().lower()

        if isinstance(exchange_pos_after, dict):
            try:
                residual_qty = abs(float(exchange_pos_after.get("qty") or 0.0))
            except Exception:
                residual_qty = 0.0
            try:
                residual_side = str(exchange_pos_after.get("side") or residual_side).strip().lower() or residual_side
            except Exception:
                pass

        min_qty = 0.0
        min_notional = 0.0
        try:
            s_info = self._get_symbol_info(sym)
            if isinstance(s_info, dict):
                min_qty = float(s_info.get("min_qty") or 0.0)
                min_notional = float(s_info.get("min_notional") or 0.0)
        except Exception:
            min_qty = 0.0
            min_notional = 0.0

        dust_threshold = max(float(min_qty), 0.0)

        result = {
            "symbol": sym,
            "residual_qty": float(residual_qty),
            "residual_side": residual_side,
            "dust_threshold": float(dust_threshold),
            "min_notional": float(min_notional),
            "fully_closed": False,
            "dust_remaining": False,
            "partial_remaining": False,
        }

        if residual_qty <= 0.0:
            self._del_position(sym)
            self._remove_from_bridge_state(sym)

            try:
                chk_local = self._get_position(sym)
                chk_bridge = self._get_bridge_state()
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] cleanup after close | symbol=%s local_exists=%s bridge_exists=%s",
                        sym,
                        bool(isinstance(chk_local, dict) and chk_local),
                        bool(isinstance(chk_bridge, dict) and sym in chk_bridge),
                    )
            except Exception:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] cleanup verify failed | symbol=%s",
                        sym,
                    )

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE] CLOSED | symbol=%s side=%s qty=%.10f close=%.6f realized_pnl=%.6f reason=%s",
                        sym,
                        residual_side or side,
                        float(pos.get("qty") or 0.0),
                        float(close_price),
                        float(realized_pnl),
                        str(reason),
                    )
            except Exception:
                pass

            result["fully_closed"] = True
            return result

        residual_pos = dict(pos) if isinstance(pos, dict) else {}
        residual_pos["symbol"] = sym
        residual_pos["side"] = residual_side
        residual_pos["qty"] = float(residual_qty)

        if isinstance(exchange_pos_after, dict):
            try:
                residual_pos["entry_price"] = float(
                    exchange_pos_after.get("entry_price")
                    or residual_pos.get("entry_price")
                    or 0.0
                )
            except Exception:
                pass

        self._set_position(sym, residual_pos)
        self._upsert_bridge_state_on_open(
            symbol=sym,
            side=residual_side,
            interval=str(interval or residual_pos.get("interval") or "5m"),
            intent_id="",
        )

        if dust_threshold > 0.0 and residual_qty <= dust_threshold:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][CLOSE] residual dust remains on exchange | symbol=%s side=%s residual_qty=%.10f dust_threshold=%.10f",
                        sym,
                        residual_side,
                        float(residual_qty),
                        float(dust_threshold),
                    )
            except Exception:
                pass

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] partial close state updated | symbol=%s residual_qty=%.10f",
                        sym,
                        float(residual_qty),
                    )
            except Exception:
                pass

            result["dust_remaining"] = True
            return result

        try:
            if self.logger:
                self.logger.warning(
                    "[EXEC][CLOSE] residual position remains on exchange | symbol=%s side=%s residual_qty=%.10f",
                    sym,
                    residual_side,
                    float(residual_qty),
                )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][STATE] partial close state updated | symbol=%s residual_qty=%.10f",
                    sym,
                    float(residual_qty),
                )
        except Exception:
            pass

        result["partial_remaining"] = True
        return result

    def _remove_from_bridge_state(self, symbol: str) -> None:
        sym = str(symbol).upper().strip()
        if not sym or self.redis is None:
            return

        key = str(os.getenv("BRIDGE_STATE_KEY", "open_positions_state")).strip() or "open_positions_state"

        try:
            raw = self.redis.get(key)
            if not raw:
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] bridge state already empty | key=%s symbol=%s",
                        key,
                        sym,
                    )
                return

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            obj = json.loads(raw)
            if not isinstance(obj, dict):
                if self.logger:
                    self.logger.warning(
                        "[EXEC][STATE] bridge state invalid json object | key=%s symbol=%s",
                        key,
                        sym,
                    )
                return

            existed = sym in obj
            if existed:
                del obj[sym]
                self.redis.set(key, json.dumps(obj, ensure_ascii=False, default=str))

            # doğrulama
            raw2 = self.redis.get(key)
            ok_removed = True
            if raw2:
                if isinstance(raw2, (bytes, bytearray)):
                    raw2 = raw2.decode("utf-8", errors="ignore")
                obj2 = json.loads(raw2)
                if isinstance(obj2, dict) and sym in obj2:
                    ok_removed = False

            if self.logger:
                if existed and ok_removed:
                    self.logger.info(
                        "[EXEC][STATE] removed from bridge state | key=%s symbol=%s",
                        key,
                        sym,
                    )
                elif existed and not ok_removed:
                    self.logger.warning(
                        "[EXEC][STATE] bridge state delete attempted but symbol still exists | key=%s symbol=%s",
                        key,
                        sym,
                    )
                else:
                    self.logger.info(
                        "[EXEC][STATE] bridge state symbol already absent | key=%s symbol=%s",
                        key,
                        sym,
                    )
        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] remove from bridge state failed | key=%s symbol=%s",
                        key,
                        sym,
                    )
            except Exception:
                pass

    def _get_bridge_state(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        key = str(os.getenv("BRIDGE_STATE_KEY", "open_positions_state")).strip() or "open_positions_state"

        try:
            r = getattr(self, "redis", None)
            if r is None:
                r = getattr(self, "redis_client", None)

            if r is None:
                return {}

            raw = r.get(key)
            if not raw:
                return {}

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            obj = json.loads(raw)
            if not isinstance(obj, dict):
                return {}

            if symbol is None:
                return obj

            sym = str(symbol).upper().strip()
            if not sym:
                return {}

            st = obj.get(sym)
            return st if isinstance(st, dict) else {}

        except Exception:
            try:
                if self.logger:
                    if symbol is None:
                        self.logger.exception(
                            "[EXEC][STATE] get bridge state failed | key=%s",
                            key,
                        )
                    else:
                        self.logger.exception(
                            "[EXEC][STATE] bridge state read failed | symbol=%s key=%s",
                            str(symbol).upper().strip(),
                            key,
                        )
            except Exception:
                pass
            return {}

    def _upsert_bridge_state_on_open(
        self,
        symbol: str,
        side: str,
        interval: str,
        intent_id: str = "",
    ) -> None:
        sym = str(symbol).upper().strip()
        side_n = str(side or "").lower().strip()
        interval_n = str(interval or "").strip()
        if not sym or self.redis is None:
            return

        key = str(os.getenv("BRIDGE_STATE_KEY", "open_positions_state")).strip() or "open_positions_state"

        try:
            raw = self.redis.get(key)
            obj: Dict[str, Any] = {}

            if raw:
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="ignore")
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        obj = parsed
                except Exception:
                    obj = {}

            obj[sym] = {
                "side": side_n,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "intent_id": str(intent_id or "").strip(),
                "interval": interval_n,
                "expires_at": float(time.time()) + float(12 * 3600),
            }

            self.redis.set(
                key,
                json.dumps(obj, ensure_ascii=False, default=str),
            )

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] bridge state upserted | key=%s symbol=%s side=%s interval=%s intent_id=%s",
                        key,
                        sym,
                        side_n,
                        interval_n,
                        str(intent_id or "").strip(),
                    )
            except Exception:
                pass

        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] bridge state upsert failed | symbol=%s",
                        sym,
                    )
            except Exception:
                pass

    def sync_positions_with_exchange(self) -> Dict[str, Any]:
        summary = {
            "exchange_open": 0,
            "local_open": 0,
            "bridge_open": 0,
            "removed_local": [],
            "removed_bridge": [],
            "added_local": [],
            "added_bridge": [],
            "kept": [],
            "skipped_fresh_local": [],
            "skipped_fresh_bridge": [],
        }

        if not self.position_sync_enabled:
            return summary

        now_ts = time.time()

        def _state_ts(obj: Any) -> float:
            if not isinstance(obj, dict):
                return 0.0

            for key in ("sync_grace_until", "bridge_written_ts", "created_ts", "last_best_ts", "expires_at"):
                try:
                    v = float(obj.get(key) or 0.0)
                    if v > 0:
                        return v
                except Exception:
                    pass

            try:
                ts_utc = str(obj.get("ts_utc") or "").strip()
                if ts_utc:
                    return datetime.fromisoformat(ts_utc.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

            try:
                opened_at = str(obj.get("opened_at") or "").strip()
                if opened_at:
                    return datetime.fromisoformat(opened_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

            return 0.0
        def _fresh(obj: Any, grace_sec: float = 45.0) -> bool:
            ts = _state_ts(obj)
            if ts <= 0:
                return False

            # sync_grace_until absolute timestamp gibi de kullanılabilir
            try:
                sg = float((obj or {}).get("sync_grace_until") or 0.0)
                if sg > now_ts:
                    return True
            except Exception:
                pass

            return (now_ts - ts) < float(grace_sec)

        exchange_map = self._get_exchange_open_positions_map()
        summary["exchange_open"] = len(exchange_map)

        local_map = self._get_all_local_positions()
        summary["local_open"] = len(local_map)

        bridge_all = self._get_bridge_state()
        bridge_all = bridge_all if isinstance(bridge_all, dict) else {}
        summary["bridge_open"] = len(bridge_all)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][SYNC] start | exchange_open=%s local_open=%s bridge_open=%s",
                    int(summary["exchange_open"]),
                    int(summary["local_open"]),
                    int(summary["bridge_open"]),
                )
        except Exception:
            pass

        # 1) exchange'de var ama local'de yok -> hydrate et
        for sym, ex_pos in exchange_map.items():
            if sym in local_map:
                summary["kept"].append(sym)
                continue

            try:
                bridge_st = bridge_all.get(sym, {}) if isinstance(bridge_all, dict) else {}

                hydrated = {
                    "symbol": sym,
                    "side": str(ex_pos.get("side") or "").strip().lower(),
                    "qty": float(ex_pos.get("qty") or 0.0),
                    "entry_price": float(ex_pos.get("entry_price") or 0.0),
                    "notional": float(ex_pos.get("notional") or 0.0),
                    "interval": str(bridge_st.get("interval") or "5m").strip() or "5m",
                    "opened_at": datetime.now(timezone.utc).isoformat(),

                    # >>> NEW
                    "created_ts": float(now_ts),
                    "bridge_written_ts": float(now_ts),
                    "sync_grace_until": float(now_ts + 45.0),
                    # <<< NEW

                    "sl_price": 0.0,
                    "tp_price": 0.0,
                    "trailing_pct": float(getattr(self, "default_trailing_pct", 0.03) or 0.03),
                    "stall_ttl_sec": int(getattr(self, "default_stall_ttl_sec", 7200) or 7200),
                    "best_pnl_pct": 0.0,
                    "last_best_ts": float(now_ts),
                    "atr_value": 0.0,
                    "highest_price": float(ex_pos.get("entry_price") or 0.0),
                    "lowest_price": float(ex_pos.get("entry_price") or 0.0),
                    "meta": {
                        "probs": {},
                        "extra": {
                            "hydrated_from_exchange": True,
                            "bridge_state": bridge_st,
                        },
                    },
                }

                self._set_position(sym, hydrated)
                self._upsert_bridge_state_on_open(
                    symbol=sym,
                    side=str(hydrated.get("side") or ""),
                    interval=str(hydrated.get("interval") or "5m"),
                    intent_id=str(bridge_st.get("intent_id") or ""),
                )

                summary["added_local"].append(sym)
                summary["added_bridge"].append(sym)

                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][SYNC] hydrated missing local position from exchange | symbol=%s side=%s qty=%.10f entry=%.10f interval=%s",
                            sym,
                            str(hydrated.get("side") or ""),
                            float(hydrated.get("qty") or 0.0),
                            float(hydrated.get("entry_price") or 0.0),
                            str(hydrated.get("interval") or ""),
                        )
                except Exception:
                    pass

            except Exception:
                try:
                    if self.logger:
                        self.logger.exception("[EXEC][SYNC] hydrate failed | symbol=%s", sym)
                except Exception:
                    pass

        # local map'i refresh et
        refreshed_local_map = self._get_all_local_positions()
        refreshed_bridge_all = self._get_bridge_state()
        refreshed_bridge_all = refreshed_bridge_all if isinstance(refreshed_bridge_all, dict) else {}

        # 2) local'de var ama exchange'de yok -> fresh ise dokunma, değilse temizle
        for sym, pos in refreshed_local_map.items():
            if sym not in exchange_map:
                if _fresh(pos, grace_sec=90.0):
                    summary["skipped_fresh_local"].append(sym)
                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYNC] skip fresh local-only state | symbol=%s",
                                sym,
                            )
                    except Exception:
                        pass
                    continue

                try:
                    if self.position_sync_remove_orphans:
                        self._del_position(sym)
                        summary["removed_local"].append(sym)

                        try:
                            if self.logger:
                                self.logger.info(
                                    "[EXEC][SYNC] removed orphan local position | symbol=%s",
                                    sym,
                                )
                        except Exception:
                            pass

                    if sym in refreshed_bridge_all and not _fresh(refreshed_bridge_all.get(sym, {}), grace_sec=45.0):
                        self._remove_from_bridge_state(sym)
                        summary["removed_bridge"].append(sym)

                except Exception:
                    try:
                        if self.logger:
                            self.logger.exception(
                                "[EXEC][SYNC] failed removing orphan state | symbol=%s",
                                sym,
                            )
                    except Exception:
                        pass
            else:
                if sym not in summary["kept"]:
                    summary["kept"].append(sym)

        # 3) bridge'de var ama ne exchange'de ne local'de yok -> fresh ise dokunma
        final_local_map = self._get_all_local_positions()
        final_bridge_all = self._get_bridge_state()
        final_bridge_all = final_bridge_all if isinstance(final_bridge_all, dict) else {}

        for sym, bst in final_bridge_all.items():
            if sym not in exchange_map and sym not in final_local_map:
                try:
                    created_ts = float(bst.get("created_ts") or 0.0)
                except Exception:
                    created_ts = 0.0

                if created_ts > 0 and (time.time() - created_ts) < 5.0:
                    summary["skipped_fresh_bridge"].append(sym)
                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYNC] skip very-fresh bridge-only state | symbol=%s",
                                sym,
                            )
                    except Exception:
                        pass
                    continue

                final_local_map[sym] = bst
                summary["added_local"].append(sym)
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][SYNC] promote bridge-only state to local | symbol=%s",
                            sym,
                        )
                except Exception:
                    pass
                try:
                    self._remove_from_bridge_state(sym)
                    summary["removed_bridge"].append(sym)
                    try:
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYNC] removed stale bridge-only state | symbol=%s",
                                sym,
                            )
                    except Exception:
                        pass
                except Exception:
                    try:
                        if self.logger:
                            self.logger.exception(
                                "[EXEC][SYNC] failed removing stale bridge-only state | symbol=%s",
                                sym,
                            )
                    except Exception:
                        pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][SYNC] done | exchange_open=%s local_open=%s bridge_open=%s removed_local=%s removed_bridge=%s added_local=%s added_bridge=%s kept=%s skipped_fresh_local=%s skipped_fresh_bridge=%s",
                    int(summary["exchange_open"]),
                    int(summary["local_open"]),
                    int(summary["bridge_open"]),
                    summary["removed_local"],
                    summary["removed_bridge"],
                    summary["added_local"],
                    summary["added_bridge"],
                    summary["kept"],
                    summary["skipped_fresh_local"],
                    summary["skipped_fresh_bridge"],
                )
        except Exception:
            pass

        return summary

    async def _position_sync_loop(self) -> None:
        while True:
            try:
                self.sync_positions_with_exchange()
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception("[EXEC][SYNC] periodic sync failed")
                except Exception:
                    pass

            try:
                await asyncio.sleep(max(30, int(self.position_sync_interval_sec)))
            except Exception:
                await asyncio.sleep(300)

    async def _position_lifecycle_loop(self) -> None:
        interval_sec = max(1, int(getattr(self, "position_lifecycle_interval_sec", 15)))

        while True:
            try:
                positions = self._get_all_local_positions()

                # local boş ama exchange/local senkron kaçmış olabilir -> hydrate dene
                if not positions and bool(getattr(self, "position_sync_enabled", True)):
                    try:
                        sync_summary = self.sync_positions_with_exchange()
                        positions = self._get_all_local_positions()
                        if self.logger:
                            self.logger.info(
                                "[EXEC][LIFECYCLE] post-sync refresh | positions=%s summary=%s",
                                len(positions),
                                sync_summary,
                            )
                    except Exception as e:
                        try:
                            if self.logger:
                                self.logger.exception(
                                    "[EXEC][LIFECYCLE] sync-on-empty failed | err=%s",
                                    str(e),
                                )
                        except Exception:
                            pass

                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][LIFECYCLE] tick | positions=%s symbols=%s",
                            len(positions),
                            list(positions.keys()),
                        )
                except Exception:
                    pass

                for sym, pos in positions.items():
                    try:
                        sym_u = str(sym).upper().strip()
                        side = str(pos.get("side") or "").strip().lower()
                        qty = float(pos.get("qty") or 0.0)
                        interval = str(pos.get("interval") or "5m").strip() or "5m"
                        entry_price = float(pos.get("entry_price") or 0.0)

                        if side not in ("long", "short") or qty <= 0 or entry_price <= 0:
                            try:
                                if self.logger:
                                    self.logger.info(
                                        "[EXEC][LIFECYCLE] skip invalid position | symbol=%s side=%s qty=%.10f entry=%.6f",
                                        sym_u,
                                        side,
                                        qty,
                                        entry_price,
                                    )
                            except Exception:
                                pass
                            continue

                        px = None
                        price_source = ""

                        try:
                            px = self._resolve_price(symbol=sym_u)
                            if px is not None and px > 0:
                                price_source = "cache"
                        except Exception:
                            px = None

                        if px is None or px <= 0:
                            try:
                                client = getattr(self, "client", None)
                                fn = getattr(client, "futures_mark_price", None) if client is not None else None
                                if callable(fn):
                                    mp = fn(symbol=sym_u)
                                    if isinstance(mp, dict):
                                        px = self._clip_float(mp.get("markPrice"), None)
                                        if px is not None and px > 0:
                                            price_source = "mark_price"
                            except Exception:
                                px = None

                        if px is None or px <= 0:
                            try:
                                if self.logger:
                                    self.logger.info(
                                        "[EXEC][LIFECYCLE] skip no price | symbol=%s side=%s qty=%.10f interval=%s entry=%.6f",
                                        sym_u,
                                        side,
                                        qty,
                                        interval,
                                        entry_price,
                                    )
                            except Exception:
                                pass
                            continue

                        try:
                            if self.logger:
                                self.logger.info(
                                    "[EXEC][LIFECYCLE] check | symbol=%s side=%s qty=%.10f entry=%.6f price=%.6f interval=%s src=%s",
                                    sym_u,
                                    side,
                                    qty,
                                    entry_price,
                                    float(px),
                                    interval,
                                    price_source or "-",
                                )
                        except Exception:
                            pass

                        self._check_sl_tp_trailing(
                            symbol=sym_u,
                            price=float(px),
                            interval=interval,
                        )

                    except Exception as e:
                        try:
                            if self.logger:
                                self.logger.exception(
                                    "[EXEC][LIFECYCLE] CRASH(symbol) | symbol=%s err=%s",
                                    str(sym),
                                    str(e),
                                )
                        except Exception:
                            pass
                        continue

            except Exception as e:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][LIFECYCLE] CRASH(loop) | err=%s",
                            str(e),
                        )
                except Exception:
                    pass

                await asyncio.sleep(1)

            try:
                await asyncio.sleep(interval_sec)
            except Exception:
                await asyncio.sleep(1)

    def _get_exchange_residual_qty(self, symbol: str, side: str) -> float:
        sym = str(symbol).upper().strip()
        side0 = str(side or "").strip().lower()

        if side0 not in ("long", "short"):
            return 0.0

        ex_pos = self._get_exchange_position(sym)
        if not isinstance(ex_pos, dict):
            return 0.0

        ex_side = str(ex_pos.get("side") or "").strip().lower()
        ex_qty = float(ex_pos.get("qty") or 0.0)

        if ex_side != side0:
            return 0.0

        return float(ex_qty)
    def _iter_local_positions(self) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        try:
            mp = self._get_all_local_positions()
            for sym, pos in mp.items():
                if isinstance(pos, dict):
                    out.append((str(sym).upper().strip(), pos))
        except Exception:
            pass
        return out

    @staticmethod
    def _calc_pnl_pct(side: str, entry_price: float, current_price: float) -> float:
        ep = float(entry_price or 0.0)
        cp = float(current_price or 0.0)
        if ep <= 0 or cp <= 0:
            return 0.0
        s = str(side or "").strip().lower()
        if s == "long":
            return (cp / ep) - 1.0
        if s == "short":
            return (ep / cp) - 1.0
        return 0.0

    def _latest_symbol_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        snap = self.last_snapshot_by_symbol.get(sym)
        if not isinstance(snap, dict):
            return None

        ts_epoch = float(snap.get("ts_epoch") or 0.0)
        if ts_epoch <= 0:
            return None

        age = time.time() - ts_epoch
        if age > float(self.weak_signal_fresh_sec):
            return None
        return snap

    def _protect_position(self, symbol: str, pos: Dict[str, Any], current_price: float, reason: str) -> None:
        sym = str(symbol).upper().strip()
        side = str(pos.get("side") or "").strip().lower()
        entry_price = float(pos.get("entry_price") or 0.0)
        cp = float(current_price or 0.0)

        if entry_price <= 0 or cp <= 0 or side not in ("long", "short"):
            return

        old_sl = float(pos.get("sl_price") or 0.0)

        if side == "long":
            new_sl = max(old_sl, entry_price)
        else:
            if old_sl > 0:
                new_sl = min(old_sl, entry_price)
            else:
                new_sl = entry_price

        pos["sl_price"] = float(new_sl)
        self._set_position(sym, pos)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][PROTECT] symbol=%s side=%s old_sl=%.6f new_sl=%.6f reason=%s",
                    sym,
                    side,
                    float(old_sl),
                    float(new_sl),
                    str(reason),
                )
        except Exception:
            pass

    def _reduce_position_partial(
        self,
        symbol: str,
        pos: Dict[str, Any],
        reduce_frac: float,
        current_price: float,
        reason: str,
    ) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()
        side = str(pos.get("side") or "").strip().lower()
        qty = float(pos.get("qty") or 0.0)

        if side not in ("long", "short") or qty <= 0:
            return {"status": "skip", "reason": "bad_position"}

        frac = max(0.05, min(float(reduce_frac or 0.5), 0.95))
        reduce_qty_raw = qty * frac

        symbol_info = self._get_symbol_info(sym)
        norm_qty, qmeta = self._normalize_order_qty(
            symbol=sym,
            raw_qty=float(reduce_qty_raw),
            price=float(current_price or pos.get("entry_price") or 0.0),
            symbol_info=symbol_info,
        )

        if norm_qty <= 0 or norm_qty >= qty:
            return self._close_position(
                symbol=sym,
                price=float(current_price or 0.0),
                reason=str(reason),
                interval=str(pos.get("interval") or ""),
            ) or {"status": "skip", "reason": "reduce_failed"}

        if not self.dry_run:
            self._exchange_close_market(sym, side, float(norm_qty))

        remaining_qty = max(0.0, float(qty) - float(norm_qty))
        if remaining_qty <= 0:
            return self._close_position(
                symbol=sym,
                price=float(current_price or 0.0),
                reason=str(reason),
                interval=str(pos.get("interval") or ""),
            ) or {"status": "skip", "reason": "reduce_to_zero"}

        pos["qty"] = float(remaining_qty)
        entry_price = float(pos.get("entry_price") or 0.0)
        pos["notional"] = float(entry_price * remaining_qty) if entry_price > 0 else float(pos.get("notional") or 0.0)
        self._set_position(sym, pos)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][REDUCE] symbol=%s side=%s reduced_qty=%.10f remain_qty=%.10f reason=%s qmeta=%s",
                    sym,
                    side,
                    float(norm_qty),
                    float(remaining_qty),
                    str(reason),
                    self._safe_json(qmeta, limit=600),
                )
        except Exception:
            pass

        return {
            "status": "reduced",
            "symbol": sym,
            "side": side,
            "reduced_qty": float(norm_qty),
            "remaining_qty": float(remaining_qty),
            "reason": str(reason),
        }
    # ---------------------------------------------------------
    # position dict / pnl / notify
    # ---------------------------------------------------------
    def _create_position_dict(
        self,
        signal: str,
        symbol: str,
        price: float,
        qty: float,
        notional: float,
        interval: str,
        probs: Dict[str, float],
        extra: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        opened_at = self._utc_now_iso()
        now_ts = time.time()

        trail_pct = self._clip_float(extra.get("trail_pct"), 0.03) or 0.03
        stall_ttl = int(extra.get("stall_ttl_sec", 7200) or 7200)

        sl_price = 0.0
        tp_price = 0.0
        atr_value = 0.0
        bias = self._whale_bias(str(signal), extra)
        sl_mult_adj = 1.0
        tp_mult_adj = 1.0

        pos: Dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "side": str(signal),
            "qty": float(qty),
            "entry_price": float(price),
            "notional": float(notional),
            "interval": str(interval or ""),
            "opened_at": opened_at,
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "trailing_pct": float(trail_pct),
            "stall_ttl_sec": int(stall_ttl),
            "best_pnl_pct": 0.0,
            "last_best_ts": float(now_ts),
            "atr_value": float(atr_value),
            "highest_price": float(price),
            "lowest_price": float(price),
            "meta": {
                "probs": dict(probs or {}),
                "extra": dict(extra or {}),
                "whale_bias_on_open": str(bias),
                "sl_mult_adj": float(sl_mult_adj),
                "tp_mult_adj": float(tp_mult_adj),
            },
        }
        return pos, opened_at

    @staticmethod
    def _calc_pnl(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if qty <= 0:
            return 0.0
        if side == "long":
            return (float(exit_price) - float(entry_price)) * float(qty)
        if side == "short":
            return (float(entry_price) - float(exit_price)) * float(qty)
        return 0.0

    def _notify_position_open(
        self,
        symbol: str,
        interval: str,
        side: str,
        qty: float,
        price: float,
        extra: Dict[str, Any],
    ) -> None:
        bot = getattr(self, "telegram_bot", None)
        if bot is None:
            return

        text = (
            "🚀 POSITION OPENED\n"
            f"symbol={str(symbol).upper()} side={str(side)} qty={float(qty)}\n"
            f"price={float(price)} interval={str(interval or '')}"
        )

        try:
            fn = getattr(bot, "send_message", None)
            if callable(fn):
                out = fn(text)
                self._fire_and_forget(out, label="tg_open")
        except Exception:
            pass

    def _notify_position_close(
        self,
        symbol: str,
        interval: str,
        side: str,
        qty: float,
        price: float,
        realized_pnl: float,
        daily_realized_pnl: float,
    ) -> None:
        bot = getattr(self, "telegram_bot", None)
        if bot is None:
            return

        text = (
            "✅ POSITION CLOSED\n"
            f"symbol={str(symbol).upper()} side={str(side)} qty={float(qty)}\n"
            f"price={float(price)} interval={str(interval or '')}\n"
            f"realized_pnl={float(realized_pnl):.4f} daily_realized_pnl={float(daily_realized_pnl):.4f}"
        )

        try:
            fn = getattr(bot, "send_message", None)
            if callable(fn):
                out = fn(text)
                self._fire_and_forget(out, label="tg_close")
        except Exception:
            pass

    # ---------------------------------------------------------
    # verify / poll
    # ---------------------------------------------------------
    def _verify_position_sync(self, symbol: str) -> Dict[str, Any]:
        sym = str(symbol).upper()
        client = getattr(self, "client", None)
        if client is None:
            return {"verify": "skip", "symbol": sym, "reason": "client_none"}

        try:
            fn = getattr(client, "futures_position_information", None)
            if not callable(fn):
                return {"verify": "skip", "symbol": sym, "reason": "fn_missing"}

            rows = fn(symbol=sym)
            if not isinstance(rows, list):
                rows = [rows]

            for row in rows:
                if not isinstance(row, dict):
                    continue
                amt = self._clip_float(row.get("positionAmt"), 0.0) or 0.0
                if abs(float(amt)) > 0:
                    entry = self._clip_float(row.get("entryPrice"), 0.0) or 0.0
                    unreal = self._clip_float(row.get("unRealizedProfit"), 0.0) or 0.0
                    lev = self._clip_float(row.get("leverage"), 0.0) or 0.0
                    return {
                        "verify": "ok",
                        "symbol": sym,
                        "positionAmt": amt,
                        "entryPrice": entry,
                        "unrealized": unreal,
                        "leverage": lev,
                    }

            return {"verify": "ok", "symbol": sym, "positionAmt": 0.0}
        except Exception as e:
            return {
                "verify": "skip",
                "symbol": sym,
                "reason": "position_verify_failed",
                "err": str(e)[:300],
            }

    def _poll_order_status(
        self,
        symbol: str,
        order_id: Any = None,
        client_order_id: str = "",
        max_wait_s: float = 2.0,
    ) -> Dict[str, Any]:
        if self.dry_run:
            return {"poll": "skip", "reason": "dry_run"}

        sym = str(symbol).upper().strip()
        if not sym:
            return {"poll": "skip", "reason": "empty_symbol"}

        client = getattr(self, "client", None)
        if client is None:
            return {"poll": "skip", "reason": "client_none"}

        fn = getattr(client, "futures_get_order", None)
        if not callable(fn):
            return {"poll": "skip", "reason": "fn_missing"}

        if order_id is None and not str(client_order_id or "").strip():
            return {"poll": "skip", "reason": "no_order_ref"}

        started = time.time()
        t_end = started + max(0.2, float(max_wait_s or 2.0))
        last: Any = None
        last_err = ""

        while time.time() < t_end:
            try:
                payload: Dict[str, Any] = {"symbol": sym}
                if order_id is not None:
                    payload["orderId"] = order_id
                else:
                    payload["origClientOrderId"] = str(client_order_id).strip()

                last = fn(**payload)

                if isinstance(last, dict):
                    status = str(last.get("status", "")).upper().strip()

                    if status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                        return {
                            "poll": "ok",
                            "result": self._summarize_order(last),
                            "terminal_status": status,
                            "waited_sec": round(time.time() - started, 3),
                        }

                    if status in ("NEW", "PARTIALLY_FILLED", "PENDING_NEW"):
                        time.sleep(0.2)
                        continue

                    # tanınmayan ama dict döndü
                    return {
                        "poll": "ok",
                        "result": self._summarize_order(last),
                        "terminal_status": status or "UNKNOWN",
                        "waited_sec": round(time.time() - started, 3),
                    }

                time.sleep(0.2)

            except Exception as e:
                last_err = str(e)[:300]
                time.sleep(0.2)

        if isinstance(last, dict):
            return {
                "poll": "timeout",
                "result": self._summarize_order(last),
                "terminal_status": str(last.get("status", "")).upper().strip(),
                "waited_sec": round(time.time() - started, 3),
                "err": last_err,
            }

        if last_err:
            return {
                "poll": "error",
                "err": last_err,
                "waited_sec": round(time.time() - started, 3),
            }

        return {
            "poll": "timeout",
            "reason": "no_terminal_status",
            "waited_sec": round(time.time() - started, 3),
        }
    # ---------------------------------------------------------
    # exchange order functions
    # ---------------------------------------------------------
    def _exchange_open_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()
        extra0 = extra if isinstance(extra, dict) else {}

        if self.dry_run:
            return {
                "status": "dry_run",
                "symbol": sym,
                "side": str(side),
                "qty": float(qty),
                "reduceOnly": bool(reduce_only),
            }

        client = getattr(self, "client", None)
        if client is None:
            raise RuntimeError("client is None (cannot place order)")

        s = str(side or "").strip().lower()
        if s == "long":
            order_side = "BUY"
        elif s == "short":
            order_side = "SELL"
        else:
            raise ValueError(f"bad side={side}")

        raw_q = self._round_qty(float(qty))

        price_for_norm = self._clip_float(price, None)
        if price_for_norm is None or price_for_norm <= 0:
            price_for_norm = self._resolve_price(symbol=sym)

        if price_for_norm is None or price_for_norm <= 0:
            try:
                fn_mp = getattr(client, "futures_mark_price", None)
                if callable(fn_mp):
                    mp = fn_mp(symbol=sym)
                    if isinstance(mp, dict):
                        price_for_norm = self._clip_float(mp.get("markPrice"), None)
            except Exception:
                price_for_norm = None

        if price_for_norm is None or price_for_norm <= 0:
            raise RuntimeError(f"price_for_norm <= 0 for {sym}")

        symbol_info = self._get_symbol_info(sym)
        if not symbol_info:
            raise RuntimeError(f"symbol_info not found for {sym}")

        q, qmeta = self._normalize_order_qty(
            symbol=sym,
            raw_qty=float(raw_q),
            price=float(price_for_norm),
            symbol_info=symbol_info,
        )

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][QTY][NORMALIZE] symbol=%s raw_qty=%.10f norm_qty=%.10f price=%.6f step=%.10f min_qty=%.10f min_notional=%.6f reject=%s",
                    sym,
                    float(raw_q),
                    float(q),
                    float(price_for_norm),
                    float(qmeta.get("step_size", 0.0) or 0.0),
                    float(qmeta.get("min_qty", 0.0) or 0.0),
                    float(qmeta.get("min_notional", 0.0) or 0.0),
                    str(qmeta.get("reject_reason", "") or "-"),
                )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info("[EXEC][QTY][DECISION] %s", self._safe_json(qmeta, limit=900))
        except Exception:
            pass

        if q <= 0:
            raise ValueError(f"qty invalid after normalization: {self._safe_json(qmeta, limit=900)}")

        position_side = self._side_to_position_side(side)

        target_leverage = self._resolve_target_leverage(extra0)
        target_leverage = int(max(1, min(int(target_leverage), int(self.max_leverage))))
        applied_leverage = int(target_leverage)

        try:
            if self.enable_dynamic_leverage and not reduce_only:
                applied_leverage = self._set_symbol_leverage(sym, int(target_leverage))
            elif reduce_only:
                applied_leverage = int(target_leverage)
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][LEV][FAIL] symbol=%s target=%s reduce_only=%s err=%s",
                        sym,
                        int(target_leverage),
                        bool(reduce_only),
                        str(e),
                    )
            except Exception:
                pass
            applied_leverage = int(target_leverage)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][LEV] symbol=%s side=%s reduce_only=%s target=%s applied=%s positionSide=%s",
                    sym,
                    str(side),
                    bool(reduce_only),
                    int(target_leverage),
                    int(applied_leverage),
                    position_side,
                )
        except Exception:
            pass

        tag = "close" if reduce_only else "open"
        client_oid = self._make_client_order_id(sym, tag)

        payload: Dict[str, Any] = {
            "symbol": sym,
            "side": order_side,
            "type": "MARKET",
            "quantity": float(q),
            "newClientOrderId": client_oid,
        }

        if position_side in ("LONG", "SHORT"):
            payload["positionSide"] = position_side

        if reduce_only and not bool(getattr(self, "hedge_mode_enabled", True)):
            payload["reduceOnly"] = True

        used = "futures_create_order"
        t0 = self._now_ms()
        attempts = int(os.getenv("ORDER_RETRY_ATTEMPTS", "3"))
        base_sleep = float(os.getenv("ORDER_RETRY_SLEEP_S", "0.6"))

        try:
            fn = getattr(client, "futures_create_order", None)
            if callable(fn):
                resp = self._call_with_retry(fn, payload, attempts=attempts, base_sleep=base_sleep)
            else:
                fn = getattr(client, "create_order", None)
                used = "create_order"
                if callable(fn):
                    resp = self._call_with_retry(fn, payload, attempts=attempts, base_sleep=base_sleep)
                else:
                    fn = getattr(client, "new_order", None)
                    used = "new_order"
                    if callable(fn):
                        resp = self._call_with_retry(fn, payload, attempts=attempts, base_sleep=base_sleep)
                    else:
                        raise RuntimeError(
                            "no supported order function on client "
                            "(futures_create_order/create_order/new_order)"
                        )
        except Exception as e:
            dt = self._now_ms() - t0
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][ORDER] %s FAIL | fn=%s symbol=%s side=%s qty=%.10f "
                        "reduceOnly=%s lev=%s dt_ms=%d client_oid=%s payload=%s err=%s",
                        ("CLOSE" if reduce_only else "OPEN"),
                        used,
                        sym,
                        order_side,
                        float(q),
                        bool(reduce_only),
                        int(applied_leverage),
                        int(dt),
                        client_oid,
                        self._safe_json(payload, limit=900),
                        str(e)[:300],
                    )
            except Exception:
                pass
            raise

        dt = self._now_ms() - t0

        summ = self._summarize_order(resp)
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][ORDER] %s OK | fn=%s symbol=%s side=%s qty=%.10f "
                    "reduceOnly=%s lev=%s dt_ms=%d summary=%s",
                    ("CLOSE" if reduce_only else "OPEN"),
                    used,
                    sym,
                    order_side,
                    float(q),
                    bool(reduce_only),
                    int(applied_leverage),
                    int(dt),
                    self._safe_json(summ, limit=900),
                )
        except Exception:
            pass

        try:
            if self.order_poll_status:
                oid = None
                coid = ""
                if isinstance(resp, dict):
                    oid = resp.get("orderId")
                    coid = str(resp.get("clientOrderId") or resp.get("newClientOrderId") or client_oid or "")
                pol = self._poll_order_status(
                    sym,
                    order_id=oid,
                    client_order_id=coid,
                    max_wait_s=self.order_poll_wait_s,
                )
                try:
                    if self.logger:
                        self.logger.info("[EXEC][ORDER][POLL] %s", self._safe_json(pol, limit=900))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.order_verify_position:
                v = self._verify_position_sync(sym)
                try:
                    if self.logger:
                        self.logger.info("[EXEC][VERIFY] %s", self._safe_json(v, limit=900))
                except Exception:
                    pass
        except Exception:
            pass

        return resp

    def _exchange_close_market(self, symbol: str, side: str, qty: float) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        pos_side = str(side or "").strip().lower()

        if pos_side not in ("long", "short"):
            raise ValueError(f"bad close side={side}")

        if self.dry_run:
            return {
                "status": "dry_run",
                "symbol": sym,
                "close_side": pos_side,
                "qty": float(qty),
            }

        client = getattr(self, "client", None)
        if client is None:
            raise RuntimeError("client is None (cannot place close order)")

        # long kapat -> SELL + positionSide=LONG
        # short kapat -> BUY  + positionSide=SHORT
        order_side = "SELL" if pos_side == "long" else "BUY"
        position_side = "LONG" if pos_side == "long" else "SHORT"

        raw_q = self._round_qty(float(qty))

        price_for_norm = self._resolve_price(symbol=sym)
        if price_for_norm is None or price_for_norm <= 0:
            try:
                fn_mp = getattr(client, "futures_mark_price", None)
                if callable(fn_mp):
                    mp = fn_mp(symbol=sym)
                    if isinstance(mp, dict):
                        price_for_norm = self._clip_float(mp.get("markPrice"), None)
            except Exception:
                price_for_norm = None

        if price_for_norm is None or price_for_norm <= 0:
            ex_pos = self._get_exchange_position(sym)
            if isinstance(ex_pos, dict):
                price_for_norm = self._clip_float(ex_pos.get("entry_price"), None)

        if price_for_norm is None or price_for_norm <= 0:
            raise RuntimeError(f"close price_for_norm <= 0 for {sym}")

        symbol_info = self._get_symbol_info(sym)
        if not symbol_info:
            raise RuntimeError(f"symbol_info not found for {sym}")

        q, qmeta = self._normalize_order_qty(
            symbol=sym,
            raw_qty=float(raw_q),
            price=float(price_for_norm),
            symbol_info=symbol_info,
        )

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][CLOSE][QTY] symbol=%s raw_qty=%.10f norm_qty=%.10f price=%.6f step=%.10f min_qty=%.10f min_notional=%.6f reject=%s",
                    sym,
                    float(raw_q),
                    float(q),
                    float(price_for_norm),
                    float(qmeta.get("step_size", 0.0) or 0.0),
                    float(qmeta.get("min_qty", 0.0) or 0.0),
                    float(qmeta.get("min_notional", 0.0) or 0.0),
                    str(qmeta.get("reject_reason", "") or "-"),
                )
        except Exception:
            pass

        if q <= 0:
            raise ValueError(f"close qty invalid after normalization: {self._safe_json(qmeta, limit=900)}")

        client_oid = self._make_client_order_id(sym, "close")

        payload: Dict[str, Any] = {
            "symbol": sym,
            "side": order_side,
            "type": "MARKET",
            "quantity": float(q),
            "newClientOrderId": client_oid,
            "positionSide": position_side,
        }

        if not bool(getattr(self, "hedge_mode_enabled", True)):
            payload["reduceOnly"] = True

        used = "futures_create_order"
        t0 = self._now_ms()
        attempts = int(os.getenv("ORDER_RETRY_ATTEMPTS", "3"))
        base_sleep = float(os.getenv("ORDER_RETRY_SLEEP_S", "0.6"))

        try:
            fn = getattr(client, "futures_create_order", None)
            if callable(fn):
                resp = self._call_with_retry(fn, payload, attempts=attempts, base_sleep=base_sleep)
            else:
                raise RuntimeError("futures_create_order not supported on client")
        except Exception as e:
            dt = self._now_ms() - t0
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][ORDER] CLOSE FAIL | fn=%s symbol=%s side=%s posSide=%s qty=%.10f dt_ms=%d client_oid=%s payload=%s err=%s",
                        used,
                        sym,
                        order_side,
                        position_side,
                        float(q),
                        int(dt),
                        client_oid,
                        self._safe_json(payload, limit=900),
                        str(e)[:300],
                    )
            except Exception:
                pass
            raise

        dt = self._now_ms() - t0
        summ = self._summarize_order(resp)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][ORDER] CLOSE OK | fn=%s symbol=%s side=%s posSide=%s qty=%.10f dt_ms=%d summary=%s",
                    used,
                    sym,
                    order_side,
                    position_side,
                    float(q),
                    int(dt),
                    self._safe_json(summ, limit=900),
                )
        except Exception:
            pass

        try:
            if self.order_poll_status:
                oid = None
                coid = ""
                if isinstance(resp, dict):
                    oid = resp.get("orderId")
                    coid = str(resp.get("clientOrderId") or resp.get("newClientOrderId") or client_oid or "")

                pol = self._poll_order_status(
                    sym,
                    order_id=oid,
                    client_order_id=coid,
                    max_wait_s=self.order_poll_wait_s,
                )

                try:
                    if self.logger:
                        self.logger.info("[EXEC][CLOSE][POLL] %s", self._safe_json(pol, limit=900))
                except Exception:
                    pass

                try:
                    if (
                        isinstance(pol, dict)
                        and pol.get("poll") == "ok"
                        and isinstance(pol.get("result"), dict)
                    ):
                        resp = dict(resp or {})
                        resp["_poll_result"] = dict(pol["result"])

                        avg_price = self._clip_float(pol["result"].get("avgPrice"), None)
                        executed_qty = self._clip_float(pol["result"].get("executedQty"), None)

                        if avg_price is not None and avg_price > 0:
                            resp["avgPrice"] = avg_price
                        if executed_qty is not None and executed_qty > 0:
                            resp["executedQty"] = executed_qty
                except Exception:
                    pass
        except Exception:
            pass

        return resp
    # ---------------------------------------------------------
    # close helpers
    # ---------------------------------------------------------
    def _close_position(
        self,
        symbol: str,
        price: float = 0.0,
        reason: str = "manual",
        interval: str = "",
    ) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        if not sym:
            return None

        pos = self._get_position(sym)
        if not isinstance(pos, dict) or not pos:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE-BLOCK] no open position found | symbol=%s reason=%s interval=%s",
                        sym,
                        str(reason),
                        str(interval or ""),
                    )
            except Exception:
                pass

            try:
                self._del_position(sym)
                self._remove_from_bridge_state(sym)
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] cleanup on missing position executed | symbol=%s",
                        sym,
                    )
            except Exception:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] cleanup on missing position failed | symbol=%s",
                        sym,
                    )
            return None

        side = str(pos.get("side") or "").strip().lower()
        qty = float(pos.get("qty") or 0.0)
        entry_price = float(pos.get("entry_price") or 0.0)

        if side not in ("long", "short") or qty <= 0:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][CLOSE-BLOCK] invalid local position | symbol=%s side=%s qty=%.10f",
                        sym,
                        side,
                        float(qty),
                    )
            except Exception:
                pass
            return None

        close_price = self._resolve_price(symbol=sym, price=price)
        if close_price is None or float(close_price) <= 0:
            close_price = float(entry_price or 0.0)

        exchange_pos_before = self._get_exchange_position(symbol=sym)
        exchange_has_position = isinstance(exchange_pos_before, dict)

        if exchange_has_position:
            try:
                qty = abs(float(exchange_pos_before.get("qty") or qty or 0.0))
            except Exception:
                pass
            try:
                ex_side = str(exchange_pos_before.get("side") or side).strip().lower()
                if ex_side in ("long", "short"):
                    side = ex_side
            except Exception:
                pass
            try:
                ex_entry = float(exchange_pos_before.get("entry_price") or entry_price or 0.0)
                if ex_entry > 0:
                    entry_price = ex_entry
            except Exception:
                pass

        qty_meta = self._normalize_close_quantity(
            symbol=sym,
            qty=float(qty),
            price=float(close_price or 0.0),
        )

        norm_qty = float(qty_meta.get("norm_qty") or 0.0)
        step = float(qty_meta.get("step") or 0.0)
        min_qty = float(qty_meta.get("min_qty") or 0.0)
        min_notional = float(qty_meta.get("min_notional") or 0.0)
        reject = str(qty_meta.get("reject_reason") or "-")

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][CLOSE][QTY] symbol=%s raw_qty=%.10f norm_qty=%.10f price=%.6f step=%.10f min_qty=%.10f min_notional=%.6f reject=%s",
                    sym,
                    float(qty),
                    float(norm_qty),
                    float(close_price or 0.0),
                    float(step),
                    float(min_qty),
                    float(min_notional),
                    reject,
                )
        except Exception:
            pass

        if norm_qty <= 0:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][CLOSE-BLOCK] normalized qty invalid | symbol=%s raw_qty=%.10f norm_qty=%.10f reject=%s",
                        sym,
                        float(qty),
                        float(norm_qty),
                        reject,
                    )
            except Exception:
                pass
            return None

        realized_pnl = 0.0
        if entry_price > 0 and norm_qty > 0:
            if side == "long":
                realized_pnl = (float(close_price) - float(entry_price)) * float(norm_qty)
            else:
                realized_pnl = (float(entry_price) - float(close_price)) * float(norm_qty)

        def _risk_on_close(_close_price: float, _realized_pnl: float) -> None:
            try:
                rm = getattr(self, "risk_manager", None)
                if rm is not None and hasattr(rm, "on_position_close"):
                    rm.on_position_close(
                        symbol=sym,
                        side=side,
                        qty=float(norm_qty),
                        notional=float(pos.get("notional") or (norm_qty * entry_price)),
                        price=float(_close_price),
                        interval=str(interval or pos.get("interval") or ""),
                        realized_pnl=float(_realized_pnl),
                        meta={
                            "reason": str(reason),
                            "entry_price": float(entry_price),
                            "closed_side": side,
                            "interval": str(interval or pos.get("interval") or ""),
                            "qty": float(norm_qty),
                            "notional": float(pos.get("notional") or (norm_qty * entry_price)),
                            "probs": dict(
                                pos.get("meta", {}).get("probs", {})
                                if isinstance(pos.get("meta"), dict)
                                else {}
                            ),
                            "extra": dict(
                                pos.get("meta", {}).get("extra", {})
                                if isinstance(pos.get("meta"), dict)
                                else {}
                            ),
                        },
                    )
            except Exception:
                if self.logger:
                    self.logger.exception("[RISK] on_position_close failed")

        def _cleanup_closed_state() -> None:
            self._del_position(sym)
            self._remove_from_bridge_state(sym)

            try:
                chk_local = self._get_position(sym)
                chk_bridge_all = self._get_bridge_state()
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] cleanup after close | symbol=%s local_exists=%s bridge_exists=%s",
                        sym,
                        bool(isinstance(chk_local, dict) and chk_local),
                        bool(isinstance(chk_bridge_all, dict) and sym in chk_bridge_all),
                    )
            except Exception:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] cleanup verify failed | symbol=%s",
                        sym,
                    )

        if bool(getattr(self, "dry_run", False)):
            _risk_on_close(float(close_price), float(realized_pnl))
            _cleanup_closed_state()

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC] DRY_RUN=True close emri gönderilmeyecek | symbol=%s side=%s qty=%.10f reason=%s",
                        sym,
                        side,
                        float(norm_qty),
                        str(reason),
                    )
            except Exception:
                pass

            return {
                "symbol": sym,
                "side": side,
                "qty": float(norm_qty),
                "close_price": float(close_price),
                "entry_price": float(entry_price),
                "realized_pnl": float(realized_pnl),
                "reason": str(reason),
                "dry_run": True,
                "residual_qty": 0.0,
                "residual_side": "",
            }

        if not exchange_has_position:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][CLOSE] exchange position already missing | symbol=%s side=%s reason=%s -> local cleanup only",
                        sym,
                        side,
                        str(reason),
                    )
            except Exception:
                pass

            _risk_on_close(float(close_price), float(realized_pnl))
            _cleanup_closed_state()

            return {
                "symbol": sym,
                "side": side,
                "qty": float(norm_qty),
                "close_price": float(close_price),
                "entry_price": float(entry_price),
                "realized_pnl": float(realized_pnl),
                "reason": str(reason),
                "order": {},
                "residual_qty": 0.0,
                "residual_side": "",
                "exchange_missing_cleanup": True,
            }

        client = getattr(self, "client", None)
        fn = getattr(client, "futures_create_order", None) if client is not None else None
        if not callable(fn):
            try:
                if self.logger:
                    self.logger.error(
                        "[EXEC][CLOSE-BLOCK] futures_create_order unavailable | symbol=%s",
                        sym,
                    )
            except Exception:
                pass
            return None

        order_side = "SELL" if side == "long" else "BUY"
        position_side = "LONG" if side == "long" else "SHORT"

        started_ms = int(time.time() * 1000)
        order_resp: Dict[str, Any] = {}

        close_kwargs = self._build_close_order_kwargs(
            symbol=sym,
            side=order_side,
            qty=float(self._fmt_qty(sym, norm_qty)),
            position_side=position_side,
            reduce_only=bool(int(os.getenv("CLOSE_REDUCEONLY_DEFAULT", "1"))),
        )
        close_kwargs["newClientOrderId"] = f"b1_close_{sym}_{uuid.uuid4().hex[:12]}"

        try:
            order_resp = fn(**close_kwargs) or {}
        except Exception as e:
            if bool(int(os.getenv("CLOSE_REDUCEONLY_RETRY_ENABLE", "1"))) and self._is_reduceonly_retryable_error(e):
                try:
                    if self.logger:
                        self.logger.warning(
                            "[EXEC][CLOSE] retry without reduceOnly | symbol=%s",
                            sym,
                        )
                except Exception:
                    pass

                try:
                    close_kwargs.pop("reduceOnly", None)
                    order_resp = fn(**close_kwargs) or {}
                except Exception as retry_exc:
                    try:
                        if self.logger:
                            self.logger.exception(
                                "[EXEC][CLOSE-BLOCK] close order failed after retry | symbol=%s side=%s qty=%.10f err=%s",
                                sym,
                                side,
                                float(norm_qty),
                                str(retry_exc),
                            )
                    except Exception:
                        pass
                    return None
            else:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][CLOSE-BLOCK] close order failed | symbol=%s side=%s qty=%.10f err=%s",
                            sym,
                            side,
                            float(norm_qty),
                            str(e),
                        )
                except Exception:
                    pass
                return None

        dt_ms = int(time.time() * 1000) - started_ms
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][ORDER] CLOSE OK | fn=futures_create_order symbol=%s side=%s posSide=%s qty=%.10f dt_ms=%s summary=%s",
                    sym,
                    order_side,
                    position_side,
                    float(norm_qty),
                    int(dt_ms),
                    self._safe_json(order_resp, limit=900),
                )
        except Exception:
            pass

        poll_result = self._poll_order_status(
            symbol=sym,
            order_id=order_resp.get("orderId") if isinstance(order_resp, dict) else None,
            client_order_id=str(order_resp.get("clientOrderId") or "") if isinstance(order_resp, dict) else "",
            max_wait_s=5.0,
        )
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][CLOSE][POLL] %s",
                    self._safe_json(poll_result, limit=900),
                )
        except Exception:
            pass

        filled = {}
        if isinstance(poll_result, dict):
            filled = poll_result.get("result") if isinstance(poll_result.get("result"), dict) else {}
        if not filled:
            filled = order_resp if isinstance(order_resp, dict) else {}

        try:
            avg_price = float(filled.get("avgPrice") or 0.0)
        except Exception:
            avg_price = 0.0
        if avg_price > 0:
            close_price = avg_price

        realized_pnl = 0.0
        if entry_price > 0 and norm_qty > 0:
            if side == "long":
                realized_pnl = (float(close_price) - float(entry_price)) * float(norm_qty)
            else:
                realized_pnl = (float(entry_price) - float(close_price)) * float(norm_qty)

        _risk_on_close(float(close_price), float(realized_pnl))

        exchange_pos_after = self._get_exchange_position(symbol=sym)
        residual_qty = 0.0
        residual_side = side

        if isinstance(exchange_pos_after, dict):
            try:
                residual_qty = abs(float(exchange_pos_after.get("qty") or 0.0))
            except Exception:
                residual_qty = 0.0
            try:
                residual_side = str(exchange_pos_after.get("side") or side).strip().lower() or side
            except Exception:
                residual_side = side

        dust_threshold = max(float(min_qty or 0.0), 0.0)

        if residual_qty <= 0.0:
            _cleanup_closed_state()

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE] CLOSED | symbol=%s side=%s qty=%.10f entry=%.6f close=%.6f realized_pnl=%.6f reason=%s",
                        sym,
                        side,
                        float(norm_qty),
                        float(entry_price),
                        float(close_price),
                        float(realized_pnl),
                        str(reason),
                    )
            except Exception:
                pass
        else:
            residual_pos = dict(pos) if isinstance(pos, dict) else {}
            residual_pos["symbol"] = sym
            residual_pos["side"] = residual_side
            residual_pos["qty"] = float(residual_qty)
            residual_pos["last_price"] = float(close_price)

            if isinstance(exchange_pos_after, dict):
                try:
                    residual_entry = float(
                        exchange_pos_after.get("entry_price")
                        or residual_pos.get("entry_price")
                        or 0.0
                    )
                    if residual_entry > 0:
                        residual_pos["entry_price"] = residual_entry
                except Exception:
                    pass

            self._set_position(sym, residual_pos)
            self._upsert_bridge_state_on_open(
                symbol=sym,
                side=residual_side,
                interval=str(interval or residual_pos.get("interval") or "5m"),
                intent_id="",
            )

            try:
                chk_local = self._get_position(sym)
                chk_bridge_all = self._get_bridge_state()
                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] partial close state updated | symbol=%s residual_qty=%.10f local_exists=%s bridge_exists=%s",
                        sym,
                        float(residual_qty),
                        bool(isinstance(chk_local, dict) and chk_local),
                        bool(isinstance(chk_bridge_all, dict) and sym in chk_bridge_all),
                    )
            except Exception:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] partial close verify failed | symbol=%s",
                        sym,
                    )

            try:
                if self.logger:
                    if dust_threshold > 0.0 and residual_qty <= dust_threshold:
                        self.logger.warning(
                            "[EXEC][CLOSE] residual dust remains on exchange | symbol=%s side=%s residual_qty=%.10f dust_threshold=%.10f",
                            sym,
                            residual_side,
                            float(residual_qty),
                            float(dust_threshold),
                        )
                    else:
                        self.logger.warning(
                            "[EXEC][CLOSE] residual position remains on exchange | symbol=%s side=%s residual_qty=%.10f",
                            sym,
                            residual_side,
                            float(residual_qty),
                        )
            except Exception:
                pass

        return {
            "symbol": sym,
            "side": side,
            "qty": float(norm_qty),
            "close_price": float(close_price),
            "entry_price": float(entry_price),
            "realized_pnl": float(realized_pnl),
            "reason": str(reason),
            "order": filled if isinstance(filled, dict) else {},
            "residual_qty": float(residual_qty),
            "residual_side": str(residual_side),
        }

    def close_position(
        self,
        symbol: str,
        price: Optional[float] = None,
        reason: str = "manual",
        interval: str = "",
        intent_id: Optional[str] = None,
        **_ignored: Any,
    ) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper().strip()
        if not sym:
            return None

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][CLOSE] request | symbol=%s reason=%s interval=%s intent_id=%s input_price=%s",
                    sym,
                    str(reason or "manual"),
                    str(interval or ""),
                    str(intent_id or ""),
                    str(price),
                )
        except Exception:
            pass

        p = self._resolve_price(symbol=sym, price=price)
        if p is None:
            p = 0.0

        return self._close_position(
            symbol=sym,
            price=float(p),
            reason=str(reason or "manual"),
            interval=str(interval or ""),
        )

    # ---------------------------------------------------------
    # monitoring stubs
    # ---------------------------------------------------------
    def _get_latest_signal_for_symbol(self, symbol: str) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()
        if not sym:
            return {}

        r = getattr(self, "redis", None) or getattr(self, "redis_client", None)
        if r is None:
            return {}

        try:
            key = f"bot:last_signal:{sym}"
            raw = r.get(key)
            if not raw:
                return {}

            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8", errors="ignore")

            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][WEAK] latest signal read failed | symbol=%s",
                        sym,
                    )
            except Exception:
                pass
            return {}

    def _check_opposite_signal_close(
        self,
        symbol: str,
        side: str,
        interval: str,
        price: float,
    ) -> Optional[str]:
        try:
            enabled = bool(int(str(os.getenv("OPPOSITE_SIGNAL_CLOSE_ENABLE", "1")).strip()))
        except Exception:
            enabled = True

        if not enabled:
            return None

        try:
            min_score = float(str(os.getenv("OPPOSITE_SIGNAL_MIN_SCORE", "0.58")).strip())
        except Exception:
            min_score = 0.58

        sig = self._get_latest_signal_for_symbol(symbol)
        if not isinstance(sig, dict) or not sig:
            return None

        sig_side = str(sig.get("side") or "").strip().lower()
        sig_interval = str(sig.get("interval") or "").strip()
        sig_score = float(sig.get("score") or 0.0)

        if sig_interval and interval and sig_interval != interval:
            return None

        if side == "long" and sig_side == "short" and sig_score >= min_score:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][WEAK] opposite signal close trigger | symbol=%s pos_side=%s sig_side=%s score=%.6f interval=%s price=%.6f",
                        str(symbol).upper().strip(),
                        side,
                        sig_side,
                        float(sig_score),
                        str(interval or ""),
                        float(price),
                    )
            except Exception:
                pass
            return "opposite_signal"

        if side == "short" and sig_side == "long" and sig_score >= min_score:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][WEAK] opposite signal close trigger | symbol=%s pos_side=%s sig_side=%s score=%.6f interval=%s price=%.6f",
                        str(symbol).upper().strip(),
                        side,
                        sig_side,
                        float(sig_score),
                        str(interval or ""),
                        float(price),
                    )
            except Exception:
                pass
            return "opposite_signal"

        return None

    def _check_sl_tp_trailing(self, symbol: str, price: float, interval: str) -> None:
        sym = str(symbol).upper().strip()
        pos = self._get_position(sym)
        if not isinstance(pos, dict):
            return

        side = str(pos.get("side") or "").strip().lower()
        qty = float(pos.get("qty") or 0.0)
        entry_price = float(pos.get("entry_price") or 0.0)

        if side not in ("long", "short") or qty <= 0 or price <= 0 or entry_price <= 0:
            return

        now_ts = time.time()

        highest_price = float(pos.get("highest_price") or entry_price)
        lowest_price = float(pos.get("lowest_price") or entry_price)
        best_pnl_pct = float(pos.get("best_pnl_pct") or 0.0)
        last_best_ts = float(pos.get("last_best_ts") or now_ts)

        trailing_pct = float(pos.get("trailing_pct") or 0.0)
        stall_ttl_sec = int(pos.get("stall_ttl_sec") or 0)

        sl_price = float(pos.get("sl_price") or 0.0)
        tp_price = float(pos.get("tp_price") or 0.0)

        sl_pct = float(getattr(self, "default_sl_pct", 0.0) or 0.0)
        tp_pct = float(getattr(self, "default_tp_pct", 0.0) or 0.0)
        trailing_activation_pct = float(
            getattr(self, "trailing_activation_pct", 0.0) or 0.0
        )

        tp_runner_enable = bool(getattr(self, "tp_runner_enable", True))
        tp_arm_pnl_pct = float(getattr(self, "tp_arm_pnl_pct", 0.02) or 0.02)
        tp_runner_pullback_pct = float(
            getattr(self, "tp_runner_pullback_pct", 0.006) or 0.006
        )

        hard_stop_enable = bool(getattr(self, "hard_stop_enable", True))
        hard_stop_loss_pct = float(getattr(self, "hard_stop_loss_pct", 0.04) or 0.04)

        profit_protect_enable = bool(getattr(self, "profit_protect_enable", True))
        profit_protect_arm_pct = float(
            getattr(self, "profit_protect_arm_pct", 0.02) or 0.02
        )
        profit_protect_pullback_pct = float(
            getattr(self, "profit_protect_pullback_pct", 0.004) or 0.004
        )
        profit_protect_reverse_score = float(
            getattr(self, "profit_protect_reverse_score", 0.55) or 0.55
        )

        tp_runner_armed = bool(pos.get("tp_runner_armed", False))
        tp_runner_armed_ts = float(pos.get("tp_runner_armed_ts") or 0.0)

        close_reason = None
        trailing_armed = False
        trail_stop_price = 0.0
        pnl_pct = 0.0

        if sl_price <= 0 and sl_pct > 0:
            sl_price = (
                entry_price * (1 - sl_pct)
                if side == "long"
                else entry_price * (1 + sl_pct)
            )

        if tp_price <= 0 and tp_pct > 0:
            tp_price = (
                entry_price * (1 + tp_pct)
                if side == "long"
                else entry_price * (1 - tp_pct)
            )

        if side == "long":
            highest_price = max(highest_price, price)
            pnl_pct = (price - entry_price) / max(entry_price, 1e-12)

            if pnl_pct > best_pnl_pct:
                best_pnl_pct = pnl_pct
                last_best_ts = now_ts

            if tp_runner_enable and best_pnl_pct >= tp_arm_pnl_pct:
                tp_runner_armed = True
                if tp_runner_armed_ts <= 0:
                    tp_runner_armed_ts = now_ts

            trailing_armed = (
                trailing_pct > 0
                and highest_price > entry_price
                and (
                    trailing_activation_pct <= 0
                    or best_pnl_pct >= trailing_activation_pct
                )
            )
            trail_stop_price = highest_price * (1 - trailing_pct) if trailing_armed else 0.0

            protect_armed = (
                profit_protect_enable and best_pnl_pct >= profit_protect_arm_pct
            )

            if close_reason is None and protect_armed:
                pullback = (
                    (highest_price - price) / max(highest_price, 1e-12)
                    if highest_price > 0
                    else 0.0
                )

                reverse_meta = self._evaluate_position_signal_exit(
                    symbol=sym,
                    pos=pos,
                    price=float(price),
                ) or {}

                reverse_hit = (
                    isinstance(reverse_meta, dict)
                    and reverse_meta.get("reason") == "reverse_signal"
                    and float(reverse_meta.get("signal_score", 0.0))
                    >= profit_protect_reverse_score
                )

                if reverse_hit:
                    close_reason = "profit_protect_reverse"
                elif pullback >= profit_protect_pullback_pct:
                    close_reason = "profit_protect_pullback"

            if close_reason is None:
                if hard_stop_enable and pnl_pct <= -abs(hard_stop_loss_pct):
                    close_reason = "hard_stop_hit"
                elif sl_price > 0 and price <= sl_price:
                    close_reason = "sl_hit"
                elif trailing_armed and price <= trail_stop_price:
                    close_reason = "trailing_hit"
                elif tp_runner_armed and (best_pnl_pct - pnl_pct) >= tp_runner_pullback_pct:
                    close_reason = "tp_runner_exit"
                elif tp_price > 0 and price >= tp_price:
                    close_reason = "tp_hit"
        else:
            lowest_price = min(lowest_price, price)
            pnl_pct = (entry_price - price) / max(entry_price, 1e-12)

            if pnl_pct > best_pnl_pct:
                best_pnl_pct = pnl_pct
                last_best_ts = now_ts

            trailing_armed = trailing_pct > 0 and lowest_price < entry_price
            trail_stop_price = lowest_price * (1 + trailing_pct) if trailing_armed else 0.0

            protect_armed = (
                profit_protect_enable and best_pnl_pct >= profit_protect_arm_pct
            )

            if close_reason is None and protect_armed:
                bounce = (
                    (price - lowest_price) / max(lowest_price, 1e-12)
                    if lowest_price > 0
                    else 0.0
                )

                reverse_meta = self._evaluate_position_signal_exit(
                    symbol=sym,
                    pos=pos,
                    price=float(price),
                ) or {}

                reverse_hit = (
                    isinstance(reverse_meta, dict)
                    and reverse_meta.get("reason") == "reverse_signal"
                    and float(reverse_meta.get("signal_score", 0.0))
                    >= profit_protect_reverse_score
                )

                if reverse_hit:
                    close_reason = "profit_protect_reverse"
                elif bounce >= profit_protect_pullback_pct:
                    close_reason = "profit_protect_pullback"

            if close_reason is None:
                if hard_stop_enable and pnl_pct <= -abs(hard_stop_loss_pct):
                    close_reason = "hard_stop_hit"
                elif sl_price > 0 and price >= sl_price:
                    close_reason = "sl_hit"
                elif trailing_armed and price >= trail_stop_price:
                    close_reason = "trailing_hit"
                elif tp_price > 0 and price <= tp_price:
                    close_reason = "tp_hit"

        scalp_reason = None
        try:
            scalp_reason = self._evaluate_scalp_exit(
                symbol=sym,
                pos=pos,
                price=float(price),
                side=side,
                entry_price=float(entry_price),
                best_pnl_pct=float(best_pnl_pct),
                highest_price=float(highest_price),
                lowest_price=float(lowest_price),
                now_ts=float(now_ts),
                last_best_ts=float(last_best_ts),
            )
        except Exception as e:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][SCALP] evaluate failed | symbol=%s err=%s",
                        sym,
                        str(e),
                    )
            except Exception:
                pass

        if close_reason is None and scalp_reason:
            close_reason = str(scalp_reason)
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][SCALP] exit trigger | symbol=%s side=%s reason=%s price=%.6f entry=%.6f pnl_pct=%.6f best_pnl_pct=%.6f",
                        sym,
                        side,
                        str(close_reason),
                        float(price),
                        float(entry_price),
                        float(pnl_pct),
                        float(best_pnl_pct),
                    )
            except Exception:
                pass

        if close_reason is None and getattr(self, "weak_signal_enabled", False):
            weak = self._evaluate_position_signal_exit(sym, pos, price)
            if weak:
                close_reason = weak.get("reason", "weak_exit")

        if close_reason is None:
            stalled_for = now_ts - last_best_ts
            if stall_ttl_sec > 0 and stalled_for >= stall_ttl_sec:
                close_reason = "stall_exit"

        pos.update({
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "best_pnl_pct": best_pnl_pct,
            "last_best_ts": last_best_ts,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "tp_runner_armed": tp_runner_armed,
            "tp_runner_armed_ts": tp_runner_armed_ts,
            "last_price": price,
        })

        self._set_position(sym, pos)

        if close_reason:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE][TRIGGER] symbol=%s reason=%s pnl=%.6f best=%.6f trailing_armed=%s trail_stop=%.6f",
                        sym,
                        str(close_reason),
                        float(pnl_pct),
                        float(best_pnl_pct),
                        bool(trailing_armed),
                        float(trail_stop_price),
                    )
            except Exception:
                pass

            self.close_position(
                symbol=sym,
                price=float(price),
                reason=str(close_reason),
                interval=str(interval or pos.get("interval") or ""),
            )

        # ===== ROI PROFIT LOCK =====
        profit_lock_reason = None
        try:
            lev = float(self._resolve_position_leverage(pos))
            roi_pct = float(pnl_pct) * lev
            best_roi_pct = float(best_pnl_pct) * lev

            if side == "long":
                retrace_pct = (
                    (float(highest_price) - float(price)) / max(float(highest_price), 1e-12)
                    if float(highest_price) > 0
                    else 0.0
                )
            else:
                retrace_pct = (
                    (float(price) - float(lowest_price)) / max(float(lowest_price), 1e-12)
                    if float(lowest_price) > 0
                    else 0.0
                )

            retrace_roi_pct = float(retrace_pct) * lev

            profit_lock_arm_roi_pct = float(
                getattr(self, "profit_lock_arm_roi_pct", 0.0175) or 0.0175
            )
            profit_lock_retrace_roi_pct = float(
                getattr(self, "profit_lock_retrace_roi_pct", 0.0045) or 0.0045
            )
            profit_floor_roi_pct = float(
                getattr(self, "profit_floor_roi_pct", 0.0040) or 0.0040
            )

            if (
                float(best_roi_pct) >= float(profit_lock_arm_roi_pct)
                and float(retrace_roi_pct) >= float(profit_lock_retrace_roi_pct)
            ):
                profit_lock_reason = "profit_lock_roi"

            if (
                profit_lock_reason is None
                and float(best_roi_pct) >= float(profit_lock_arm_roi_pct)
                and float(roi_pct) <= float(profit_floor_roi_pct)
            ):
                profit_lock_reason = "profit_floor_roi_exit"

            if profit_lock_reason is not None:
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][PROFIT-LOCK] trigger | symbol=%s side=%s reason=%s roi_pct=%.6f best_roi_pct=%.6f retrace_roi_pct=%.6f lev=%.2f",
                            sym,
                            side,
                            str(profit_lock_reason),
                            float(roi_pct),
                            float(best_roi_pct),
                            float(retrace_roi_pct),
                            float(lev),
                        )
                except Exception:
                    pass
        except Exception as e:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][PROFIT-LOCK] evaluate failed | symbol=%s err=%s",
                        sym,
                        str(e),
                    )
            except Exception:
                pass

        if close_reason is None and profit_lock_reason is not None:
            close_reason = str(profit_lock_reason)

        # ===== USDT LOSS KILL =====
        try:
            qty_now = float(pos.get("qty") or 0.0)
        except Exception:
            qty_now = 0.0

        unrealized_pnl_usdt = self._calc_unrealized_pnl_usdt(
            side=side,
            entry_price=float(entry_price),
            price=float(price),
            qty=float(qty_now),
        )

        if (
            close_reason is None
            and bool(getattr(self, "usdt_loss_kill_enable", True))
            and float(unrealized_pnl_usdt) <= -abs(float(getattr(self, "max_loss_usdt_per_trade", 4.0) or 4.0))
        ):
            close_reason = "usdt_loss_kill"
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][LOSS-KILL] trigger | symbol=%s side=%s pnl_usdt=%.4f threshold=%.4f",
                        sym,
                        side,
                        float(unrealized_pnl_usdt),
                        float(getattr(self, "max_loss_usdt_per_trade", 4.0) or 4.0),
                    )
            except Exception:
                pass

        # ===== DEAD TRADE EXIT =====
        dead_trade_reason = None
        try:
            if bool(getattr(self, "dead_trade_exit_enable", True)):
                opened_ts = 0.0
                try:
                    opened_at_raw = str(pos.get("opened_at") or "").strip()
                    if opened_at_raw:
                        opened_ts = datetime.fromisoformat(
                            opened_at_raw.replace("Z", "+00:00")
                        ).timestamp()
                except Exception:
                    opened_ts = 0.0

                if opened_ts > 0:
                    alive_sec = float(now_ts) - float(opened_ts)
                    max_sec = float(getattr(self, "dead_trade_max_sec", 90) or 90)
                    min_best = float(
                        getattr(self, "dead_trade_min_best_pnl_pct", 0.002) or 0.002
                    )

                    if (
                        float(alive_sec) >= float(max_sec)
                        and float(best_pnl_pct) < float(min_best)
                    ):
                        dead_trade_reason = "dead_trade_exit"

                        try:
                            if self.logger:
                                self.logger.info(
                                    "[EXEC][DEAD-TRADE] trigger | symbol=%s side=%s alive_sec=%.1f best_pnl_pct=%.6f min_best=%.6f",
                                    sym,
                                    side,
                                    float(alive_sec),
                                    float(best_pnl_pct),
                                    float(min_best),
                                )
                        except Exception:
                            pass
        except Exception as e:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][DEAD-TRADE] evaluate failed | symbol=%s err=%s",
                        sym,
                        str(e),
                    )
            except Exception:
                pass

        if close_reason is None and dead_trade_reason is not None:
            close_reason = str(dead_trade_reason)

    def _handle_weak_signal_position(self, symbol: str, price: float, interval: str) -> None:
        sym = str(symbol).upper().strip()
        pos = self._get_position(sym)
        if not isinstance(pos, dict):
            return

        side = str(pos.get("side") or "").strip().lower()
        qty = float(pos.get("qty") or 0.0)
        entry_price = float(pos.get("entry_price") or 0.0)

        if side not in ("long", "short") or qty <= 0 or price <= 0 or entry_price <= 0:
            return

        meta = pos.get("meta", {}) or {}
        probs = meta.get("probs", {}) if isinstance(meta, dict) else {}
        extra = meta.get("extra", {}) if isinstance(meta, dict) else {}

        p_used = None
        try:
            if isinstance(probs, dict):
                p_used = probs.get("p_used")
            if p_used is None and isinstance(extra, dict):
                p_used = (
                    extra.get("p_buy_ema")
                    or extra.get("p_buy_raw")
                    or extra.get("ensemble_p")
                    or extra.get("p_used")
                )
            p_used = float(p_used) if p_used is not None else None
        except Exception:
            p_used = None

        if p_used is None:
            return

        if side == "long":
            pnl_pct = ((float(price) - entry_price) / entry_price) if entry_price > 0 else 0.0
        else:
            pnl_pct = ((entry_price - float(price)) / entry_price) if entry_price > 0 else 0.0

        if pnl_pct < float(self.weak_signal_min_pnl_pct):
            return

        now_ts = time.time()
        weak_meta = extra.get("weak_signal_meta", {}) if isinstance(extra, dict) else {}
        last_action_ts = 0.0
        try:
            if isinstance(weak_meta, dict):
                last_action_ts = float(weak_meta.get("last_action_ts") or 0.0)
        except Exception:
            last_action_ts = 0.0

        if (now_ts - last_action_ts) < float(self.weak_signal_cooldown_sec):
            return

        action = None

        if p_used <= float(self.weak_signal_close_thr):
            action = "close"
        elif p_used <= float(self.weak_signal_reduce_thr):
            action = "reduce"
        elif p_used <= float(self.weak_signal_protect_thr):
            action = "protect"

        if action is None:
            return

        try:
            if self.logger:
                self.logger.info(
                    f"[EXEC][WEAK] trigger | symbol={sym} side={side} action={action} "
                    f"p_used={float(p_used):.6f} pnl_pct={float(pnl_pct):.6f} price={float(price):.6f}"
                )
        except Exception:
            pass

        if action == "close":
            self.close_position(
                symbol=sym,
                price=float(price),
                reason="weak_signal_close",
                interval=str(interval or ""),
            )
            return

        if action == "protect":
            if side == "long":
                new_sl = max(float(pos.get("sl_price") or 0.0), float(entry_price))
            else:
                cur_sl = float(pos.get("sl_price") or 0.0)
                new_sl = float(entry_price) if cur_sl <= 0 else min(cur_sl, float(entry_price))

            pos["sl_price"] = float(new_sl)

            try:
                if not isinstance(extra, dict):
                    extra = {}
                if not isinstance(weak_meta, dict):
                    weak_meta = {}
                weak_meta["last_action"] = "protect"
                weak_meta["last_action_ts"] = float(now_ts)
                weak_meta["last_p_used"] = float(p_used)
                extra["weak_signal_meta"] = weak_meta
                meta["extra"] = extra
                pos["meta"] = meta
            except Exception:
                pass

            self._set_position(sym, pos)

            try:
                if self.logger:
                    self.logger.info(
                        f"[EXEC][PROTECT] breakeven stop updated | symbol={sym} side={side} sl_price={float(new_sl):.6f}"
                    )
            except Exception:
                pass
            return

        if action == "reduce":
            reduce_frac = max(0.05, min(float(self.weak_signal_reduce_frac), 0.95))
            reduce_qty_raw = float(qty) * float(reduce_frac)

            symbol_info = self._get_symbol_info(sym)
            reduce_qty, qmeta = self._normalize_order_qty(
                symbol=sym,
                raw_qty=float(reduce_qty_raw),
                price=float(price),
                symbol_info=symbol_info,
            )

            if reduce_qty <= 0 or reduce_qty >= float(qty):
                try:
                    if self.logger:
                        self.logger.info(
                            f"[EXEC][REDUCE] skipped invalid reduce qty | symbol={sym} qty={float(qty):.10f} "
                            f"reduce_qty_raw={float(reduce_qty_raw):.10f} meta={self._safe_json(qmeta, limit=500)}"
                        )
                except Exception:
                    pass
                return

            try:
                self._exchange_close_market(sym, side, float(reduce_qty))
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][REDUCE] exchange reduce failed | symbol=%s side=%s reduce_qty=%.10f",
                            sym,
                            side,
                            float(reduce_qty),
                        )
                except Exception:
                    pass
                return

            remaining_qty = float(qty) - float(reduce_qty)
            if remaining_qty <= 0:
                self.close_position(
                    symbol=sym,
                    price=float(price),
                    reason="weak_signal_reduce_to_zero",
                    interval=str(interval or ""),
                )
                return

            pos["qty"] = float(remaining_qty)
            pos["notional"] = float(remaining_qty) * float(price)

            try:
                if not isinstance(extra, dict):
                    extra = {}
                if not isinstance(weak_meta, dict):
                    weak_meta = {}
                weak_meta["last_action"] = "reduce"
                weak_meta["last_action_ts"] = float(now_ts)
                weak_meta["last_p_used"] = float(p_used)
                weak_meta["last_reduce_qty"] = float(reduce_qty)
                extra["weak_signal_meta"] = weak_meta
                meta["extra"] = extra
                pos["meta"] = meta
            except Exception:
                pass

            self._set_position(sym, pos)

            try:
                if self.logger:
                    self.logger.info(
                        f"[EXEC][REDUCE] partial reduce executed | symbol={sym} side={side} "
                        f"reduce_qty={float(reduce_qty):.10f} remaining_qty={float(remaining_qty):.10f}"
                    )
            except Exception:
                pass

    def _append_hold_csv(self, row: Dict[str, Any]) -> None:
        return

    def _append_trade_csv(self, row: Dict[str, Any]) -> None:
        return

    def _extract_open_gate_metrics(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        d = self._extract_whale_context(payload if isinstance(payload, dict) else {})

        def _to_float(v: Any, default: float = 0.0) -> float:
            try:
                if v is None or v == "":
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        score = _to_float(d.get("score"), 0.0)
        if score <= 0:
            score = _to_float(d.get("score_total"), 0.0)
        if score <= 0:
            score = _to_float(d.get("_score_total_final"), 0.0)
        if score <= 0:
            score = _to_float(d.get("_score_selected"), 0.0)

        whale_score = _to_float(d.get("whale_score"), 0.0)
        whale_dir_raw = str(d.get("whale_dir", "none") or "none").strip().lower()

        if whale_dir_raw in ("buy", "bull", "up"):
            whale_dir = "long"
        elif whale_dir_raw in ("sell", "bear", "down"):
            whale_dir = "short"
        elif whale_dir_raw in ("long", "short"):
            whale_dir = whale_dir_raw
        else:
            whale_dir = "none"

        return {
            "ctx": d,
            "score": float(score),
            "whale_score": float(whale_score),
            "whale_dir": str(whale_dir),
            "whale_action": self._whale_action(d),
        }

    def _validate_open_signal_gate(
        self,
        symbol: str,
        side: str,
        score: float,
        whale_score: float,
        whale_dir: str,
        whale_action: str,
    ) -> Optional[Dict[str, Any]]:
        sym_u = str(symbol).upper().strip()
        side0 = str(side or "").strip().lower()

        try:
            open_min_score = float(
                os.getenv("OPEN_MIN_SCORE", os.getenv("MIN_SCORE", "0.42")) or 0.42
            )
        except Exception:
            open_min_score = 0.42

        try:
            whale_open_min_score = float(
                os.getenv("WHALE_OPEN_MIN_SCORE", os.getenv("W_MIN", "0.54")) or 0.54
            )
        except Exception:
            whale_open_min_score = 0.54

        require_whale = str(
            os.getenv("REQUIRE_WHALE_FOR_OPEN", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        strict_alignment = str(
            os.getenv("STRICT_WHALE_ALIGNMENT", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        if side0 not in ("long", "short"):
            return {"status": "skip", "reason": "bad_side"}

        if float(score) < float(open_min_score):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] low score | symbol=%s side=%s score=%.4f min_open=%.4f",
                        sym_u,
                        side0,
                        float(score),
                        float(open_min_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "open_score_too_low",
                "symbol": sym_u,
                "side": side0,
            }

        if require_whale:
            if float(whale_score) < float(whale_open_min_score):
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][OPEN-BLOCK] whale score too low | symbol=%s side=%s whale_score=%.4f min_whale=%.4f",
                            sym_u,
                            side0,
                            float(whale_score),
                            float(whale_open_min_score),
                        )
                except Exception:
                    pass
                return {
                    "status": "skip",
                    "reason": "whale_score_too_low",
                    "symbol": sym_u,
                    "side": side0,
                }

            if str(whale_dir) not in ("long", "short"):
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][OPEN-BLOCK] whale required but missing direction | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                            sym_u,
                            side0,
                            str(whale_dir),
                            float(whale_score),
                        )
                except Exception:
                    pass
                return {
                    "status": "skip",
                    "reason": "whale_direction_missing",
                    "symbol": sym_u,
                    "side": side0,
                }

        if strict_alignment and str(whale_dir) in ("long", "short") and str(whale_dir) != side0:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale misaligned | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                        sym_u,
                        side0,
                        str(whale_dir),
                        float(whale_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_misaligned",
                "symbol": sym_u,
                "side": side0,
            }

        if str(whale_action).strip().lower() in {"block", "avoid", "hard_block"}:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale action veto | symbol=%s side=%s whale_action=%s",
                        sym_u,
                        side0,
                        str(whale_action),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_action_block",
                "symbol": sym_u,
                "side": side0,
            }

        return None

    # ---------------------------------------------------------
    # open by intent
    # ---------------------------------------------------------
    def open_position_from_signal(
        self,
        symbol: str,
        side: str,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sym_u = str(symbol).upper().strip()

        side0 = str(side or "long").strip().lower()
        if side0 not in ("long", "short"):
            side0 = "long"

        def _maybe_json(x: Any) -> Optional[Dict[str, Any]]:
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    obj = json.loads(x)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None

        def _merge_pref(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> None:
            if not isinstance(src, dict):
                return
            for k, v in src.items():
                if k not in dst or dst.get(k) in (None, "", 0, 0.0, [], {}):
                    dst[k] = v

        meta_in = meta if isinstance(meta, dict) else {}
        raw1 = _maybe_json(meta_in.get("raw"))
        raw2 = _maybe_json(raw1.get("raw")) if isinstance(raw1, dict) else None
        raw3 = _maybe_json(raw2.get("raw")) if isinstance(raw2, dict) else None

        meta0: Dict[str, Any] = {}
        _merge_pref(meta0, meta_in)
        _merge_pref(meta0, raw1)
        _merge_pref(meta0, raw2)
        _merge_pref(meta0, raw3)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][OPEN] enter | symbol=%s side=%s interval=%s meta_keys=%s",
                    sym_u,
                    side0,
                    str(interval),
                    sorted(list(meta_in.keys())) if isinstance(meta_in, dict) else [],
                )
        except Exception:
            pass

        try:
            open_min_score = float(
                os.getenv("OPEN_MIN_SCORE", os.getenv("MIN_SCORE", "0.42")) or 0.42
            )
        except Exception:
            open_min_score = 0.42

        try:
            whale_open_min_score = float(
                os.getenv("WHALE_OPEN_MIN_SCORE", os.getenv("W_MIN", "0.54")) or 0.54
            )
        except Exception:
            whale_open_min_score = 0.54

        require_whale_for_open = str(
            os.getenv("REQUIRE_WHALE_FOR_OPEN", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        strict_whale_alignment = str(
            os.getenv("STRICT_WHALE_ALIGNMENT", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        signal_score = 0.0
        for k in (
            "signal_score",
            "score",
            "score_total",
            "_score_total_final",
            "_score_selected",
        ):
            try:
                v = float(meta0.get(k) or 0.0)
            except Exception:
                v = 0.0
            if v > signal_score:
                signal_score = v

        try:
            whale_score = float(meta0.get("whale_score") or 0.0)
        except Exception:
            whale_score = 0.0

        whale_dir = str(meta0.get("whale_dir", "none") or "none").strip().lower()
        if whale_dir in ("buy", "bull", "up"):
            whale_dir = "long"
        elif whale_dir in ("sell", "bear", "down"):
            whale_dir = "short"
        elif whale_dir not in ("long", "short"):
            whale_dir = "none"

        whale_action = str(meta0.get("whale_action") or "").strip().lower()
        if not whale_action:
            try:
                whale_action = str(self._whale_action(meta0) or "").strip().lower()
            except Exception:
                whale_action = ""

        whale_bias_now = str(meta0.get("whale_bias") or "").strip().lower()
        if not whale_bias_now:
            whale_bias_now = whale_dir if whale_dir in ("long", "short") else "hold"

        meta0["signal_score"] = float(signal_score)
        meta0["whale_score"] = float(whale_score)
        meta0["whale_dir"] = whale_dir
        meta0["whale_action"] = whale_action
        meta0["whale_bias"] = whale_bias_now

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][WHALE][CTX] symbol=%s side=%s whale_dir=%s whale_score=%.4f signal_score=%.4f require_whale=%s strict_align=%s",
                    sym_u,
                    side0,
                    whale_dir,
                    float(whale_score),
                    float(signal_score),
                    bool(require_whale_for_open),
                    bool(strict_whale_alignment),
                )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][LEV][INTENT] symbol=%s side=%s recommended=%s score=%.4f whale_dir=%s whale_score=%.4f require_whale=%s strict_align=%s",
                    sym_u,
                    side0,
                    str(meta0.get("recommended_leverage")),
                    float(signal_score),
                    whale_dir,
                    float(whale_score),
                    bool(require_whale_for_open),
                    bool(strict_whale_alignment),
                )
        except Exception:
            pass

        if float(signal_score) < float(open_min_score):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] low score | symbol=%s side=%s score=%.4f min_open=%.4f",
                        sym_u,
                        side0,
                        float(signal_score),
                        float(open_min_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "open_score_too_low",
                "symbol": sym_u,
                "side": side0,
            }

        if require_whale_for_open and float(whale_score) < float(whale_open_min_score):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale score too low | symbol=%s side=%s whale_score=%.4f min_whale=%.4f",
                        sym_u,
                        side0,
                        float(whale_score),
                        float(whale_open_min_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_score_too_low",
                "symbol": sym_u,
                "side": side0,
            }

        if require_whale_for_open and whale_dir not in ("long", "short"):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale required but missing direction | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                        sym_u,
                        side0,
                        whale_dir,
                        float(whale_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_direction_missing",
                "symbol": sym_u,
                "side": side0,
            }

        if strict_whale_alignment and whale_dir in ("long", "short") and whale_dir != side0:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale misaligned | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                        sym_u,
                        side0,
                        whale_dir,
                        float(whale_score),
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_misaligned",
                "symbol": sym_u,
                "side": side0,
            }

        if whale_action in {"block", "avoid", "hard_block"}:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][WHALE][OPEN-BLOCK] symbol=%s side=%s whale_dir=%s whale_score=%.3f action=%s",
                        sym_u,
                        side0,
                        whale_dir,
                        float(whale_score),
                        whale_action,
                    )
            except Exception:
                pass
            return {
                "status": "skip",
                "reason": "whale_block",
                "symbol": sym_u,
                "side": side0,
            }
        intent_price = self._resolve_price(
            symbol=sym_u,
            price=meta0.get("price"),
            mark_price=meta0.get("mark_price"),
            last_price=meta0.get("last_price"),
        )
        if intent_price is None or intent_price <= 0:
            try:
                client = getattr(self, "client", None)
                fn = getattr(client, "futures_mark_price", None) if client is not None else None
                if callable(fn):
                    mp = fn(symbol=sym_u)
                    if isinstance(mp, dict):
                        intent_price = self._clip_float(mp.get("markPrice"), None)
            except Exception:
                intent_price = None

        if intent_price is None or intent_price <= 0:
            try:
                intent_price = self._get_cached_mid_price(sym_u)
            except Exception:
                intent_price = None

        if intent_price is None or intent_price <= 0:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][INTENT] missing price -> skip open | symbol=%s side=%s",
                        sym_u,
                        side0,
                    )
            except Exception:
                pass
            return {"status": "skip", "reason": "missing_price"}

        order_price: Optional[float] = None

        try:
            client = getattr(self, "client", None)
            fn = getattr(client, "futures_mark_price", None) if client is not None else None
            if callable(fn):
                mp = fn(symbol=sym_u)
                if isinstance(mp, dict):
                    order_price = self._clip_float(mp.get("markPrice"), None)
        except Exception:
            order_price = None

        if order_price is None or order_price <= 0:
            try:
                order_price = self._get_cached_mid_price(sym_u)
            except Exception:
                order_price = None

        if order_price is None or order_price <= 0:
            order_price = self._resolve_price(
                symbol=sym_u,
                price=meta0.get("price"),
                mark_price=meta0.get("mark_price"),
                last_price=meta0.get("last_price"),
            )

        if order_price is None or order_price <= 0:
            order_price = float(intent_price)

        cur = self._get_effective_position(sym_u)
        cur_side = str(cur.get("side")).lower().strip() if isinstance(cur, dict) else None

        if cur_side in ("long", "short"):
            if cur_side == side0:
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][OPEN-BLOCK] symbol already open same-side | symbol=%s current=%s incoming=%s",
                            sym_u,
                            cur_side,
                            side0,
                        )
                except Exception:
                    pass
                return {
                    "status": "skip",
                    "reason": "symbol_already_open_same_side",
                    "symbol": sym_u,
                    "side": side0,
                }

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][REVERSE] opposite-side intent -> closing current position | symbol=%s current=%s incoming=%s",
                        sym_u,
                        cur_side,
                        side0,
                    )
            except Exception:
                pass

            try:
                close_res = self.close_position(
                    symbol=sym_u,
                    price=float(order_price),
                    reason=f"reverse_to_{side0}",
                    interval=str(interval or ""),
                    intent_id=str(meta0.get("intent_id") or ""),
                )
                return {
                    "status": "closed_for_reverse",
                    "symbol": sym_u,
                    "previous_side": cur_side,
                    "incoming_side": side0,
                    "close_result": close_res,
                }
            except Exception as e:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][REVERSE] close-before-open failed | symbol=%s current=%s incoming=%s err=%s",
                            sym_u,
                            cur_side,
                            side0,
                            str(e)[:300],
                        )
                except Exception:
                    pass
                return {
                    "status": "skip",
                    "reason": "reverse_close_failed",
                    "symbol": sym_u,
                    "side": side0,
                }

        try:
            open_count = int(self._count_open_positions())
            if open_count >= int(self.max_open_positions):
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] max_open_positions reached | symbol=%s side=%s open_count=%s limit=%s",
                        sym_u,
                        side0,
                        int(open_count),
                        int(self.max_open_positions),
                    )
                return {
                    "status": "skip",
                    "reason": "max_open_positions_reached",
                    "symbol": sym_u,
                    "side": side0,
                }
        except Exception:
            pass
        # ===== NO-TRADE ZONE =====
        no_trade_reason = self._in_no_trade_zone(
            symbol=sym_u,
            side=side0,
            signal_score=float(signal_score),
            whale_score=float(whale_score),
            whale_dir=str(whale_dir),
            order_price=float(order_price),
            meta=meta0,
        )

        if no_trade_reason is not None:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] no-trade zone | symbol=%s side=%s reason=%s score=%.4f whale=%.4f dir=%s",
                        sym_u,
                        side0,
                        str(no_trade_reason),
                        float(signal_score),
                        float(whale_score),
                        str(whale_dir),
                    )
            except Exception:
                pass

            return {
                "status": "skip",
                "reason": str(no_trade_reason),
                "symbol": sym_u,
                "side": side0,
            }
        try:
            eq_live = self._get_futures_equity_usdt_sync()
            if eq_live > 0:
                meta0["equity_usdt"] = float(eq_live)
        except Exception:
            pass

        try:
            avail_live = self._get_available_balance_usdt_sync()
            if avail_live > 0:
                meta0["available_balance_usdt"] = float(avail_live)
        except Exception:
            pass
        npct = self._clip_float(meta0.get("recommended_notional_pct"), None)
        if npct is None:
            npct = self._clip_float(meta0.get("notional_pct"), None)
        npct = float(npct) if npct is not None else None

        notional = self._compute_balance_based_notional(
            sym_u,
            side0,
            float(order_price),
            meta0,
        )

        raw_notional = float(notional)
        notional = self._apply_whale_open_adjustments(side0, float(notional), meta0)
        notional = float(min(float(notional), float(self.max_position_notional)))
        notional = float(max(10.0, float(notional)))

        raw_qty = float(notional) / float(order_price)
        raw_qty = self._round_qty(raw_qty)

        symbol_info = self._get_symbol_info(sym_u)
        qty, qmeta = self._normalize_order_qty(
            symbol=sym_u,
            raw_qty=float(raw_qty),
            price=float(order_price),
            symbol_info=symbol_info,
        )
        confirmed = self._confirm_entry_signal(
            symbol=sym_u,
            side=side0,
            interval=str(interval or "5m"),
            score=float(signal_score),
        )

        if not confirmed:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] waiting confirmation | symbol=%s side=%s score=%.4f",
                        sym_u,
                        side0,
                        float(signal_score),
                    )
            except Exception:
                pass

            return {
                "status": "skip",
                "reason": "waiting_entry_confirmation",
                "symbol": sym_u,
                "side": side0,
            }
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][INTENT][QTY] symbol=%s intent_price=%.6f order_price=%.6f "
                    "raw_qty=%.10f norm_qty=%.10f step=%.10f min_qty=%.10f "
                    "min_notional=%.6f reject=%s",
                    sym_u,
                    float(intent_price),
                    float(order_price),
                    float(raw_qty),
                    float(qty),
                    float(qmeta.get("step_size", 0.0) or 0.0),
                    float(qmeta.get("min_qty", 0.0) or 0.0),
                    float(qmeta.get("min_notional", 0.0) or 0.0),
                    str(qmeta.get("reject_reason", "") or "-"),
                )
        except Exception:
            pass

        if qty <= 0:
            return {
                "status": "skip",
                "reason": "bad_qty_after_normalization",
                "meta": qmeta,
            }

        final_notional = float(qty) * float(order_price)

        extra_safe = dict(meta0)
        extra_safe.setdefault("trail_pct", meta0.get("trail_pct", None))
        extra_safe.setdefault("stall_ttl_sec", meta0.get("stall_ttl_sec", None))
        extra_safe["signal_score"] = float(signal_score)
        extra_safe["whale_action"] = whale_action
        extra_safe["whale_score"] = float(whale_score)
        extra_safe["whale_dir"] = whale_dir
        extra_safe["intent_price"] = float(intent_price)
        extra_safe["order_price"] = float(order_price)
        extra_safe["whale_open_notional_before"] = float(raw_notional)
        extra_safe["whale_open_notional_after"] = float(final_notional)
        extra_safe["whale_notional_adjusted"] = bool(
            abs(float(final_notional) - float(raw_notional)) > 1e-12
        )

        target_leverage = self._resolve_target_leverage(extra_safe)
        target_leverage = int(max(1, min(int(target_leverage), int(self.max_leverage))))
        extra_safe["target_leverage"] = int(target_leverage)

        whale_bias_now = self._whale_bias(side=side0, extra=extra_safe)
        extra_safe["whale_bias"] = whale_bias_now

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][WHALE][OPEN-CHECK] symbol=%s side=%s action=%s bias=%s "
                    "whale_dir=%s whale_score=%.3f signal_score=%.4f raw_notional=%.2f "
                    "final_notional=%.2f target_leverage=%s",
                    sym_u,
                    side0,
                    whale_action or "-",
                    whale_bias_now,
                    whale_dir,
                    float(whale_score),
                    float(signal_score),
                    float(raw_notional),
                    float(final_notional),
                    int(target_leverage),
                )
        except Exception:
            pass

        if not self.dry_run:
            try:
                self._exchange_open_market(
                    symbol=sym_u,
                    side=side0,
                    qty=float(qty),
                    price=float(order_price),
                    reduce_only=False,
                    extra=extra_safe,
                )
            except Exception as e:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][INTENT] exchange_open_failed | symbol=%s side=%s "
                            "qty=%.10f price=%.6f err=%s",
                            sym_u,
                            side0,
                            float(qty),
                            float(order_price),
                            str(e)[:300],
                        )
                except Exception:
                    pass
                return {"status": "skip", "reason": "exchange_open_failed"}

        probs: Dict[str, float] = {}
        now_ts = time.time()
        sync_grace_sec = 45.0

        pos, _opened_at = self._create_position_dict(
            signal=side0,
            symbol=sym_u,
            price=float(order_price),
            qty=float(qty),
            notional=float(final_notional),
            interval=str(interval or ""),
            probs=probs,
            extra=extra_safe,
        )

        if not isinstance(pos, dict):
            pos = {}

        pos["created_ts"] = float(now_ts)
        pos["bridge_written_ts"] = 0.0
        pos["sync_grace_until"] = float(now_ts + sync_grace_sec)

        pos["symbol"] = sym_u
        pos["side"] = side0
        pos["qty"] = float(qty)
        pos["entry_price"] = float(pos.get("entry_price") or order_price or 0.0)
        pos["notional"] = float(pos.get("notional") or final_notional or 0.0)
        pos["interval"] = str(pos.get("interval") or interval or "5m").strip() or "5m"
        self._set_position(sym_u, pos)

        try:
            chk_local = self._get_position(sym_u)
            if self.logger:
                self.logger.info(
                    "[EXEC][STATE] local position set | symbol=%s exists=%s created_ts=%.3f grace_until=%.3f",
                    sym_u,
                    bool(isinstance(chk_local, dict) and chk_local),
                    float(now_ts),
                    float(now_ts + sync_grace_sec),
                )
        except Exception:
            pass

        try:
            self._upsert_bridge_state_on_open(
                symbol=sym_u,
                side=side0,
                interval=str(interval or ""),
                intent_id=str(extra_safe.get("intent_id") or ""),
            )
        except Exception:
            pass

        try:
            pos_after = self._get_position(sym_u)
            if isinstance(pos_after, dict) and pos_after:
                now2 = time.time()
                pos_after["bridge_written_ts"] = float(now2)
                pos_after["sync_grace_until"] = float(now2 + sync_grace_sec)
                self._set_position(sym_u, pos_after)

                if self.logger:
                    self.logger.info(
                        "[EXEC][STATE] bridge/local grace refreshed | symbol=%s bridge_written_ts=%.3f grace_until=%.3f",
                        sym_u,
                        float(now2),
                        float(now2 + sync_grace_sec),
                    )
        except Exception:
            try:
                if self.logger:
                    self.logger.exception(
                        "[EXEC][STATE] bridge/local grace refresh failed | symbol=%s",
                        sym_u,
                    )
            except Exception:
                pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][INTENT] OPEN %s | symbol=%s qty=%.10f intent_price=%.6f "
                    "order_price=%.6f notional=%.2f npct=%s lev=%s whale_action=%s "
                    "whale_bias=%s score=%.4f dry_run=%s",
                    side0.upper(),
                    sym_u,
                    float(qty),
                    float(intent_price),
                    float(order_price),
                    float(final_notional),
                    ("-" if npct is None else f"{npct:.4f}"),
                    int(target_leverage),
                    whale_action or "-",
                    whale_bias_now,
                    float(signal_score),
                    self.dry_run,
                )
        except Exception:
            pass

        try:
            self._notify_position_open(
                sym_u,
                str(interval or ""),
                side0,
                float(qty),
                float(order_price),
                extra_safe,
            )
        except Exception:
            pass

        try:
            rm = getattr(self, "risk_manager", None)
            if rm is not None:
                out = rm.on_position_open(
                    symbol=sym_u,
                    side=side0,
                    qty=float(qty),
                    notional=float(final_notional),
                    price=float(order_price),
                    interval=str(interval or ""),
                    meta={
                        "reason": "INTENT_OPEN",
                        **extra_safe,
                        "created_ts": float(now_ts),
                        "bridge_written_ts": float(now_ts),
                        "sync_grace_until": float(now_ts + sync_grace_sec),
                    },
                )
                self._fire_and_forget(out, label="risk_on_open_intent")
        except Exception:
            pass

        return {
            "status": "opened" if not self.dry_run else "dry_run",
            "symbol": sym_u,
            "side": side0,
            "qty": float(qty),
            "price": float(order_price),
            "notional": float(final_notional),
            "trail_pct": float(pos.get("trailing_pct") or 0.0),
            "stall_ttl_sec": int(pos.get("stall_ttl_sec") or 0),
            "target_leverage": int(target_leverage),
            "whale_action": whale_action,
            "whale_bias": whale_bias_now,
            "signal_score": float(signal_score),
            "whale_score": float(whale_score),
            "whale_dir": whale_dir,
            "created_ts": float(pos.get("created_ts") or now_ts),
            "bridge_written_ts": float(pos.get("bridge_written_ts") or now_ts),
            "sync_grace_until": float(
                pos.get("sync_grace_until") or (now_ts + sync_grace_sec)
            ),
        }

    # ---------------------------------------------------------
    # async decision executor
    # ---------------------------------------------------------
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
        extra0 = self._extract_whale_context(extra if isinstance(extra, dict) else {})
        raw_signal = str(signal or "").strip().lower()
        sym_u = str(symbol).upper().strip()

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][DECISION] enter | symbol=%s signal=%s interval=%s price=%s size=%s extra_keys=%s",
                    sym_u,
                    str(signal),
                    str(interval),
                    str(price),
                    str(size),
                    sorted(list((extra or {}).keys())) if isinstance(extra, dict) else [],
                )
        except Exception:
            pass
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][WHALE][CTX] symbol=%s side=%s whale_dir=%s whale_score=%.4f keys=%s",
                    sym_u,
                    raw_signal,
                    str(extra0.get("whale_dir") or "none"),
                    float(extra0.get("whale_score") or 0.0),
                    sorted(list(extra0.keys())),
                )
        except Exception:
            pass

        try:
            open_min_score = float(
                os.getenv("OPEN_MIN_SCORE", os.getenv("MIN_SCORE", "0.42")) or 0.42
            )
        except Exception:
            open_min_score = 0.42

        try:
            whale_open_min_score = float(
                os.getenv("WHALE_OPEN_MIN_SCORE", os.getenv("W_MIN", "0.54")) or 0.54
            )
        except Exception:
            whale_open_min_score = 0.54

        require_whale_for_open = str(
            os.getenv("REQUIRE_WHALE_FOR_OPEN", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        strict_whale_alignment = str(
            os.getenv("STRICT_WHALE_ALIGNMENT", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        signal_score = 0.0
        for k in ("score", "score_total", "_score_total_final", "_score_selected"):
            try:
                v = float(extra0.get(k) or 0.0)
            except Exception:
                v = 0.0
            if v > signal_score:
                signal_score = v

        whale_action = str(self._whale_action(extra0) or "").strip().lower()
        whale_dir = str(extra0.get("whale_dir", "none") or "none").strip().lower()
        whale_score = float(extra0.get("whale_score", 0.0) or 0.0)

        if whale_dir in ("buy", "bull", "up"):
            whale_dir = "long"
        elif whale_dir in ("sell", "bear", "down"):
            whale_dir = "short"
        elif whale_dir not in ("long", "short"):
            whale_dir = "none"

        if raw_signal in ("close", "exit", "flat"):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE] explicit close signal received | symbol=%s signal=%s interval=%s",
                        sym_u,
                        raw_signal,
                        str(interval or ""),
                    )
            except Exception:
                pass

            try:
                self.close_position(
                    symbol=sym_u,
                    price=price,
                    reason=f"signal_{raw_signal}",
                    interval=str(interval or ""),
                    intent_id=str(extra0.get("intent_id") or ""),
                )
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][CLOSE] explicit close execution failed | symbol=%s signal=%s",
                            sym_u,
                            raw_signal,
                        )
                except Exception:
                    pass
            return

        signal_u = self._signal_u_from_any(signal)
        side_norm = self._normalize_side(signal_u)

        try:
            p_used = extra0.get("ensemble_p")
            if p_used is None:
                p_used = extra0.get("p_buy_ema") or extra0.get("p_buy_raw")
            if p_used is None and isinstance(probs, dict):
                p_used = probs.get("p_used") or probs.get("p_single")

            snap = {
                "ts": datetime.utcnow().isoformat(),
                "ts_epoch": time.time(),
                "symbol": sym_u,
                "interval": interval,
                "signal": signal_u,
                "signal_norm": side_norm,
                "signal_source": str(
                    extra0.get("signal_source") or extra0.get("p_buy_source") or ""
                ),
                "p_used": p_used,
                "p_single": probs.get("p_single") if isinstance(probs, dict) else None,
                "p_buy_raw": extra0.get("p_buy_raw"),
                "p_buy_ema": extra0.get("p_buy_ema"),
                "whale_dir": whale_dir,
                "whale_score": whale_score,
                "whale_action": whale_action,
                "signal_score": float(signal_score),
                "extra": extra0,
            }
            self.last_snapshot = snap
            self.last_snapshot_by_symbol[sym_u] = snap
        except Exception:
            pass

        if signal_u == "HOLD":
            try:
                ens = extra0.get("ensemble_p")
                mcf = extra0.get("model_confidence_factor")
                pbe = extra0.get("p_buy_ema")
                pbr = extra0.get("p_buy_raw")

                p_val = ens if ens is not None else (pbe if pbe is not None else pbr)
                p_src = (
                    "ensemble_p" if ens is not None
                    else ("p_buy_ema" if pbe is not None else ("p_buy_raw" if pbr is not None else "none"))
                )

                pv = self._clip_float(p_val, None)
                if pv is not None:
                    pv = max(0.0, min(1.0, pv))

                self._append_hold_csv({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": sym_u,
                    "interval": interval,
                    "signal": "HOLD",
                    "p": pv,
                    "p_source": p_src,
                    "ensemble_p": ens,
                    "model_confidence_factor": mcf,
                    "p_buy_ema": pbe,
                    "p_buy_raw": pbr,
                })
            except Exception:
                pass

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC] Signal=HOLD symbol=%s whale_action=%s whale_dir=%s whale_score=%.3f signal_score=%.4f",
                        sym_u,
                        whale_action or "-",
                        whale_dir,
                        float(whale_score),
                        float(signal_score),
                    )
            except Exception:
                pass
            try:
                cur = self._get_effective_position(sym_u)
                if isinstance(cur, dict):
                    hold_price = self._resolve_price(
                        symbol=sym_u,
                        price=price,
                        mark_price=extra0.get("mark_price"),
                        last_price=extra0.get("last_price"),
                    )

                    if hold_price is not None and hold_price > 0:
                        self._check_sl_tp_trailing(
                            symbol=sym_u,
                            price=float(hold_price),
                            interval=str(interval or ""),
                        )

                    if self.hold_close_enabled:
                        side_cur = str(cur.get("side") or "").strip().lower()
                        entry_cur = float(cur.get("entry_price") or 0.0)

                        if side_cur in ("long", "short") and entry_cur > 0 and hold_price and hold_price > 0:
                            if side_cur == "long":
                                pnl_pct_cur = (
                                    (float(hold_price) - float(entry_cur))
                                    / max(float(entry_cur), 1e-12)
                                )
                            else:
                                pnl_pct_cur = (
                                    (float(entry_cur) - float(hold_price))
                                    / max(float(entry_cur), 1e-12)
                                )

                            if pnl_pct_cur >= float(self.hold_close_min_pnl_pct):
                                self.close_position(
                                    symbol=sym_u,
                                    price=float(hold_price),
                                    reason="hold_take_profit",
                                    interval=str(interval or ""),
                                    intent_id=str(extra0.get("intent_id") or ""),
                                )
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][HOLD] monitor/close check failed | symbol=%s",
                            sym_u,
                        )
                except Exception:
                    pass
            return

        if self._truthy_env("SHADOW_MODE", "0"):
            return
        if training_mode:
            return
        if side_norm not in ("long", "short"):
            return

        if float(signal_score) < float(open_min_score):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] low score | symbol=%s side=%s signal_score=%.4f min_open=%.4f",
                        sym_u,
                        side_norm,
                        float(signal_score),
                        float(open_min_score),
                    )
            except Exception:
                pass
            return

        if require_whale_for_open and float(whale_score) < float(whale_open_min_score):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale score too low | symbol=%s side=%s whale_score=%.4f min_whale=%.4f",
                        sym_u,
                        side_norm,
                        float(whale_score),
                        float(whale_open_min_score),
                    )
            except Exception:
                pass
            return

        if require_whale_for_open and whale_dir not in ("long", "short"):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale required but missing direction | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                        sym_u,
                        side_norm,
                        whale_dir,
                        float(whale_score),
                    )
            except Exception:
                pass
            return

        if strict_whale_alignment and whale_dir in ("long", "short") and whale_dir != side_norm:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] whale misaligned | symbol=%s side=%s whale_dir=%s whale_score=%.4f",
                        sym_u,
                        side_norm,
                        whale_dir,
                        float(whale_score),
                    )
            except Exception:
                pass
            return

        try:
            if self._should_block_open_by_whale(side_norm, extra0):
                if self.logger:
                    self.logger.info(
                        "[EXEC][VETO] WHALE_BLOCK | symbol=%s side=%s whale_dir=%s whale_score=%.3f action=%s -> SKIP",
                        sym_u,
                        side_norm,
                        whale_dir,
                        float(whale_score),
                        whale_action or "-",
                    )
                return
        except Exception:
            pass

        cur = self._get_effective_position(sym_u)
        cur_side = str(cur.get("side")).lower().strip() if isinstance(cur, dict) else None

        if cur_side in ("long", "short"):
            if cur_side == side_norm:
                try:
                    current_price = self._resolve_price(
                        symbol=sym_u,
                        price=price,
                        mark_price=extra0.get("mark_price"),
                        last_price=extra0.get("last_price"),
                    )
                    if current_price is not None and current_price > 0:
                        self._check_sl_tp_trailing(
                            symbol=sym_u,
                            price=float(current_price),
                            interval=str(interval or ""),
                        )
                except Exception:
                    pass

                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][OPEN-BLOCK] symbol already open same-side | symbol=%s current=%s incoming=%s",
                            sym_u,
                            cur_side,
                            side_norm,
                        )
                except Exception:
                    pass
                return

            if not self.reverse_close_enabled:
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][REVERSE] reverse close disabled | symbol=%s current=%s incoming=%s",
                            sym_u,
                            cur_side,
                            side_norm,
                        )
                except Exception:
                    pass
                return

            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][REVERSE] opposite-side signal -> closing current position | symbol=%s current=%s incoming=%s",
                        sym_u,
                        cur_side,
                        side_norm,
                    )
            except Exception:
                pass
            try:
                self.close_position(
                    symbol=sym_u,
                    price=price,
                    reason=f"reverse_to_{side_norm}",
                    interval=str(interval or ""),
                    intent_id=str(extra0.get("intent_id") or ""),
                )
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][REVERSE] close-before-open failed | symbol=%s current=%s incoming=%s",
                            sym_u,
                            cur_side,
                            side_norm,
                        )
                except Exception:
                    pass
            return

        try:
            open_count = int(self._count_open_positions())
            if open_count >= int(self.max_open_positions):
                if self.logger:
                    self.logger.info(
                        "[EXEC][OPEN-BLOCK] max_open_positions reached | symbol=%s side=%s open_count=%s limit=%s",
                        sym_u,
                        side_norm,
                        int(open_count),
                        int(self.max_open_positions),
                    )
                return
        except Exception:
            pass

        intent_price = self._resolve_price(
            symbol=sym_u,
            price=price,
            mark_price=extra0.get("mark_price"),
            last_price=extra0.get("last_price"),
        )
        if intent_price is None or intent_price <= 0:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC] live_price unavailable -> skip | symbol=%s signal=%s",
                        sym_u,
                        signal_u,
                    )
            except Exception:
                pass
            return

        order_price: Optional[float] = None

        try:
            client = getattr(self, "client", None)
            fn = getattr(client, "futures_mark_price", None) if client is not None else None
            if callable(fn):
                mp = fn(symbol=sym_u)
                if isinstance(mp, dict):
                    order_price = self._clip_float(mp.get("markPrice"), None)
        except Exception:
            order_price = None

        if order_price is None or order_price <= 0:
            try:
                order_price = self._get_cached_mid_price(sym_u)
            except Exception:
                order_price = None

        if order_price is None or order_price <= 0:
            order_price = self._resolve_price(
                symbol=sym_u,
                price=price,
                mark_price=extra0.get("mark_price"),
                last_price=extra0.get("last_price"),
            )

        if order_price is None or order_price <= 0:
            order_price = float(intent_price)

        try:
            self._check_sl_tp_trailing(
                symbol=sym_u,
                price=float(order_price),
                interval=interval,
            )
        except Exception:
            pass

        await asyncio.to_thread(
            self.open_position_from_signal,
            sym_u,
            side_norm,
            str(interval or ""),
            dict(extra0, price=float(intent_price), signal_score=float(signal_score)),
        )
        return

    # ---------------------------------------------------------
    # compatibility wrappers
    # ---------------------------------------------------------
    async def open_position(self, *args, **kwargs):
        try:
            pm = getattr(self, "position_manager", None)
            if pm is not None and hasattr(pm, "open_position"):
                res = pm.open_position(*args, **kwargs)
                try:
                    if inspect.isawaitable(res):
                        return await res
                except Exception:
                    pass
                return res
        except Exception:
            try:
                if self.logger:
                    self.logger.exception("[EXEC][OPEN] position_manager open failed")
            except Exception:
                pass

        try:
            if len(args) >= 3:
                symbol = args[0]
                side = args[1]
                interval = args[2]
                meta = kwargs.get("meta", {})
                return await asyncio.to_thread(
                    self.open_position_from_signal,
                    symbol,
                    side,
                    interval,
                    meta if isinstance(meta, dict) else {},
                )
        except Exception:
            try:
                if self.logger:
                    self.logger.exception("[EXEC][OPEN] fallback open_position_from_signal failed")
            except Exception:
                pass

        return None

    async def execute_trade(self, *args, **kwargs):
        return await self.open_position(*args, **kwargs)
