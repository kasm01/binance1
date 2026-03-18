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
from typing import Any, Dict, List, Optional, Tuple

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

        self.exchange_info_ttl_sec = float(os.getenv("EXCHANGE_INFO_TTL_SEC", "300") or 300.0)
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
        self.position_sync_enabled = str(
            os.getenv("POSITION_SYNC_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_sync_interval_sec = int(
            float(os.getenv("POSITION_SYNC_INTERVAL_SEC", "300") or 300)
        )

        self.position_sync_remove_orphans = str(
            os.getenv("POSITION_SYNC_REMOVE_ORPHANS", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_lifecycle_interval_sec = int(
            float(os.getenv("POSITION_LIFECYCLE_INTERVAL_SEC", "15") or 15)
        )

        self.weak_signal_mode = str(
            os.getenv("WEAK_SIGNAL_MODE", "protect")
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

        self.position_lifecycle_enabled = str(
            os.getenv("POSITION_LIFECYCLE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_lifecycle_interval_sec = int(
            float(os.getenv("POSITION_LIFECYCLE_INTERVAL_SEC", "15") or 15)
        )

        self.default_sl_pct = float(os.getenv("SL_PCT", "0.01") or 0.01)
        self.default_tp_pct = float(os.getenv("TP_PCT", "0.02") or 0.02)
        self.default_trailing_pct = float(os.getenv("TRAILING_PCT", "0.03") or 0.03)

        self.trailing_activation_pct = float(
            os.getenv("TRAILING_ACTIVATION_PCT", "0.003") or 0.003
        )  # %0.3 kâra geçince trail arm

        self.stall_min_pnl_pct = float(
            os.getenv("STALL_MIN_PNL_PCT", "0.002") or 0.002
        )  # stall exit için minimum istenen kâr

        self.weak_signal_enabled = str(
            os.getenv("WEAK_SIGNAL_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.weak_signal_grace_sec = int(
            float(os.getenv("WEAK_SIGNAL_GRACE_SEC", "120") or 120)
        )

        self.weak_signal_fresh_sec = int(
            float(os.getenv("WEAK_SIGNAL_FRESH_SEC", "180") or 180)
        )

        self.weak_signal_action = str(
            os.getenv("WEAK_SIGNAL_ACTION", "protect")
        ).strip().lower()  # protect / reduce / close

        self.weak_signal_reduce_frac = float(
            os.getenv("WEAK_SIGNAL_REDUCE_FRAC", "0.50") or 0.50
        )

        self.weak_signal_hold_below = float(
            os.getenv("WEAK_SIGNAL_HOLD_BELOW", "0.55") or 0.55
        )

        self.weak_signal_reverse_above = float(
            os.getenv("WEAK_SIGNAL_REVERSE_ABOVE", "0.60") or 0.60
        )

        self.last_snapshot_by_symbol: Dict[str, Dict[str, Any]] = {}
        self.position_lifecycle_enabled = str(
            os.getenv("POSITION_LIFECYCLE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self.position_lifecycle_interval_sec = int(
            float(os.getenv("POSITION_LIFECYCLE_INTERVAL_SEC", "15") or 15)
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

        self.sl_pct_default = float(os.getenv("SL_PCT", "0.015") or 0.015)
        self.tp_pct_default = float(os.getenv("TP_PCT", "0.025") or 0.025)
        self.trailing_pct_default = float(os.getenv("TRAILING_PCT", "0.03") or 0.03)

        self.stall_close_enabled = str(
            os.getenv("STALL_CLOSE_ENABLED", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        if self.logger:
            try:
                self.logger.info(
                    f"[EXEC][INIT] "
                    f"dry_run={bool(self.dry_run)} "
                    f"base_order_notional={float(self.base_order_notional):.2f} "
                    f"max_position_notional={float(self.max_position_notional):.2f} "
                    f"max_leverage={float(self.max_leverage):.2f} "
                    f"max_open_positions={int(self.max_open_positions)} "
                    f"per_position_balance_pct={float(self.per_position_balance_pct):.4f} "
                    f"block_same_symbol_reentry={bool(self.block_same_symbol_reentry)} "
                    f"opposite_signal_min_score={float(self.opposite_signal_min_score):.4f} "
                    f"opposite_signal_max_age_sec={float(self.opposite_signal_max_age_sec):.1f} "
                    f"opposite_signal_confirm_count={int(self.opposite_signal_confirm_count)}"
                )
            except Exception:
                pass
    # ---------------------------------------------------------
    # basic helpers
    # ---------------------------------------------------------
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

        # 1) Önce futures_account() içindeki positions alanını dene
        try:
            fn_acc = getattr(client, "futures_account", None)
            if callable(fn_acc):
                acc = fn_acc()
                if isinstance(acc, dict):
                    rows = acc.get("positions", []) or []
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

                    if out:
                        return out
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][EXCHANGE-POS] futures_account positions read failed err=%s",
                        str(e)[:300],
                    )
            except Exception:
                pass

        # 2) Fallback: futures_position_information()
        try:
            fn = getattr(client, "futures_position_information", None)
            if not callable(fn):
                return out

            rows = fn()
            if not isinstance(rows, list):
                return out

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
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning(
                        "[EXEC][EXCHANGE-POS] futures_position_information read failed err=%s",
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

                if sym and side in ("long", "short") and qty > 0:
                    out[sym] = row
        except Exception:
            pass

        return out

    def _count_open_positions(self) -> int:
        seen: Dict[str, Dict[str, Any]] = {}

        try:
            ex_rows = self._get_exchange_open_positions()
            for row in ex_rows:
                sym = str(row.get("symbol") or "").upper().strip()
                if sym:
                    seen[sym] = row
        except Exception:
            pass

        try:
            if self.redis is not None:
                keys = self.redis.keys("bot:positions:*") or []
                for key in keys:
                    try:
                        raw = self.redis.get(key)
                        if not raw:
                            continue
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="ignore")
                        obj = json.loads(raw)
                        if not isinstance(obj, dict):
                            continue
                        sym = str(obj.get("symbol") or "").upper().strip()
                        side = str(obj.get("side") or "").lower().strip()
                        qty = float(obj.get("qty") or 0.0)
                        if sym and side in ("long", "short") and qty > 0:
                            seen.setdefault(sym, {"symbol": sym, "side": side, "qty": qty})
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
    def _extract_whale_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(extra or {})
        raw = out.get("raw")
        if isinstance(raw, dict):
            for k in ("whale_score", "whale_dir", "recommended_leverage", "recommended_notional_pct"):
                if k not in out and k in raw:
                    out[k] = raw.get(k)
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
        sym = str(symbol).upper()

        try:
            exch = self._get_exchange_info_cached(force_refresh=False)
            symbols = exch.get("symbols", []) if isinstance(exch, dict) else []
            for s in symbols:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper() == sym:
                    return s
        except Exception:
            pass

        try:
            exch = self._get_exchange_info_cached(force_refresh=True)
            symbols = exch.get("symbols", []) if isinstance(exch, dict) else []
            for s in symbols:
                if isinstance(s, dict) and str(s.get("symbol", "")).upper() == sym:
                    return s
        except Exception as e:
            try:
                if self.logger:
                    self.logger.warning("[EXEC][SYMBOL_INFO] fetch failed | symbol=%s err=%s", sym, str(e))
            except Exception:
                pass

        return {}

    def _normalize_close_quantity(
        self,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        sym = str(symbol).upper().strip()

        out: Dict[str, Any] = {
            "symbol": sym,
            "raw_qty": 0.0,
            "norm_qty": 0.0,
            "price": 0.0,
            "step": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
            "reject_reason": "",
        }

        try:
            raw_qty = float(qty or 0.0)
        except Exception:
            raw_qty = 0.0

        try:
            px = float(price or 0.0)
        except Exception:
            px = 0.0

        out["raw_qty"] = float(raw_qty)
        out["price"] = float(px)

        if not sym or raw_qty <= 0:
            out["reject_reason"] = "qty_invalid"
            return out

        step = 0.0
        min_qty = 0.0
        min_notional = 0.0

        try:
            s_info = self._get_symbol_info(sym)
            if isinstance(s_info, dict):
                step = float(
                    s_info.get("step")
                    or s_info.get("step_size")
                    or 0.0
                )
                min_qty = float(s_info.get("min_qty") or 0.0)
                min_notional = float(s_info.get("min_notional") or 0.0)
        except Exception:
            step = 0.0
            min_qty = 0.0
            min_notional = 0.0

        out["step"] = float(step)
        out["min_qty"] = float(min_qty)
        out["min_notional"] = float(min_notional)

        norm_qty = float(raw_qty)

        try:
            if step > 0:
                precision = 0
                s = f"{step:.12f}".rstrip("0")
                if "." in s:
                    precision = len(s.split(".", 1)[1])

                norm_qty = math.floor(raw_qty / step) * step
                norm_qty = round(norm_qty, precision)
        except Exception:
            norm_qty = float(raw_qty)

        if norm_qty <= 0:
            out["reject_reason"] = "qty_rounded_to_zero"
            out["norm_qty"] = 0.0
            return out

        if min_qty > 0 and norm_qty < min_qty:
            out["reject_reason"] = "below_min_qty"
            out["norm_qty"] = float(norm_qty)
            return out

        if px > 0 and min_notional > 0 and (norm_qty * px) < min_notional:
            out["reject_reason"] = "below_min_notional"
            out["norm_qty"] = float(norm_qty)
            return out

        out["norm_qty"] = float(norm_qty)
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
            "removed_local": [],
            "removed_bridge": [],
            "added_local": [],
            "added_bridge": [],
            "kept": [],
        }

        if not self.position_sync_enabled:
            return summary

        exchange_map = self._get_exchange_open_positions_map()
        summary["exchange_open"] = len(exchange_map)

        local_map = self._get_all_local_positions()
        summary["local_open"] = len(local_map)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][SYNC] start | exchange_open=%s local_open=%s",
                    int(summary["exchange_open"]),
                    int(summary["local_open"]),
                )
        except Exception:
            pass

        # 1) exchange'de var ama local'de yok -> hydrate et
        for sym, ex_pos in exchange_map.items():
            if sym in local_map:
                summary["kept"].append(sym)
                continue

            try:
                bridge_st = self._get_bridge_state(sym)

                hydrated = {
                    "symbol": sym,
                    "side": str(ex_pos.get("side") or "").strip().lower(),
                    "qty": float(ex_pos.get("qty") or 0.0),
                    "entry_price": float(ex_pos.get("entry_price") or 0.0),
                    "notional": float(ex_pos.get("notional") or 0.0),
                    "interval": str(bridge_st.get("interval") or "5m").strip() or "5m",
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "sl_price": 0.0,
                    "tp_price": 0.0,
                    "trailing_pct": float(
                        getattr(self, "default_trailing_pct", 0.03) or 0.03
                    ),
                    "stall_ttl_sec": int(
                        getattr(self, "default_stall_ttl_sec", 7200) or 7200
                    ),
                    "best_pnl_pct": 0.0,
                    "last_best_ts": float(time.time()),
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

                try:
                    chk = self._get_position(sym)
                    if self.logger:
                        self.logger.info(
                            "[EXEC][SYNC] post-set local verify | symbol=%s exists=%s",
                            sym,
                            bool(isinstance(chk, dict) and chk),
                        )
                except Exception:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][SYNC] post-set local verify failed | symbol=%s",
                            sym,
                        )

                # _set_position çalışmasa bile redis fallback'e düşmesi için zorla yaz
                try:
                    r = getattr(self, "redis", None)
                    if r is None:
                        r = getattr(self, "redis_client", None)

                    if r is not None:
                        r.set(
                            f"bot:positions:{sym}",
                            json.dumps(hydrated, ensure_ascii=False, default=str),
                        )
                        if self.logger:
                            self.logger.info(
                                "[EXEC][SYNC] redis fallback write ok | symbol=%s",
                                sym,
                            )
                except Exception:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][SYNC] redis fallback write failed | symbol=%s",
                            sym,
                        )

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
                        self.logger.exception(
                            "[EXEC][SYNC] hydrate failed | symbol=%s",
                            sym,
                        )
                except Exception:
                    pass

        # 2) local'de var ama exchange'de yok -> temizle
        refreshed_local_map = self._get_all_local_positions()

        for sym, pos in refreshed_local_map.items():
            if sym not in exchange_map:
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

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][SYNC] done | exchange_open=%s local_open=%s removed_local=%s removed_bridge=%s added_local=%s added_bridge=%s kept=%s",
                    int(summary["exchange_open"]),
                    int(summary["local_open"]),
                    summary["removed_local"],
                    summary["removed_bridge"],
                    summary["added_local"],
                    summary["added_bridge"],
                    summary["kept"],
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
        interval_sec = max(3, int(getattr(self, "position_lifecycle_interval_sec", 15)))

        while True:
            try:
                positions = self._get_all_local_positions()

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

                        if side not in ("long", "short") or qty <= 0:
                            try:
                                if self.logger:
                                    self.logger.info(
                                        "[EXEC][LIFECYCLE] skip invalid position | symbol=%s side=%s qty=%.10f",
                                        sym_u,
                                        side,
                                        qty,
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

                    except Exception:
                        try:
                            if self.logger:
                                self.logger.exception(
                                    "[EXEC][LIFECYCLE] symbol processing failed | symbol=%s",
                                    sym,
                                )
                        except Exception:
                            pass

            except Exception:
                try:
                    if self.logger:
                        self.logger.exception("[EXEC][LIFECYCLE] loop failed")
                except Exception:
                    pass

            try:
                await asyncio.sleep(interval_sec)
            except Exception:
                await asyncio.sleep(5)

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

        sym = str(symbol).upper()
        client = getattr(self, "client", None)
        if client is None:
            return {"poll": "skip", "reason": "client_none"}

        fn = getattr(client, "futures_get_order", None)
        if not callable(fn):
            return {"poll": "skip", "reason": "fn_missing"}

        t_end = time.time() + float(max_wait_s)
        last: Any = None

        while time.time() < t_end:
            try:
                payload: Dict[str, Any] = {"symbol": sym}
                if order_id is not None:
                    payload["orderId"] = order_id
                elif client_order_id:
                    payload["origClientOrderId"] = client_order_id
                else:
                    return {"poll": "skip", "reason": "no_order_ref"}

                last = fn(**payload)
                if isinstance(last, dict):
                    status = str(last.get("status", "")).upper()
                    if status in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                        return {"poll": "ok", "result": self._summarize_order(last)}
            except Exception as e:
                return {"poll": "error", "err": str(e)[:300]}

            time.sleep(0.2)

        return {"poll": "timeout", "last": self._summarize_order(last)}
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
        if isinstance(exchange_pos_before, dict):
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

        norm_qty = float(qty_meta.get("qty") or 0.0)
        step = float(qty_meta.get("step") or 0.0)
        min_qty = float(qty_meta.get("min_qty") or 0.0)
        min_notional = float(qty_meta.get("min_notional") or 0.0)
        reject = str(qty_meta.get("reject") or "-")

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

        if bool(getattr(self, "dry_run", False)):
            realized_pnl = 0.0
            if entry_price > 0 and norm_qty > 0:
                if side == "long":
                    realized_pnl = (float(close_price) - float(entry_price)) * float(norm_qty)
                else:
                    realized_pnl = (float(entry_price) - float(close_price)) * float(norm_qty)

            try:
                rm = getattr(self, "risk_manager", None)
                if rm is not None and hasattr(rm, "on_position_close"):
                    rm.on_position_close(
                        symbol=sym,
                        side=side,
                        qty=float(norm_qty),
                        notional=float(pos.get("notional") or (norm_qty * entry_price)),
                        price=float(close_price),
                        interval=str(interval or pos.get("interval") or ""),
                        realized_pnl=float(realized_pnl),
                        meta={
                            "reason": str(reason),
                            "entry_price": float(entry_price),
                            "closed_side": side,
                            "interval": str(interval or pos.get("interval") or ""),
                            "qty": float(norm_qty),
                            "notional": float(pos.get("notional") or (norm_qty * entry_price)),
                            "probs": dict(pos.get("meta", {}).get("probs", {}) if isinstance(pos.get("meta"), dict) else {}),
                            "extra": dict(pos.get("meta", {}).get("extra", {}) if isinstance(pos.get("meta"), dict) else {}),
                        },
                    )
            except Exception:
                if self.logger:
                    self.logger.exception("[RISK] on_position_close failed")

            self._del_position(sym)
            self._remove_from_bridge_state(sym)

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
                "realized_pnl": 0.0,
                "reason": str(reason),
                "dry_run": True,
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

        try:
            order_resp = fn(
                symbol=sym,
                side=order_side,
                type="MARKET",
                quantity=self._fmt_qty(sym, norm_qty),
                reduceOnly=True,
                positionSide=position_side,
                newClientOrderId=f"b1_close_{sym}_{uuid.uuid4().hex[:12]}",
            ) or {}
        except Exception as e:
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

        poll_result = self._poll_order_fill(
            symbol=sym,
            order_response=order_resp,
            timeout_sec=5.0,
            sleep_sec=0.20,
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

        try:
            rm = getattr(self, "risk_manager", None)
            if rm is not None and hasattr(rm, "on_position_close"):
                rm.on_position_close(
                    symbol=sym,
                    side=side,
                    qty=float(norm_qty),
                    notional=float(pos.get("notional") or (norm_qty * entry_price)),
                    price=float(close_price),
                    interval=str(interval or pos.get("interval") or ""),
                    realized_pnl=float(realized_pnl),
                    meta={
                        "reason": str(reason),
                        "entry_price": float(entry_price),
                        "closed_side": side,
                        "interval": str(interval or pos.get("interval") or ""),
                        "qty": float(norm_qty),
                        "notional": float(pos.get("notional") or (norm_qty * entry_price)),
                        "probs": dict(pos.get("meta", {}).get("probs", {}) if isinstance(pos.get("meta"), dict) else {}),
                        "extra": dict(pos.get("meta", {}).get("extra", {}) if isinstance(pos.get("meta"), dict) else {}),
                    },
                )
        except Exception:
            if self.logger:
                self.logger.exception("[RISK] on_position_close failed")

        exchange_pos_after = self._get_exchange_position(symbol=sym)
        residual_qty = 0.0
        residual_side = side

        if isinstance(exchange_pos_after, dict):
            try:
                residual_qty = abs(float(exchange_pos_after.get("qty") or 0.0))
            except Exception:
                residual_qty = 0.0
            try:
                residual_side = str(
                    exchange_pos_after.get("side") or side
                ).strip().lower() or side
            except Exception:
                residual_side = side

        min_qty = 0.0
        try:
            s_info = self._get_symbol_info(sym)
            if isinstance(s_info, dict):
                min_qty = float(s_info.get("min_qty") or 0.0)
        except Exception:
            min_qty = 0.0

        dust_threshold = max(min_qty, 0.0)

        if residual_qty <= 0.0:
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

        elif dust_threshold > 0.0 and residual_qty <= dust_threshold:
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

        else:
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

        return {
            "symbol": sym,
            "side": side,
            "qty": float(norm_qty),
            "close_price": float(close_price),
            "entry_price": float(entry_price),
            "realized_pnl": float(realized_pnl),
            "reason": str(reason),
            "order": filled if isinstance(filled, dict) else {},
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
            if intent_id and self.logger:
                self.logger.info(
                    f"[CLOSE] intent_id={intent_id} symbol={sym} price={price} interval={interval}"
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

        highest_price = float(pos.get("highest_price") or entry_price)
        lowest_price = float(pos.get("lowest_price") or entry_price)
        best_pnl_pct = float(pos.get("best_pnl_pct") or 0.0)
        last_best_ts = float(pos.get("last_best_ts") or time.time())

        trailing_pct = float(pos.get("trailing_pct") or 0.0)
        stall_ttl_sec = int(pos.get("stall_ttl_sec") or 0)

        sl_price = float(pos.get("sl_price") or 0.0)
        tp_price = float(pos.get("tp_price") or 0.0)

        if sl_price <= 0:
            try:
                sl_pct = float(getattr(self, "sl_pct_default", 0.0) or 0.0)
            except Exception:
                sl_pct = 0.0

            if sl_pct > 0:
                if side == "long":
                    sl_price = entry_price * (1.0 - sl_pct)
                else:
                    sl_price = entry_price * (1.0 + sl_pct)

        if tp_price <= 0:
            try:
                tp_pct = float(getattr(self, "tp_pct_default", 0.0) or 0.0)
            except Exception:
                tp_pct = 0.0

            if tp_pct > 0:
                if side == "long":
                    tp_price = entry_price * (1.0 + tp_pct)
                else:
                    tp_price = entry_price * (1.0 - tp_pct)

        close_reason = None
        trail_ref_price = entry_price
        trail_stop_price = 0.0

        if side == "long":
            highest_price = max(highest_price, float(price))
            pnl_pct = ((float(price) - entry_price) / entry_price) if entry_price > 0 else 0.0
            trail_ref_price = highest_price

            if trailing_pct > 0:
                trail_stop_price = highest_price * (1.0 - trailing_pct)

            if pnl_pct > best_pnl_pct:
                best_pnl_pct = float(pnl_pct)
                last_best_ts = time.time()

            if sl_price > 0 and float(price) <= float(sl_price):
                close_reason = "sl_hit"
            elif tp_price > 0 and float(price) >= float(tp_price):
                close_reason = "tp_hit"
            elif trailing_pct > 0 and highest_price > entry_price and float(price) <= float(trail_stop_price):
                close_reason = "trailing_hit"

        else:
            lowest_price = min(lowest_price, float(price))
            pnl_pct = ((entry_price - float(price)) / entry_price) if entry_price > 0 else 0.0
            trail_ref_price = lowest_price

            if trailing_pct > 0:
                trail_stop_price = lowest_price * (1.0 + trailing_pct)

            if pnl_pct > best_pnl_pct:
                best_pnl_pct = float(pnl_pct)
                last_best_ts = time.time()

            if sl_price > 0 and float(price) >= float(sl_price):
                close_reason = "sl_hit"
            elif tp_price > 0 and float(price) <= float(tp_price):
                close_reason = "tp_hit"
            elif trailing_pct > 0 and lowest_price < entry_price and float(price) >= float(trail_stop_price):
                close_reason = "trailing_hit"

        if close_reason is None and bool(getattr(self, "reverse_close_enabled", True)):
            opposite_close_reason = self._check_opposite_signal_close(
                symbol=sym,
                side=side,
                interval=str(interval or ""),
                price=float(price),
            )
            if opposite_close_reason is not None:
                close_reason = str(opposite_close_reason)
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][WEAK] close trigger | symbol=%s side=%s reason=%s price=%.6f",
                            sym,
                            side,
                            str(close_reason),
                            float(price),
                        )
                except Exception:
                    pass

        if close_reason is None and stall_ttl_sec > 0 and bool(getattr(self, "stall_close_enabled", True)):
            stalled_for = float(time.time()) - float(last_best_ts)
            if stalled_for >= float(stall_ttl_sec):
                close_reason = "stall_exit"
                try:
                    if self.logger:
                        self.logger.info(
                            "[EXEC][STALL] exit trigger | symbol=%s side=%s stalled_for=%.1fs stall_ttl_sec=%s",
                            sym,
                            side,
                            float(stalled_for),
                            int(stall_ttl_sec),
                        )
                except Exception:
                    pass

        pos["highest_price"] = float(highest_price)
        pos["lowest_price"] = float(lowest_price)
        pos["best_pnl_pct"] = float(best_pnl_pct)
        pos["last_best_ts"] = float(last_best_ts)
        pos["sl_price"] = float(sl_price)
        pos["tp_price"] = float(tp_price)

        self._set_position(sym, pos)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][LIFECYCLE] updated | symbol=%s side=%s price=%.6f entry=%.6f pnl_pct=%.6f best_pnl_pct=%.6f sl=%.6f tp=%.6f trail_ref=%.6f trail_stop=%.6f",
                    sym,
                    side,
                    float(price),
                    float(entry_price),
                    float(pnl_pct),
                    float(best_pnl_pct),
                    float(sl_price),
                    float(tp_price),
                    float(trail_ref_price),
                    float(trail_stop_price),
                )
        except Exception:
            pass

        if close_reason is not None:
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE] lifecycle trigger | symbol=%s side=%s reason=%s price=%.6f",
                        sym,
                        side,
                        str(close_reason),
                        float(price),
                    )
            except Exception:
                pass

            self.close_position(
                symbol=sym,
                price=float(price),
                reason=str(close_reason),
                interval=str(interval or ""),
            )

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
        meta0 = self._extract_whale_context(meta if isinstance(meta, dict) else {})
        sym_u = str(symbol).upper().strip()

        side0 = str(side or "long").strip().lower()
        if side0 not in ("long", "short"):
            side0 = "long"

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

        whale_action = self._whale_action(meta0)
        whale_score = float(meta0.get("whale_score", 0.0) or 0.0)
        whale_dir = str(meta0.get("whale_dir", "none") or "none").strip().lower()

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][LEV][INTENT] symbol=%s side=%s recommended=%s",
                    sym_u,
                    side0,
                    str(meta0.get("recommended_leverage")),
                )
        except Exception:
            pass

        if self._should_block_open_by_whale(side0, meta0):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][WHALE][OPEN-BLOCK] symbol=%s side=%s whale_dir=%s "
                        "whale_score=%.3f action=%s",
                        sym_u,
                        side0,
                        whale_dir,
                        whale_score,
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

        extra = dict(meta0)
        extra.setdefault("trail_pct", meta0.get("trail_pct", None))
        extra.setdefault("stall_ttl_sec", meta0.get("stall_ttl_sec", None))
        extra["whale_action"] = whale_action
        extra["intent_price"] = float(intent_price)
        extra["order_price"] = float(order_price)
        extra["whale_open_notional_before"] = float(raw_notional)
        extra["whale_open_notional_after"] = float(final_notional)
        extra["whale_notional_adjusted"] = bool(
            abs(float(final_notional) - float(raw_notional)) > 1e-12
        )
        target_leverage = self._resolve_target_leverage(extra)
        target_leverage = int(max(1, min(int(target_leverage), int(self.max_leverage))))
        extra["target_leverage"] = int(target_leverage)

        whale_bias_now = self._whale_bias(side=side0, extra=extra)
        extra["whale_bias"] = whale_bias_now
        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][WHALE][OPEN-CHECK] symbol=%s side=%s action=%s bias=%s "
                    "whale_dir=%s whale_score=%.3f raw_notional=%.2f "
                    "final_notional=%.2f target_leverage=%s",
                    sym_u,
                    side0,
                    whale_action or "-",
                    whale_bias_now,
                    whale_dir,
                    whale_score,
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
                    extra=extra,
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

        pos, _opened_at = self._create_position_dict(
            signal=side0,
            symbol=sym_u,
            price=float(order_price),
            qty=float(qty),
            notional=float(final_notional),
            interval=str(interval or ""),
            probs=probs,
            extra=extra,
        )

        self._set_position(sym_u, pos)

        try:
            self._upsert_bridge_state_on_open(
                symbol=sym_u,
                side=side0,   # execute_decision içinde burada side_norm kullan
                interval=str(interval or ""),
                intent_id=str(extra.get("intent_id") or ""),
            )
        except Exception:
            pass

        try:
            self._upsert_bridge_state_on_open(
                symbol=sym_u,
                side=side0,
                interval=str(interval or ""),
                intent_id=str(extra.get("intent_id") or ""),
            )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][INTENT] OPEN %s | symbol=%s qty=%.10f intent_price=%.6f "
                    "order_price=%.6f notional=%.2f npct=%s lev=%s whale_action=%s "
                    "whale_bias=%s dry_run=%s",
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
                extra,
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
                    meta={"reason": "INTENT_OPEN", **extra},
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

        if raw_signal in ("close", "exit", "flat"):
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC][CLOSE] explicit close signal received | symbol=%s signal=%s interval=%s",
                        str(symbol).upper().strip(),
                        raw_signal,
                        str(interval or ""),
                    )
            except Exception:
                pass

            try:
                self.close_position(
                    symbol=str(symbol).upper().strip(),
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
                            str(symbol).upper().strip(),
                            raw_signal,
                        )
                except Exception:
                    pass
            return

        signal_u = self._signal_u_from_any(signal)
        side_norm = self._normalize_side(signal_u)
        sym_u = str(symbol).upper().strip()
        whale_action = self._whale_action(extra0)
        whale_dir = str(extra0.get("whale_dir", "none") or "none").strip().lower()
        whale_score = float(extra0.get("whale_score", 0.0) or 0.0)

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
                "signal_source": str(extra0.get("signal_source") or extra0.get("p_buy_source") or ""),
                "p_used": p_used,
                "p_single": probs.get("p_single") if isinstance(probs, dict) else None,
                "p_buy_raw": extra0.get("p_buy_raw"),
                "p_buy_ema": extra0.get("p_buy_ema"),
                "whale_dir": whale_dir,
                "whale_score": whale_score,
                "whale_action": whale_action,
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
                        "[EXEC] Signal=HOLD symbol=%s whale_action=%s whale_dir=%s whale_score=%.3f",
                        sym_u,
                        whale_action or "-",
                        whale_dir,
                        whale_score,
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
                                pnl_pct_cur = (float(hold_price) - float(entry_cur)) / max(float(entry_cur), 1e-12)
                            else:
                                pnl_pct_cur = (float(entry_cur) - float(hold_price)) / max(float(entry_cur), 1e-12)

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
                        self.logger.exception(f"[EXEC][HOLD] monitor/close check failed | symbol={sym_u}")
                except Exception:
                    pass

            return
        if self._truthy_env("SHADOW_MODE", "0"):
            return
        if training_mode:
            return
        if side_norm not in ("long", "short"):
            return

        try:
            if self._should_block_open_by_whale(side_norm, extra0):
                if self.logger:
                    self.logger.info(
                        "[EXEC][VETO] WHALE_BLOCK | symbol=%s side=%s whale_dir=%s "
                        "whale_score=%.3f action=%s -> SKIP",
                        sym_u,
                        side_norm,
                        whale_dir,
                        whale_score,
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
            self._check_sl_tp_trailing(symbol=sym_u, price=float(order_price), interval=interval)
        except Exception:
            pass

        raw_notional: Optional[float] = None
        symbol_info = self._get_symbol_info(sym_u)

        if size is not None and float(size) > 0:
            raw_qty = self._round_qty(float(size))
            qty, qmeta = self._normalize_order_qty(
                symbol=sym_u,
                raw_qty=float(raw_qty),
                price=float(order_price),
                symbol_info=symbol_info,
            )
            raw_notional = float(raw_qty) * float(order_price)
            notional = float(qty) * float(order_price)
        else:
            try:
                if "equity_usdt" not in extra0 or not float(extra0.get("equity_usdt") or 0.0) > 0.0:
                    extra0["equity_usdt"] = await self._get_futures_equity_usdt()
            except Exception:
                pass

            try:
                avail_live = await self._get_available_balance_usdt()
                if avail_live > 0:
                    extra0["available_balance_usdt"] = float(avail_live)
            except Exception:
                pass

            raw_notional = float(
                self._compute_balance_based_notional(
                    sym_u,
                    side_norm,
                    float(order_price),
                    extra0,
                )
            )
            notional_pre = float(self._apply_whale_open_adjustments(side_norm, float(raw_notional), extra0))
            raw_qty = self._round_qty(float(notional_pre) / float(order_price))

            qty, qmeta = self._normalize_order_qty(
                symbol=sym_u,
                raw_qty=float(raw_qty),
                price=float(order_price),
                symbol_info=symbol_info,
            )
            notional = float(qty) * float(order_price)

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC][QTY][DECISION] symbol=%s intent_price=%.6f order_price=%.6f "
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
            try:
                if self.logger:
                    self.logger.info(
                        "[EXEC] qty<=0 after normalization -> skip | symbol=%s side=%s "
                        "raw_notional=%.4f meta=%s",
                        sym_u,
                        side_norm,
                        float(raw_notional or 0.0),
                        self._safe_json(qmeta, limit=500),
                    )
            except Exception:
                pass
            return

        extra0["whale_action"] = whale_action
        extra0["intent_price"] = float(intent_price)
        extra0["order_price"] = float(order_price)
        extra0["whale_bias"] = self._whale_bias(side=side_norm, extra=extra0)
        extra0["whale_open_notional_before"] = float(raw_notional or 0.0)
        extra0["whale_open_notional_after"] = float(notional)
        extra0["whale_notional_adjusted"] = bool(
            abs(float(notional) - float(raw_notional or 0.0)) > 1e-12
        )

        target_leverage = self._resolve_target_leverage(extra0)
        target_leverage = int(max(1, min(int(target_leverage), int(self.max_leverage))))
        extra0["target_leverage"] = int(target_leverage)

        try:
            ens = extra0.get("ensemble_p")
            mcf = extra0.get("model_confidence_factor")
            pbe = extra0.get("p_buy_ema")
            pbr = extra0.get("p_buy_raw")
            p_src = str(extra0.get("signal_source") or extra0.get("p_buy_source") or "p_used")

            p_val = ens if ens is not None else (pbe if pbe is not None else pbr)
            if p_val is None and isinstance(probs, dict):
                p_val = probs.get("p_used") or probs.get("p_single")

            pv = self._clip_float(p_val, None)
            if pv is not None:
                pv = max(0.0, min(1.0, pv))

            self._append_trade_csv({
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": sym_u,
                "interval": interval,
                "signal": signal_u,
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
                    "[EXEC][OPEN-CHECK] symbol=%s side=%s intent_price=%.6f order_price=%.6f "
                    "raw_notional=%.2f final_notional=%.2f qty=%.10f target_leverage=%s "
                    "whale_action=%s whale_bias=%s whale_dir=%s whale_score=%.3f",
                    sym_u,
                    side_norm,
                    float(intent_price),
                    float(order_price),
                    float(raw_notional or 0.0),
                    float(notional),
                    float(qty),
                    int(target_leverage),
                    whale_action or "-",
                    str(extra0.get("whale_bias") or "neutral"),
                    whale_dir,
                    whale_score,
                )
        except Exception:
            pass

        if not self.dry_run:
            try:
                self._exchange_open_market(
                    symbol=sym_u,
                    side=side_norm,
                    qty=float(qty),
                    price=float(order_price),
                    reduce_only=False,
                    extra=extra0,
                )
            except Exception:
                try:
                    if self.logger:
                        self.logger.exception(
                            "[EXEC][OPEN] exchange open failed -> state set edilmeyecek | symbol=%s",
                            sym_u,
                        )
                except Exception:
                    pass
                return

        pos, _opened_at = self._create_position_dict(
            signal=side_norm,
            symbol=sym_u,
            price=float(order_price),
            qty=float(qty),
            notional=float(notional),
            interval=interval,
            probs=probs,
            extra=extra0,
        )

        self._set_position(sym_u, pos)

        try:
            self._upsert_bridge_state_on_open(
                symbol=sym_u,
                side=side_norm,
                interval=str(interval or ""),
                intent_id=str(extra0.get("intent_id") or ""),
            )
        except Exception:
            pass

        try:
            if self.logger:
                self.logger.info(
                    "[EXEC] OPEN %s | symbol=%s qty=%.10f intent_price=%.6f order_price=%.6f "
                    "notional=%.2f interval=%s lev=%s whale_action=%s whale_bias=%s dry_run=%s",
                    side_norm.upper(),
                    sym_u,
                    float(qty),
                    float(intent_price),
                    float(order_price),
                    float(notional),
                    interval,
                    int(target_leverage),
                    whale_action or "-",
                    str(extra0.get("whale_bias") or "neutral"),
                    self.dry_run,
                )
        except Exception:
            pass

        try:
            self._notify_position_open(
                symbol=sym_u,
                interval=str(interval or ""),
                side=str(side_norm),
                qty=float(qty),
                price=float(order_price),
                extra=extra0,
            )
        except Exception:
            pass

        try:
            rm = getattr(self, "risk_manager", None)
            if rm is not None:
                _side = str(side_norm).strip().lower()
                if _side not in ("long", "short"):
                    _side = "long" if signal_u == "BUY" else ("short" if signal_u == "SELL" else "hold")

                _meta: Dict[str, Any] = {}
                try:
                    if isinstance(extra0, dict):
                        _meta = dict(extra0)
                except Exception:
                    _meta = {}

                payload_meta = {"reason": "EXEC_OPEN", **_meta}

                out = rm.on_position_open(
                    symbol=sym_u,
                    side=_side,
                    qty=float(qty),
                    notional=float(notional),
                    price=float(order_price),
                    interval=str(interval or ""),
                    meta=payload_meta,
                )
                self._fire_and_forget(out, label="risk_on_open_exec")
        except Exception:
            try:
                if getattr(self, "logger", None):
                    self.logger.exception("[EXEC] risk_manager.on_position_open failed")
            except Exception:
                pass

        return

    # ---------------------------------------------------------
    # compatibility wrappers
    # ---------------------------------------------------------
    async def open_position(self, *args, **kwargs):
        pm = getattr(self, "position_manager", None)
        if pm is not None and hasattr(pm, "open_position"):
            res = pm.open_position(*args, **kwargs)
            try:
                if inspect.isawaitable(res):
                    return await res
            except Exception:
                pass
            return res
        return None

    async def execute_trade(self, *args, **kwargs):
        return await self.open_position(*args, **kwargs)
