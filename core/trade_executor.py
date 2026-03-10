from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.redis_price_cache import RedisPriceCache
from typing import Any, Dict, List, Optional, Tuple
try:
    from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError  # type: ignore
except Exception:  # pragma: no cover
    RequestsJSONDecodeError = Exception  # type: ignore

from config import config
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from tg_bot.telegram_bot import TelegramBot


class TradeExecutor:
    """
    TradeExecutor:
      - RiskManager ile entegre
      - PositionManager ile açık pozisyon state'ini yönetir
      - ATR bazlı SL/TP + trailing stop uygular
      - STALL (kâr ilerlemiyorsa TTL dolunca kapat) uygular
      - Whale "hold/exit bias" ile trailing/TTL davranışını ayarlar
      - DRY_RUN modunda gerçek emir atmadan state simüle eder
      - İsteğe bağlı PriceCache / RedisPriceCache ile tek kaynaktan fiyat okur

    Telegram politikası:
      ✅ Otomatik mesaj: SADECE pozisyon OPEN/CLOSE
      ❌ Signal/HOLD karar anında mesaj YOK
    """

    def __init__(
        self,
        client: Optional[Any],
        risk_manager: RiskManager,
        position_manager: Optional[PositionManager] = None,
        tg_bot: Optional[TelegramBot] = None,
        logger: Optional[logging.Logger] = None,
        dry_run: bool = True,
        base_order_notional: float = 50.0,
        max_position_notional: float = 500.0,
        max_leverage: float = 30.0,
        sl_pct: float = 0.01,
        tp_pct: float = 0.02,
        trailing_pct: float = 0.01,
        use_atr_sltp: bool = True,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        whale_risk_boost: float = 2.0,
        tg_bot_unused_kw: Optional[Any] = None,  # backward compat
        price_cache: Optional[Any] = None,
        redis_price_cache: Optional[RedisPriceCache] = None,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.tg_bot = tg_bot
        self.logger = logger or logging.getLogger("system")
        self.dry_run = bool(dry_run)

        self.base_order_notional = float(base_order_notional)
        self.max_position_notional = float(max_position_notional)
        self.max_leverage = float(max_leverage)

        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self.trailing_pct = float(trailing_pct)
        self.use_atr_sltp = bool(use_atr_sltp)
        self.atr_sl_mult = float(atr_sl_mult)
        self.atr_tp_mult = float(atr_tp_mult)

        self.whale_risk_boost = float(whale_risk_boost)

        # price cache
        self.price_cache: Optional[Any] = price_cache
        self.redis_price_cache: Optional[RedisPriceCache] = redis_price_cache

        # orchestration knobs (env)
        self.w_min = float(self._clip_float(os.getenv("W_MIN", "0.58"), 0.58) or 0.58)
        self.default_trail_pct = float(self._clip_float(os.getenv("TRAIL_PCT", "0.05"), 0.05) or 0.05)
        self.default_stall_ttl_sec = int(float(self._clip_float(os.getenv("STALL_TTL_SEC", "0"), 0.0) or 0.0))

        # order debug knobs
        self.order_verify_position = bool(self._truthy_env("ORDER_VERIFY_POSITION", "1"))
        self.order_poll_status = bool(self._truthy_env("ORDER_POLL_STATUS", "1"))
        self.order_poll_wait_s = float(self._clip_float(os.getenv("ORDER_POLL_WAIT_S", "2.0"), 2.0) or 2.0)

        # qty rounding knobs
        self.order_qty_round_decimals = int(
            float(self._clip_float(os.getenv("ORDER_QTY_ROUND_DECIMALS", "0"), 0.0) or 0.0)
        )

        # price cache knobs
        self.price_cache_max_age_sec = float(
            self._clip_float(os.getenv("PRICE_CACHE_MAX_AGE_SEC", "2.0"), 2.0) or 2.0
        )
        self.redis_price_cache_max_age_sec = float(
            self._clip_float(os.getenv("REDIS_PRICECACHE_MAX_AGE_SEC", "3.0"), 3.0) or 3.0
        )
        # whale decision policy
        self.whale_block_actions = {
            x.strip().lower()
            for x in str(os.getenv("WHALE_BLOCK_ACTIONS", "hard_block,force_exit")).split(",")
            if x.strip()
        }
        self.whale_reduce_actions = {
            x.strip().lower()
            for x in str(os.getenv("WHALE_REDUCE_ACTIONS", "reduce_size,tighten_risk,avoid_open")).split(",")
            if x.strip()
        }
        self.whale_boost_actions = {
            x.strip().lower()
            for x in str(os.getenv("WHALE_BOOST_ACTIONS", "confirm,strong_confirm,hold_winner")).split(",")
            if x.strip()
        }

        self.whale_hard_block_min_score = float(
            self._clip_float(os.getenv("WHALE_HARD_BLOCK_MIN_SCORE", "0.65"), 0.65) or 0.65
        )
        self.whale_reduce_min_score = float(
            self._clip_float(os.getenv("WHALE_REDUCE_MIN_SCORE", "0.40"), 0.40) or 0.40
        )
        self.whale_confirm_min_score = float(
            self._clip_float(os.getenv("WHALE_CONFIRM_MIN_SCORE", "0.55"), 0.55) or 0.55
        )

        self.whale_reduce_notional_mult = float(
            self._clip_float(os.getenv("WHALE_REDUCE_NOTIONAL_MULT", "0.65"), 0.65) or 0.65
        )
        self.whale_reduce_trailing_mult = float(
            self._clip_float(os.getenv("WHALE_REDUCE_TRAILING_MULT", "0.70"), 0.70) or 0.70
        )
        self.whale_hold_trailing_mult = float(
            self._clip_float(os.getenv("WHALE_HOLD_TRAILING_MULT", "1.20"), 1.20) or 1.20
        )
        self.whale_hold_stall_mult = float(
            self._clip_float(os.getenv("WHALE_HOLD_STALL_MULT", "1.40"), 1.40) or 1.40
        )
        self.whale_force_exit_enable = self._truthy_env("WHALE_FORCE_EXIT_ENABLE", "1")
        self.whale_force_exit_thr = float(
            self._clip_float(os.getenv("WHALE_FORCE_EXIT_THR", "0.72"), 0.72) or 0.72
        )
        self.whale_force_exit_min_pnl_pct = float(
            self._clip_float(os.getenv("WHALE_FORCE_EXIT_MIN_PNL_PCT", "-0.003"), -0.003) or -0.003
        )
        self.whale_force_exit_on_profit_only = self._truthy_env("WHALE_FORCE_EXIT_ON_PROFIT_ONLY", "0")

        self.whale_exit_trailing_mult = float(
            self._clip_float(os.getenv("WHALE_EXIT_TRAILING_MULT", "0.55"), 0.55) or 0.55
        )
        self.whale_exit_stall_mult = float(
            self._clip_float(os.getenv("WHALE_EXIT_STALL_MULT", "0.45"), 0.45) or 0.45
        )

        self.whale_reduce_sl_mult = float(
            self._clip_float(os.getenv("WHALE_REDUCE_SL_MULT", "0.75"), 0.75) or 0.75
        )
        self.whale_reduce_tp_mult = float(
            self._clip_float(os.getenv("WHALE_REDUCE_TP_MULT", "0.90"), 0.90) or 0.90
        )
        # /status snapshot
        self.last_snapshot: Dict[str, Any] = {}

        # PositionManager yoksa local fallback
        self._local_positions: Dict[str, Dict[str, Any]] = {}

        # backtest kapanış buffer
        self._closed_buffer: List[Dict[str, Any]] = []

        self._closed = False

        # requests timeout best-effort
        try:
            if self.client is not None and hasattr(self.client, "__dict__"):
                if not getattr(self.client, "requests_params", None):
                    self.client.requests_params = {"timeout": (3.05, 10)}
        except Exception:
            pass

        # redis price cache auto init
        if self.redis_price_cache is None:
            try:
                self.redis_price_cache = RedisPriceCache(
                    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    key_prefix=os.getenv("REDIS_PRICECACHE_PREFIX", "pricecache"),
                    ttl_sec=int(os.getenv("REDIS_PRICECACHE_TTL_SEC", "15")),
                )
            except Exception:
                self.redis_price_cache = None

    # -------------------------
    # price cache helpers
    # -------------------------
    def set_price_cache(
        self,
        price_cache: Optional[Any] = None,
        redis_price_cache: Optional[RedisPriceCache] = None,
    ) -> None:
        """
        main.py içinden:
            trade_executor.set_price_cache(price_cache, redis_price_cache)
        """
        if price_cache is not None:
            self.price_cache = price_cache
        if redis_price_cache is not None:
            self.redis_price_cache = redis_price_cache

    def _get_cached_mid_price(self, symbol: str, max_age_sec: Optional[float] = None) -> Optional[float]:
        sym = str(symbol).upper()
        age = self.price_cache_max_age_sec if max_age_sec is None else float(max_age_sec)

        try:
            if self.price_cache is not None and hasattr(self.price_cache, "get_mid"):
                px = self.price_cache.get_mid(sym, max_age_sec=age)
                if px is not None:
                    px = float(px)
                    if px > 0:
                        return px
        except Exception:
            pass

        try:
            if self.redis_price_cache is not None and hasattr(self.redis_price_cache, "get_mid"):
                px = self.redis_price_cache.get_mid(sym, max_age_sec=self.redis_price_cache_max_age_sec)
                if px is not None:
                    px = float(px)
                    if px > 0:
                        return px
        except Exception:
            pass

        return None

    def _resolve_price(
        self,
        symbol: str,
        price: Any = None,
        mark_price: Any = None,
        last_price: Any = None,
        *,
        max_age_sec: Optional[float] = None,
    ) -> Optional[float]:
        """
        Priority:
          1) explicit price / mark_price / last_price
          2) in-memory PriceCache mid
          3) RedisPriceCache mid
          4) client.get_price()
        """
        for candidate in (price, mark_price, last_price):
            pv = self._clip_float(candidate, None)
            if pv is not None and pv > 0:
                return float(pv)

        cached = self._get_cached_mid_price(symbol, max_age_sec=max_age_sec)
        if cached is not None and cached > 0:
            return float(cached)

        try:
            if self.client is not None:
                fn = getattr(self.client, "get_price", None)
                if callable(fn):
                    out = fn(str(symbol).upper())
                    pv = self._clip_float(out, None)
                    if pv is not None and pv > 0:
                        return float(pv)
        except Exception:
            pass

        return None
    # -------------------------
    # helpers (env / cast / normalize)
    # -------------------------
    @staticmethod
    def _truthy_env(name: str, default: str = "0") -> bool:
        return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _clip_float(x: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _signal_u_from_any(signal: Any) -> str:
        try:
            s = str(signal or "").strip().lower()
            if s in ("buy", "long", "1", "true"):
                return "BUY"
            if s in ("sell", "short", "-1", "false"):
                return "SELL"
            return "HOLD"
        except Exception:
            return "HOLD"

    @staticmethod
    def _ensure_csv_append(path: str, header: List[str], row: Dict[str, Any]) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            new_file = (not p.exists()) or p.stat().st_size == 0
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if new_file:
                    w.writeheader()
                w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in header})
        except Exception:
            pass

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _append_hold_csv(self, row: dict) -> None:
        path = os.getenv("HOLD_DECISIONS_CSV_PATH", "logs/hold_decisions.csv")
        header = [
            "timestamp", "symbol", "interval", "signal",
            "p", "p_source", "ensemble_p", "model_confidence_factor", "p_buy_ema", "p_buy_raw",
        ]
        self._ensure_csv_append(path, header, row)

    def _append_trade_csv(self, row: dict) -> None:
        path = os.getenv("TRADE_DECISIONS_CSV_PATH", "logs/trade_decisions.csv")
        header = [
            "timestamp", "symbol", "interval", "signal",
            "p", "p_source", "ensemble_p", "model_confidence_factor", "p_buy_ema", "p_buy_raw",
        ]
        self._ensure_csv_append(path, header, row)

    def _normalize_side(self, signal_u: str) -> str:
        s = str(signal_u or "").strip().lower()
        if s == "buy":
            return "long"
        if s == "sell":
            return "short"
        return "hold"

    def _round_qty(self, qty: float) -> float:
        try:
            q = float(qty)
            d = int(self.order_qty_round_decimals or 0)
            if d <= 0:
                return q
            return float(round(q, d))
        except Exception:
            return float(qty)

    # -------------------------
    # async/coro helpers
    # -------------------------
    def _fire_and_forget(self, coro: Any, *, label: str = "task") -> None:
        try:
            if coro is None:
                return
            if not asyncio.iscoroutine(coro):
                return
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)
            except RuntimeError:
                try:
                    self.logger.warning("[EXEC] %s coroutine returned but no running loop; dropped", label)
                except Exception:
                    pass
        except Exception:
            pass

    async def _await_if_coro(self, maybe: Any, *, label: str = "awaitable") -> Any:
        try:
            if asyncio.iscoroutine(maybe):
                return await maybe
        except Exception:
            try:
                self.logger.exception("[EXEC] await %s failed", label)
            except Exception:
                pass
        return maybe

    # -------------------------
    # order debug helpers
    # -------------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

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
    def _summarize_order(resp: Any) -> Dict[str, Any]:
        if not isinstance(resp, dict):
            return {"resp": str(resp)}

        keys = [
            "symbol", "side", "type",
            "orderId", "clientOrderId", "status",
            "price", "avgPrice",
            "origQty", "executedQty", "cumQty", "cumQuote",
            "reduceOnly", "positionSide",
            "timeInForce",
            "updateTime", "transactTime",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if k in resp:
                out[k] = resp.get(k)

        for k in ("code", "msg"):
            if k in resp:
                out[k] = resp.get(k)

        return out

    def _make_client_order_id(self, symbol: str, tag: str) -> str:
        s = str(symbol).upper()
        rid = uuid.uuid4().hex[:12]
        return f"b1_{tag}_{s}_{rid}"[:36]

    # -------------------------
    # exchange qty normalization helpers
    # -------------------------

    def _to_decimal(self, x: Any, default: str = "0") -> Decimal:
        try:
            return Decimal(str(x))
        except Exception:
            return Decimal(default)

    def _floor_to_step(self, value: float, step: float) -> float:
        try:
            v = self._to_decimal(value)
            s = self._to_decimal(step)

            if s <= 0:
                return float(v)

            floored = (v / s).to_integral_value(rounding=ROUND_DOWN) * s
            return float(floored)
        except Exception:
            return float(value)

    def _extract_symbol_filters(self, symbol_info: Dict[str, Any]) -> Dict[str, float]:

        out = {
            "step_size": 0.0,
            "min_qty": 0.0,
            "min_notional": 0.0,
        }

        try:
            filters = symbol_info.get("filters", []) or []

            for f in filters:

                ftype = str(f.get("filterType", "")).upper()

                if ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):

                    step = float(f.get("stepSize", 0.0) or 0.0)
                    min_qty = float(f.get("minQty", 0.0) or 0.0)

                    if step > 0:
                        out["step_size"] = max(out["step_size"], step)

                    if min_qty > 0:
                        out["min_qty"] = max(out["min_qty"], min_qty)

                elif ftype == "MIN_NOTIONAL":

                    out["min_notional"] = float(f.get("notional", 0.0) or 0.0)

        except Exception:
            pass

        return out

    def _normalize_order_qty(self, symbol: str, raw_qty: float, price: float) -> float:

        try:

            client = getattr(self, "client", None)

            if client is None:
                return raw_qty

            info = client.futures_exchange_info()

            symbol_info = None

            for s in info.get("symbols", []):
                if s.get("symbol") == symbol:
                    symbol_info = s
                    break

            if not symbol_info:
                return raw_qty

            filters = self._extract_symbol_filters(symbol_info)

            step = float(filters.get("step_size", 0.0))
            min_qty = float(filters.get("min_qty", 0.0))
            min_notional = float(filters.get("min_notional", 0.0))

            qty = float(raw_qty)

            if step > 0:
                qty = self._floor_to_step(qty, step)

            notional = qty * float(price)

            if min_qty > 0 and qty < min_qty:
                return 0.0

            if min_notional > 0 and notional < min_notional:
                return 0.0

            return float(qty)

        except Exception:
            return float(raw_qty)
    # -------------------------
    # order retry helpers
    # -------------------------
    @staticmethod
    def _is_empty_invalid_response_err(e: Exception) -> bool:
        msg = str(e) or ""
        msg_l = msg.lower()
        if "invalid response" in msg_l:
            return True
        if "jsondecodeerror" in msg_l:
            return True
        return False

    def _sleep_s(self, s: float) -> None:
        try:
            time.sleep(max(0.0, float(s)))
        except Exception:
            pass

    def _call_with_retry(
        self,
        fn,
        payload: Dict[str, Any],
        *,
        attempts: int = 3,
        base_sleep: float = 0.6,
    ):
        last_exc: Optional[Exception] = None
        attempts_i = max(1, int(attempts))

        for i in range(attempts_i):
            try:
                return fn(**payload)
            except (RequestsJSONDecodeError,) as e:
                last_exc = e
                if i < attempts_i - 1:
                    try:
                        self.logger.warning(
                            "[EXEC][ORDER][RETRY] JSONDecodeError -> retry %d/%d | payload=%s",
                            i + 1,
                            attempts_i,
                            self._safe_json(payload, limit=650),
                        )
                    except Exception:
                        pass
                    self._sleep_s(float(base_sleep) * (2 ** i))
                    continue
                raise
            except Exception as e:
                last_exc = e
                if self._is_empty_invalid_response_err(e) and i < attempts_i - 1:
                    try:
                        self.logger.warning(
                            "[EXEC][ORDER][RETRY] Invalid/empty response -> retry %d/%d | err=%s",
                            i + 1,
                            attempts_i,
                            str(e)[:200],
                        )
                    except Exception:
                        pass
                    self._sleep_s(float(base_sleep) * (2 ** i))
                    continue
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("retry failed with unknown state")
    def _verify_position_sync(self, symbol: str) -> Dict[str, Any]:
        if self.dry_run:
            return {"verify": "skip", "reason": "dry_run"}

        client = getattr(self, "client", None)
        if client is None:
            return {"verify": "skip", "reason": "no_client"}

        fn = getattr(client, "futures_position_information", None)
        if not callable(fn):
            return {"verify": "skip", "reason": "no_futures_position_information"}

        sym = str(symbol).upper()
        try:
            data = fn(symbol=sym)
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict) and str(r.get("symbol", "")).upper() == sym]
                row = rows[0] if rows else (data[0] if data else {})
            elif isinstance(data, dict):
                row = data
            else:
                row = {}

            amt = None
            entry = None
            unreal = None
            lev = None
            if isinstance(row, dict):
                amt = row.get("positionAmt")
                entry = row.get("entryPrice")
                unreal = row.get("unRealizedProfit") or row.get("unrealizedProfit")
                lev = row.get("leverage")

            return {
                "verify": "ok",
                "symbol": sym,
                "positionAmt": amt,
                "entryPrice": entry,
                "unrealized": unreal,
                "leverage": lev,
            }
        except Exception as e:
            return {"verify": "error", "symbol": sym, "err": str(e)[:300]}

    def _poll_order_status(
        self,
        symbol: str,
        order_id: Any = None,
        client_order_id: str = "",
        max_wait_s: float = 2.0,
    ) -> Dict[str, Any]:
        if self.dry_run:
            return {"poll": "skip", "reason": "dry_run"}

        client = getattr(self, "client", None)
        if client is None:
            return {"poll": "skip", "reason": "no_client"}

        fn = getattr(client, "futures_get_order", None) or getattr(client, "futures_query_order", None)
        if not callable(fn):
            return {"poll": "skip", "reason": "no_futures_get_order"}

        sym = str(symbol).upper()
        t_end = time.time() + float(max_wait_s or 0.0)
        last: Any = None

        while time.time() < t_end:
            try:
                kwargs: Dict[str, Any] = {"symbol": sym}
                if order_id is not None:
                    kwargs["orderId"] = order_id
                elif client_order_id:
                    kwargs["origClientOrderId"] = client_order_id
                else:
                    break

                last = fn(**kwargs)
                if isinstance(last, dict):
                    st = str(last.get("status", "")).upper()
                    if st in ("FILLED", "CANCELED", "REJECTED", "EXPIRED"):
                        break
                time.sleep(0.25)
            except Exception as e:
                if self._is_empty_invalid_response_err(e):
                    time.sleep(0.2)
                    continue
                return {"poll": "error", "err": str(e)[:280], "result": self._summarize_order(last)}

        return {"poll": "ok", "result": self._summarize_order(last)}

    # -------------------------
    # equity helpers
    # -------------------------
    async def _get_futures_equity_usdt(self) -> float:
        try:
            if bool(getattr(self, "dry_run", True)):
                return 0.0

            client = getattr(self, "client", None)
            if client is None:
                return 0.0

            import inspect

            fn = getattr(client, "futures_account", None)
            if fn is None:
                return 0.0

            if inspect.iscoroutinefunction(fn):
                acc = await fn()
            else:
                acc = await asyncio.to_thread(fn)

            return float(self._extract_equity_from_futures_account(acc))
        except asyncio.CancelledError:
            raise
        except Exception:
            return 0.0

    def _get_futures_equity_usdt_sync(self) -> float:
        try:
            if bool(getattr(self, "dry_run", True)):
                return 0.0

            client = getattr(self, "client", None)
            if client is None:
                return 0.0

            fn = getattr(client, "futures_account", None)
            if not callable(fn):
                return 0.0

            acc = fn()
            return float(self._extract_equity_from_futures_account(acc))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_equity_from_futures_account(acc: Any) -> float:
        try:
            if not isinstance(acc, dict):
                return 0.0

            for k in ("totalWalletBalance", "totalMarginBalance", "availableBalance"):
                v = acc.get(k)
                if v is None:
                    continue
                try:
                    eq = float(v)
                    return float(max(0.0, eq))
                except Exception:
                    continue
        except Exception:
            pass
        return 0.0

    # -------------------------
    # Telegram: only OPEN/CLOSE
    # -------------------------
    def _tg_send(self, text: str) -> None:
        try:
            if self.tg_bot is None:
                return
            self.tg_bot.send_message(text)
        except Exception:
            pass

    def _notify_position_open(
        self,
        symbol: str,
        interval: str,
        side: str,
        qty: float,
        price: float,
        extra: Dict[str, Any],
    ) -> None:
        try:
            if not self._truthy_env("TG_NOTIFY_OPEN_CLOSE", "1"):
                return
            if self.dry_run and self._truthy_env("TG_OPEN_CLOSE_ONLY_REAL", "0"):
                return

            p_used = extra.get("ensemble_p")
            if p_used is None:
                p_used = extra.get("p_buy_ema") or extra.get("p_buy_raw")

            p_txt = "?" if p_used is None else f"{float(p_used):.4f}"
            src = str(extra.get("signal_source") or extra.get("p_buy_source") or "")
            whale_dir = str(extra.get("whale_dir", "none") or "none")
            whale_score = float(extra.get("whale_score", 0.0) or 0.0)

            msg = (
                f"🟢 *OPEN* `{symbol}` `{interval}`\n"
                f"side=`{side}` qty=`{qty:.6f}` price=`{price}`\n"
                f"p=`{p_txt}` src=`{src}` whale=`{whale_dir}` score=`{whale_score:.2f}` dry_run=`{self.dry_run}`"
            )
            self._tg_send(msg)
        except Exception:
            pass

    def _notify_position_close(
        self,
        symbol: str,
        interval: str,
        side: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl_usdt: float,
        reason: str,
    ) -> None:
        try:
            if not self._truthy_env("TG_NOTIFY_OPEN_CLOSE", "1"):
                return
            if self.dry_run and self._truthy_env("TG_OPEN_CLOSE_ONLY_REAL", "0"):
                return

            msg = (
                f"🔴 *CLOSE* `{symbol}` `{interval}`\n"
                f"side=`{side}` qty=`{qty:.6f}` entry=`{entry_price}` exit=`{exit_price}`\n"
                f"pnl_usdt=`{pnl_usdt:.4f}` reason=`{reason}` dry_run=`{self.dry_run}`"
            )
            self._tg_send(msg)
        except Exception:
            pass

    # -------------------------
    # unified shutdown contract
    # -------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            pm = self.position_manager
            if pm is not None and hasattr(pm, "close"):
                pm.close()
        except Exception:
            pass

    async def shutdown(self, reason: str = "unknown") -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True

        try:
            self.logger.info("[EXEC] shutdown requested | reason=%s", reason)
        except Exception:
            pass

        try:
            pm = getattr(self, "position_manager", None)
            if pm is not None:
                if hasattr(pm, "shutdown"):
                    out = pm.shutdown(reason)  # type: ignore
                    if asyncio.iscoroutine(out):
                        await out
                elif hasattr(pm, "close"):
                    out = pm.close()  # type: ignore
                    if asyncio.iscoroutine(out):
                        await out
        except Exception:
            pass

    async def aclose(self) -> None:
        return await self.shutdown(reason="close")
    # -------------------------
    # backtest helper API
    # -------------------------
    def has_open_position(self, symbol: str) -> bool:
        pos = self._get_position(symbol)
        if not pos:
            return False
        side = str(pos.get("side") or "").lower()
        qty = float(pos.get("qty") or 0.0)
        return (side in ("long", "short")) and (qty > 0)

    def close_position_backtest(
        self,
        symbol: str,
        price: float,
        reason: str = "manual",
        interval: str = "",
        **_ignored: Any,
    ) -> Optional[Dict[str, Any]]:
        return self._close_position(
            symbol=str(symbol).upper(),
            price=float(price),
            reason=str(reason),
            interval=str(interval or ""),
        )

    def pop_closed_trades(self) -> List[Dict[str, Any]]:
        out = list(self._closed_buffer)
        self._closed_buffer.clear()
        return out

    # -------------------------
    # position access
    # -------------------------
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                return self.position_manager.get_position(sym)
            except Exception as e:
                try:
                    self.logger.warning("[EXEC] PositionManager.get_position hata: %s (local fallback)", e)
                except Exception:
                    pass
        return self._local_positions.get(sym)

    def _set_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                self.position_manager.set_position(sym, pos)
                return
            except Exception as e:
                try:
                    self.logger.warning("[EXEC] PositionManager.set_position hata: %s (local fallback)", e)
                except Exception:
                    pass
        self._local_positions[sym] = pos

    def _clear_position(self, symbol: str) -> None:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                self.position_manager.clear_position(sym)
            except Exception as e:
                try:
                    self.logger.warning("[EXEC] PositionManager.clear_position hata: %s (local fallback)", e)
                except Exception:
                    pass
        self._local_positions.pop(sym, None)
    def _extract_whale_context(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}

        try:
            raw0 = out.get("raw")
            if isinstance(raw0, str):
                try:
                    raw0 = json.loads(raw0)
                except Exception:
                    raw0 = {}
            if not isinstance(raw0, dict):
                raw0 = {}

            raw1 = raw0.get("raw")
            if isinstance(raw1, str):
                try:
                    raw1 = json.loads(raw1)
                except Exception:
                    raw1 = {}
            if not isinstance(raw1, dict):
                raw1 = {}

            raw2 = raw1.get("raw")
            if isinstance(raw2, str):
                try:
                    raw2 = json.loads(raw2)
                except Exception:
                    raw2 = {}
            if not isinstance(raw2, dict):
                raw2 = {}

            meta_out = out.get("meta") if isinstance(out.get("meta"), dict) else {}
            meta0 = raw0.get("meta") if isinstance(raw0.get("meta"), dict) else {}
            meta1 = raw1.get("meta") if isinstance(raw1.get("meta"), dict) else {}
            meta2 = raw2.get("meta") if isinstance(raw2.get("meta"), dict) else {}
            whale_meta = out.get("whale_meta") if isinstance(out.get("whale_meta"), dict) else {}

            whale_dir = (
                out.get("whale_dir")
                or raw0.get("whale_dir")
                or raw1.get("whale_dir")
                or raw2.get("whale_dir")
                or meta_out.get("whale_dir")
                or meta0.get("whale_dir")
                or meta1.get("whale_dir")
                or meta2.get("whale_dir")
                or whale_meta.get("whale_dir")
                or "none"
            )

            whale_score = (
                out.get("whale_score")
                if out.get("whale_score") is not None else
                raw0.get("whale_score")
                if raw0.get("whale_score") is not None else
                raw1.get("whale_score")
                if raw1.get("whale_score") is not None else
                raw2.get("whale_score")
                if raw2.get("whale_score") is not None else
                meta_out.get("whale_score")
                if meta_out.get("whale_score") is not None else
                meta0.get("whale_score")
                if meta0.get("whale_score") is not None else
                meta1.get("whale_score")
                if meta1.get("whale_score") is not None else
                meta2.get("whale_score")
                if meta2.get("whale_score") is not None else
                whale_meta.get("whale_score")
                if whale_meta.get("whale_score") is not None else
                0.0
            )

            whale_action = (
                out.get("whale_action")
                or out.get("whale_decision")
                or out.get("whale_policy")
                or raw0.get("whale_action")
                or raw1.get("whale_action")
                or raw2.get("whale_action")
                or meta_out.get("whale_action")
                or meta0.get("whale_action")
                or meta1.get("whale_action")
                or meta2.get("whale_action")
                or whale_meta.get("whale_action")
                or ""
            )

            out["whale_dir"] = str(whale_dir or "none").strip().lower()
            out["whale_score"] = float(whale_score or 0.0)
            if whale_action:
                out["whale_action"] = str(whale_action).strip().lower()

        except Exception:
            pass

        return out
    # -------------------------
    # whale bias helpers
    # -------------------------
    def _whale_action(self, extra: Dict[str, Any]) -> str:
        try:
            extra = self._extract_whale_context(extra)
            return str(
                extra.get("whale_action")
                or extra.get("whale_decision")
                or extra.get("whale_policy")
                or ""
            ).strip().lower()
        except Exception:
            return ""
    def _whale_dir_score(self, extra: Dict[str, Any]) -> Tuple[str, float]:
        try:
            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            wscore = float(extra.get("whale_score", 0.0) or 0.0)
            return wdir, wscore
        except Exception:
            return "none", 0.0

    def _whale_bias(self, side: str, extra: Dict[str, Any]) -> str:
        """
        Çıkış davranışı için sade bias:
          hold / exit / reduce / neutral
        Öncelik:
          1) whale_action
          2) whale_dir + whale_score fallback
        """
        try:
            action = self._whale_action(extra)
            wdir, ws = self._whale_dir_score(extra)

            if action:
                if action in self.whale_block_actions:
                    return "exit"
                if action in self.whale_reduce_actions:
                    return "reduce"
                if action in self.whale_boost_actions:
                    return "hold"

            if ws < float(self.w_min):
                return "neutral"

            if wdir in ("long", "short") and side in ("long", "short"):
                if wdir == side:
                    return "hold"
                return "exit"
        except Exception:
            pass
        return "neutral"

    def _effective_trailing_pct(self, base_trail: float, bias: str) -> float:
        bt = float(base_trail or 0.0)
        if bt <= 0:
            return 0.0

        if bias == "hold":
            return max(bt, bt * float(self.whale_hold_trailing_mult))

        if bias == "exit":
            return max(0.001, bt * float(self.whale_exit_trailing_mult))

        if bias == "reduce":
            return max(0.001, bt * float(self.whale_reduce_trailing_mult))

        return bt

    def _effective_stall_ttl(self, base_ttl: int, bias: str) -> int:
        t = int(base_ttl or 0)
        if t <= 0:
            return 0

        if bias == "hold":
            return int(max(60, round(t * float(self.whale_hold_stall_mult))))

        if bias == "exit":
            return int(max(30, round(t * float(self.whale_exit_stall_mult))))

        if bias == "reduce":
            return int(max(45, round(t * 0.70)))

        return t

    @staticmethod
    def _pnl_pct(side: str, entry: float, price: float) -> float:
        if entry <= 0:
            return 0.0
        if side == "long":
            return (price - entry) / entry
        if side == "short":
            return (entry - price) / entry
        return 0.0

    def _should_block_open_by_whale(self, side: str, extra: Dict[str, Any]) -> bool:
        extra = self._extract_whale_context(extra)

        try:
            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)
            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            side0 = str(side or "").strip().lower()

            if side0 not in ("long", "short"):
                return False

            if action in self.whale_block_actions and ws >= float(self.whale_hard_block_min_score):
                try:
                    self.logger.info(
                        "[EXEC][WHALE][BLOCK] action-block side=%s whale_dir=%s whale_score=%.3f whale_action=%s",
                        str(side0),
                        str(wdir),
                        float(ws),
                        str(action),
                    )
                except Exception:
                    pass
                return True

            if wdir in ("long", "short"):
                if wdir != side0 and ws >= float(self.whale_hard_block_min_score):
                    try:
                        self.logger.info(
                            "[EXEC][WHALE][BLOCK] contra-block side=%s whale_dir=%s whale_score=%.3f whale_action=%s",
                            str(side0),
                            str(wdir),
                            float(ws),
                            str(action),
                        )
                    except Exception:
                        pass
                    return True

            return False
        except Exception:
            return False
    def _apply_whale_open_adjustments(
        self,
        side: str,
        notional: float,
        extra: Dict[str, Any],
    ) -> float:
        extra = self._extract_whale_context(extra)

        try:
            base_notional = float(notional)
            if base_notional <= 0:
                return 0.0

            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)
            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            side0 = str(side or "").strip().lower()

            adjusted = float(base_notional)

            if action in self.whale_block_actions and ws >= float(self.whale_hard_block_min_score):
                adjusted = max(10.0, base_notional * 0.25)

            elif action in self.whale_reduce_actions and ws >= float(self.whale_reduce_min_score):
                adjusted = max(10.0, base_notional * float(self.whale_reduce_notional_mult))

            elif action in self.whale_boost_actions and ws >= float(self.whale_confirm_min_score):
                boost_mult = 1.0 + max(0.0, ws - float(self.whale_confirm_min_score)) * float(self.whale_risk_boost)
                adjusted = base_notional * boost_mult

            else:
                if side0 in ("long", "short") and wdir in ("long", "short"):
                    if wdir != side0 and ws >= float(self.whale_hard_block_min_score):
                        adjusted = max(10.0, base_notional * 0.35)
                    elif wdir != side0 and ws >= float(self.whale_reduce_min_score):
                        adjusted = max(10.0, base_notional * float(self.whale_reduce_notional_mult))
                    elif wdir == side0 and ws >= float(self.whale_confirm_min_score):
                        boost_mult = 1.0 + max(0.0, ws - float(self.whale_confirm_min_score)) * float(self.whale_risk_boost)
                        adjusted = base_notional * boost_mult

            adjusted = min(float(adjusted), float(self.max_position_notional))
            adjusted = max(10.0, float(adjusted))

            try:
                self.logger.info(
                    "[EXEC][WHALE][OPEN-ADJUST] symbol=%s side=%s base_notional=%.2f adjusted_notional=%.2f whale_dir=%s whale_score=%.3f whale_action=%s",
                    str(extra.get("symbol", "")),
                    str(side0),
                    float(base_notional),
                    float(adjusted),
                    str(wdir),
                    float(ws),
                    str(action),
                )
            except Exception:
                pass

            return float(adjusted)
        except Exception:
            return float(notional)
    def _should_force_close_by_whale(
        self,
        side: str,
        extra: Dict[str, Any],
        pnl_pct: float,
    ) -> bool:
        extra = self._extract_whale_context(extra)

        try:
            side0 = str(side or "").strip().lower()
            if side0 not in ("long", "short"):
                return False

            action = self._whale_action(extra)
            whale_dir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            whale_score = float(extra.get("whale_score", 0.0) or 0.0)

            thr = float(os.getenv("WHALE_FORCE_EXIT_THR", "0.72"))
            min_pnl_pct = float(os.getenv("WHALE_FORCE_EXIT_MIN_PNL_PCT", "-0.003"))
            profit_only = self._truthy_env("WHALE_FORCE_EXIT_ON_PROFIT_ONLY", "0")

            if whale_dir not in ("long", "short"):
                return False

            # 1) action tabanlı direkt zorla çıkış
            if action == "force_exit" and whale_score >= thr:
                if profit_only:
                    return float(pnl_pct) > 0.0
                return float(pnl_pct) >= float(min_pnl_pct)

            # 2) hard block + ters whale yönü -> zorla çıkış
            if action in self.whale_block_actions and whale_dir != side0 and whale_score >= thr:
                if profit_only:
                    return float(pnl_pct) > 0.0
                return float(pnl_pct) >= float(min_pnl_pct)

            # 3) fallback: whale ters ve çok güçlüyse zorla çıkış
            if whale_dir != side0 and whale_score >= thr:
                if profit_only:
                    return float(pnl_pct) > 0.0
                return float(pnl_pct) >= float(min_pnl_pct)

            return False
        except Exception:
            return False
    # -------------------------
    # exchange order helpers
    # -------------------------
    def _exchange_open_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()

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

        q = self._round_qty(float(qty))

        q = self._normalize_order_qty(
            symbol=sym,
            raw_qty=q,
            price=float(price),
        )

        if q <= 0:
            raise ValueError("qty invalid after normalization")
        tag = "close" if reduce_only else "open"
        client_oid = self._make_client_order_id(sym, tag)

        payload: Dict[str, Any] = {
            "symbol": sym,
            "side": order_side,
            "type": "MARKET",
            "quantity": q,
            "newClientOrderId": client_oid,
        }
        if reduce_only:
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
                            "no supported order function on client (futures_create_order/create_order/new_order)"
                        )
        except Exception as e:
            dt = self._now_ms() - t0
            try:
                self.logger.exception(
                    "[EXEC][ORDER] %s FAIL | fn=%s symbol=%s side=%s qty=%.10f reduceOnly=%s dt_ms=%d client_oid=%s payload=%s err=%s",
                    ("CLOSE" if reduce_only else "OPEN"),
                    used,
                    sym,
                    order_side,
                    float(q),
                    bool(reduce_only),
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
            self.logger.info(
                "[EXEC][ORDER] %s OK | fn=%s symbol=%s side=%s qty=%.10f reduceOnly=%s dt_ms=%d summary=%s",
                ("CLOSE" if reduce_only else "OPEN"),
                used,
                sym,
                order_side,
                float(q),
                bool(reduce_only),
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
                pol = self._poll_order_status(sym, order_id=oid, client_order_id=coid, max_wait_s=self.order_poll_wait_s)
                try:
                    self.logger.info("[EXEC][ORDER][POLL] %s", self._safe_json(pol, limit=900))
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.order_verify_position:
                v = self._verify_position_sync(sym)
                try:
                    self.logger.info("[EXEC][VERIFY] %s", self._safe_json(v, limit=900))
                except Exception:
                    pass
        except Exception:
            pass

        return resp

    def _exchange_close_market(self, symbol: str, side: str, qty: float) -> Optional[Dict[str, Any]]:
        s = str(side or "").strip().lower()
        if s == "long":
            close_side = "short"
        elif s == "short":
            close_side = "long"
        else:
            close_side = s
        return self._exchange_open_market(
            symbol=str(symbol).upper(),
            side=close_side,
            qty=float(qty),
            price=0.0,
            reduce_only=True,
        )
    # -------------------------
    # position dict
    # -------------------------
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
        opened_at = datetime.utcnow().isoformat()
        atr_value = float(extra.get("atr", 0.0) or 0.0)

        trail_pct = self._clip_float(extra.get("trail_pct"), None)
        if trail_pct is None:
            trail_pct = self._clip_float(extra.get("trailing_pct"), None)
        if trail_pct is None:
            trail_pct = float(self.default_trail_pct or self.trailing_pct)
        trail_pct = float(max(0.0, min(0.50, float(trail_pct))))

        stall_ttl = extra.get("stall_ttl_sec", None)
        try:
            stall_ttl = int(stall_ttl) if stall_ttl is not None else int(self.default_stall_ttl_sec or 0)
        except Exception:
            stall_ttl = int(self.default_stall_ttl_sec or 0)
        stall_ttl = int(max(0, stall_ttl))

        # whale bias -> open anındaki risk profili
        bias = self._whale_bias(signal, extra)

        sl_mult_adj = 1.0
        tp_mult_adj = 1.0

        if bias == "reduce":
            sl_mult_adj = float(getattr(self, "whale_reduce_sl_mult", 0.75) or 0.75)
            tp_mult_adj = float(getattr(self, "whale_reduce_tp_mult", 0.90) or 0.90)
        elif bias == "hold":
            tp_mult_adj = max(
                1.0,
                float(getattr(self, "whale_hold_stall_mult", 1.40) or 1.40) / 1.2,
            )

        if self.use_atr_sltp and atr_value > 0.0:
            if signal == "long":
                sl_price = price - (self.atr_sl_mult * sl_mult_adj * atr_value)
                tp_price = price + (self.atr_tp_mult * tp_mult_adj * atr_value)
            else:
                sl_price = price + (self.atr_sl_mult * sl_mult_adj * atr_value)
                tp_price = price - (self.atr_tp_mult * tp_mult_adj * atr_value)
        else:
            eff_sl_pct = float(self.sl_pct) * float(sl_mult_adj)
            eff_tp_pct = float(self.tp_pct) * float(tp_mult_adj)

            if signal == "long":
                sl_price = price * (1.0 - eff_sl_pct)
                tp_price = price * (1.0 + eff_tp_pct)
            else:
                sl_price = price * (1.0 + eff_sl_pct)
                tp_price = price * (1.0 - eff_tp_pct)

        now_ts = time.time()

        pos: Dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "side": signal,
            "qty": float(qty),
            "entry_price": float(price),
            "notional": float(notional),
            "interval": interval,
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
                "probs": probs,
                "extra": extra,
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
            return (exit_price - entry_price) * qty
        if side == "short":
            return (entry_price - exit_price) * qty
        return 0.0
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

    def _apply_whale_open_adjustments(
        self,
        side: str,
        notional: float,
        extra: Dict[str, Any],
    ) -> float:
        try:
            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)

            if action in self.whale_reduce_actions and ws >= float(self.whale_reduce_min_score):
                return max(10.0, float(notional) * float(self.whale_reduce_notional_mult))

            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            if side in ("long", "short") and wdir in ("long", "short"):
                if wdir != side and ws >= float(self.whale_reduce_min_score):
                    return max(10.0, float(notional) * float(self.whale_reduce_notional_mult))
        except Exception:
            pass
        return float(notional)

    def _should_force_close_by_whale(
        self,
        side: str,
        extra: Dict[str, Any],
        pnl_pct: float,
    ) -> bool:
        try:
            action = self._whale_action(extra)
            ws = float(extra.get("whale_score", 0.0) or 0.0)

            if action in ("force_exit", "hard_block") and ws >= float(self.whale_hard_block_min_score):
                return True

            wdir = str(extra.get("whale_dir", "none") or "none").strip().lower()
            if side in ("long", "short") and wdir in ("long", "short"):
                if wdir != side and ws >= float(self.whale_hard_block_min_score) and pnl_pct > -0.02:
                    return True
        except Exception:
            pass
        return False
    def _close_position(self, symbol: str, price: float, reason: str, interval: str) -> Optional[Dict[str, Any]]:
        pos = self._get_position(symbol)
        if not pos:
            return None

        side_raw = str(pos.get("side") or "hold").strip().lower()
        if side_raw in ("buy", "long"):
            side = "long"
        elif side_raw in ("sell", "short"):
            side = "short"
        else:
            side = side_raw

        qty = float(pos.get("qty") or 0.0)
        entry_price = float(pos.get("entry_price") or 0.0)
        notional = float(pos.get("notional") or (qty * entry_price))

        close_price = self._resolve_price(
            symbol=symbol,
            price=price,
            mark_price=None,
            last_price=entry_price,
            max_age_sec=self.price_cache_max_age_sec,
        )
        if close_price is None or close_price <= 0:
            close_price = float(price) if float(price or 0.0) > 0 else float(entry_price)

        realized_pnl = self._calc_pnl(
            side=side,
            entry_price=entry_price,
            exit_price=float(close_price),
            qty=qty,
        )
        # exchange close (real)
        if self.dry_run:
            try:
                self.logger.info("[EXEC] DRY_RUN=True close emri gönderilmeyecek.")
            except Exception:
                pass
        else:
            try:
                self._exchange_close_market(symbol=str(symbol).upper(), side=str(side), qty=float(qty))
            except Exception:
                try:
                    self.logger.exception(
                        "[EXEC][CLOSE] exchange close failed -> position state korunuyor | symbol=%s",
                        str(symbol).upper(),
                    )
                except Exception:
                    pass
                return None

        try:
            self._notify_position_close(
                symbol=str(symbol).upper(),
                interval=str(interval or pos.get("interval") or ""),
                side=str(side),
                qty=float(qty),
                entry_price=float(entry_price),
                exit_price=float(close_price),
                pnl_usdt=float(realized_pnl),
                reason=str(reason),
            )
        except Exception:
            pass

        try:
            rm = getattr(self, "risk_manager", None)
            if rm is not None:
                meta_dict = pos.get("meta") if isinstance(pos.get("meta"), dict) else {}
                probs_dict = meta_dict.get("probs") if isinstance(meta_dict.get("probs"), dict) else {}
                extra_dict = meta_dict.get("extra") if isinstance(meta_dict.get("extra"), dict) else {}

                payload_meta = {
                    "reason": str(reason),
                    "entry_price": float(entry_price),
                    "closed_side": str(side),
                    "interval": str(interval or ""),
                    "qty": float(qty),
                    "notional": float(notional),
                    "probs": probs_dict,
                    "extra": extra_dict,
                }

                out = rm.on_position_close(
                    symbol=str(symbol).upper(),
                    side=str(side),
                    qty=float(qty),
                    notional=float(notional),
                    price=float(close_price),
                    interval=str(interval or ""),
                    realized_pnl=float(realized_pnl),
                    meta=payload_meta,
                )
                self._fire_and_forget(out, label="risk_on_close")
        except Exception:
            try:
                if getattr(self, "logger", None):
                    self.logger.exception("[RISK] on_position_close failed")
            except Exception:
                pass

        self._clear_position(symbol)

        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["close_price"] = float(close_price)
        pos["realized_pnl"] = float(realized_pnl)
        pos["close_reason"] = str(reason)

        try:
            self._closed_buffer.append(dict(pos))
        except Exception:
            pass

        return pos

    def _check_sl_tp_trailing(self, symbol: str, price: float, interval: str) -> Optional[Dict[str, Any]]:
        pos = self._get_position(symbol)
        if not pos:
            return None

        live_price = self._resolve_price(
            symbol=symbol,
            price=price,
            max_age_sec=self.price_cache_max_age_sec,
        )
        if live_price is None or live_price <= 0:
            live_price = float(price)

        side = str(pos.get("side") or "hold").strip().lower()
        if side in ("buy", "long"):
            side = "long"
        elif side in ("sell", "short"):
            side = "short"

        sl_price = pos.get("sl_price")
        tp_price = pos.get("tp_price")
        trailing_pct_base = float(pos.get("trailing_pct") or 0.0)

        sl = float(sl_price) if sl_price is not None else None
        tp = float(tp_price) if tp_price is not None else None

        highest = float(pos.get("highest_price", live_price) or live_price)
        lowest = float(pos.get("lowest_price", live_price) or live_price)

        extra: Dict[str, Any] = {}
        try:
            meta = pos.get("meta") if isinstance(pos.get("meta"), dict) else {}
            extra0 = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
            extra = self._extract_whale_context(extra0)
        except Exception:
            extra = {}
        bias = self._whale_bias(side=side, extra=extra)
        trailing_pct = self._effective_trailing_pct(trailing_pct_base, bias)

        try:
            entry = float(pos.get("entry_price") or 0.0)
            cur_pnl_pct = float(self._pnl_pct(side, entry, float(live_price)))

            if self._should_force_close_by_whale(
                side=side,
                extra=extra,
                pnl_pct=cur_pnl_pct,
            ):
                return self._close_position(
                    symbol,
                    float(live_price),
                    reason="WHALE_FORCE_EXIT",
                    interval=interval,
                )

            best = float(pos.get("best_pnl_pct") or 0.0)
            now_ts = time.time()

            if cur_pnl_pct > best:
                pos["best_pnl_pct"] = float(cur_pnl_pct)
                pos["last_best_ts"] = float(now_ts)
                self._set_position(symbol, pos)
            else:
                stall_ttl = int(pos.get("stall_ttl_sec") or 0)
                stall_ttl_eff = self._effective_stall_ttl(stall_ttl, bias)
                last_best_ts = float(pos.get("last_best_ts") or now_ts)

                if stall_ttl_eff > 0 and cur_pnl_pct > 0.0:
                    if (now_ts - last_best_ts) >= float(stall_ttl_eff):
                        return self._close_position(
                            symbol,
                            float(live_price),
                            reason="STALL_EXIT",
                            interval=interval,
                        )
        except Exception:
            pass
        if side == "long":
            if sl is not None and live_price <= sl:
                return self._close_position(symbol, float(live_price), reason="SL_HIT", interval=interval)

            if tp is not None and live_price >= tp and bias != "hold":
                return self._close_position(symbol, float(live_price), reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if live_price > highest:
                    pos["highest_price"] = float(live_price)
                    self._set_position(symbol, pos)
                    highest = float(live_price)

                trail_sl = highest * (1.0 - trailing_pct)
                if live_price <= trail_sl:
                    return self._close_position(symbol, float(live_price), reason="TRAILING_STOP_LONG", interval=interval)

        elif side == "short":
            if sl is not None and live_price >= sl:
                return self._close_position(symbol, float(live_price), reason="SL_HIT", interval=interval)

            if tp is not None and live_price <= tp and bias != "hold":
                return self._close_position(symbol, float(live_price), reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if live_price < lowest:
                    pos["lowest_price"] = float(live_price)
                    self._set_position(symbol, pos)
                    lowest = float(live_price)

                trail_sl = lowest * (1.0 + trailing_pct)
                if live_price >= trail_sl:
                    return self._close_position(symbol, float(live_price), reason="TRAILING_STOP_SHORT", interval=interval)

        return None
    def _compute_notional(self, symbol: str, signal: str, price: float, extra: Dict[str, Any]) -> float:
        extra = self._extract_whale_context(extra)

        aggressive_mode = bool(getattr(config, "AGGRESSIVE_MODE", True))
        max_risk_mult = float(getattr(config, "MAX_RISK_MULTIPLIER", 4.0))

        base = float(self.base_order_notional)
        model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)
        model_conf = max(0.0, min(model_conf, 1.0))

        whale_dir = str(extra.get("whale_dir", "none") or "none").strip().lower()
        whale_score = float(extra.get("whale_score", 0.0) or 0.0)
        whale_action = self._whale_action(extra)

        aggr_factor = 1.0

        if aggressive_mode:
            if whale_action in self.whale_boost_actions and whale_score >= float(self.whale_confirm_min_score):
                aggr_factor *= 1.0 + max(0.0, whale_score - float(self.whale_confirm_min_score)) * float(self.whale_risk_boost)

            elif whale_action in self.whale_reduce_actions and whale_score >= float(self.whale_reduce_min_score):
                aggr_factor *= float(self.whale_reduce_notional_mult)

            elif whale_action in self.whale_block_actions and whale_score >= float(self.whale_hard_block_min_score):
                aggr_factor *= 0.25

            else:
                if whale_score > 0.0 and whale_dir in ("long", "short") and signal in ("long", "short"):
                    if whale_dir == signal and whale_score >= float(self.whale_confirm_min_score):
                        aggr_factor *= 1.0 + max(0.0, whale_score - float(self.whale_confirm_min_score)) * float(self.whale_risk_boost)
                    elif whale_dir != signal and whale_score >= float(self.whale_hard_block_min_score):
                        aggr_factor *= 0.35
                    elif whale_dir != signal and whale_score >= float(self.whale_reduce_min_score):
                        aggr_factor *= float(self.whale_reduce_notional_mult)

            aggr_factor *= (0.5 + 0.5 * model_conf)

        aggr_factor = max(0.0, min(float(aggr_factor), max_risk_mult))

        notional = base * aggr_factor
        notional = min(float(notional), float(self.max_position_notional))
        notional = max(float(notional), 10.0)

        try:
            dry_run = bool(getattr(self, "dry_run", True))
            equity_usdt = float(extra.get("equity_usdt", 0.0) or 0.0)
            alloc_pct = float(os.getenv("LIVE_EQUITY_ALLOC_PCT", "0.30"))

            if (not dry_run) and equity_usdt > 0.0 and alloc_pct > 0.0:
                cap = equity_usdt * alloc_pct
                if notional > cap:
                    notional = cap
        except Exception:
            pass

        try:
            self.logger.info(
                "[EXEC][NOTIONAL] symbol=%s side=%s base=%.2f aggr=%.3f mc=%.3f whale_dir=%s whale_score=%.3f whale_action=%s equity=%.2f notional=%.2f",
                str(symbol).upper(),
                str(signal),
                float(base),
                float(aggr_factor),
                float(model_conf),
                str(whale_dir),
                float(whale_score),
                str(whale_action),
                float(extra.get("equity_usdt", 0.0) or 0.0),
                float(notional),
            )
        except Exception:
            pass

        return float(notional)
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
        price = self._resolve_price(
            symbol=sym_u,
            price=meta0.get("price"),
            mark_price=meta0.get("mark_price"),
            last_price=meta0.get("last_price"),
        )

        if price is None or price <= 0:
            try:
                self.logger.warning(
                    "[EXEC][INTENT] missing price -> skip open | symbol=%s side=%s",
                    sym_u,
                    side0,
                )
            except Exception:
                pass
            return {"status": "skip", "reason": "missing_price"}

        whale_action = self._whale_action(meta0)
        whale_score = float(meta0.get("whale_score", 0.0) or 0.0)
        whale_dir = str(meta0.get("whale_dir", "none") or "none").strip().lower()

        if self._should_block_open_by_whale(side0, meta0):
            try:
                self.logger.info(
                    "[EXEC][WHALE][OPEN-BLOCK] symbol=%s side=%s whale_dir=%s whale_score=%.3f action=%s",
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

        npct = self._clip_float(meta0.get("recommended_notional_pct"), None)
        if npct is None:
            npct = self._clip_float(meta0.get("notional_pct"), None)
        npct = float(npct) if npct is not None else None

        try:
            eq_live = self._get_futures_equity_usdt_sync()
            if eq_live > 0:
                meta0["equity_usdt"] = float(eq_live)
        except Exception:
            pass

        eq_fallback = self._clip_float(os.getenv("DEFAULT_EQUITY_USDT", "1000"), 1000.0) or 1000.0

        notional: Optional[float] = None
        if npct is not None and npct > 0:
            base_eq = float(meta0.get("equity_usdt") or eq_fallback)
            notional = float(base_eq) * float(npct)

        if notional is None or notional <= 0:
            notional = self._compute_notional(sym_u, side0, float(price), meta0)

        raw_notional = float(notional)
        notional = self._apply_whale_open_adjustments(side0, float(notional), meta0)
        notional = float(min(float(notional), float(self.max_position_notional)))
        notional = float(max(10.0, float(notional)))

        qty = notional / float(price)
        qty = self._round_qty(qty)

        if qty <= 0:
            return {"status": "skip", "reason": "bad_qty"}

        cur = self._get_position(sym_u)
        cur_side = str(cur.get("side")).lower() if cur else None

        if cur_side in ("long", "short"):
            if cur_side == side0:
                return {"status": "ok", "reason": "already_open_same_side"}
            self._close_position(
                sym_u,
                float(price),
                reason="FLIP_INTENT",
                interval=str(interval or ""),
            )

        extra = dict(meta0)
        extra.setdefault("trail_pct", meta0.get("trail_pct", None))
        extra.setdefault("stall_ttl_sec", meta0.get("stall_ttl_sec", None))
        extra["whale_action"] = whale_action
        extra["whale_open_notional_before"] = float(raw_notional)
        extra["whale_open_notional_after"] = float(notional)
        extra["whale_notional_adjusted"] = bool(abs(float(notional) - float(raw_notional)) > 1e-12)

        whale_bias_now = self._whale_bias(side=side0, extra=extra)
        extra["whale_bias"] = whale_bias_now

        try:
            self.logger.info(
                "[EXEC][WHALE][OPEN-CHECK] symbol=%s side=%s action=%s bias=%s whale_dir=%s whale_score=%.3f raw_notional=%.2f final_notional=%.2f",
                sym_u,
                side0,
                whale_action or "-",
                whale_bias_now,
                whale_dir,
                whale_score,
                float(raw_notional),
                float(notional),
            )
        except Exception:
            pass

        if not self.dry_run:
            try:
                self._exchange_open_market(
                    symbol=sym_u,
                    side=side0,
                    qty=float(qty),
                    price=float(price),
                    reduce_only=False,
                )
            except Exception:
                return {"status": "skip", "reason": "exchange_open_failed"}

        probs: Dict[str, float] = {}

        pos, _opened_at = self._create_position_dict(
            signal=side0,
            symbol=sym_u,
            price=float(price),
            qty=float(qty),
            notional=float(notional),
            interval=str(interval or ""),
            probs=probs,
            extra=extra,
        )

        self._set_position(sym_u, pos)

        try:
            self.logger.info(
                "[EXEC][INTENT] OPEN %s | symbol=%s qty=%.10f price=%.6f notional=%.2f npct=%s whale_action=%s whale_bias=%s dry_run=%s",
                side0.upper(),
                sym_u,
                float(qty),
                float(price),
                float(notional),
                ("-" if npct is None else f"{npct:.4f}"),
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
                float(price),
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
                    notional=float(notional),
                    price=float(price),
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
            "price": float(price),
            "notional": float(notional),
            "trail_pct": float(pos.get("trailing_pct") or 0.0),
            "stall_ttl_sec": int(pos.get("stall_ttl_sec") or 0),
            "whale_action": whale_action,
            "whale_bias": whale_bias_now,
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
            if intent_id:
                self.logger.info(f"[CLOSE] intent_id={intent_id} symbol={sym} price={price} interval={interval}")
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

    def close_position_from_signal(
        self,
        symbol: str,
        interval: str = "",
        meta: Optional[Dict[str, Any]] = None,
        direction: str = "",
        price: Any = None,
        exit_price: Any = None,
    ) -> Dict[str, Any]:
        sym_u = str(symbol).upper()
        pos = self._get_position(sym_u)

        if not pos:
            return {"status": "skip", "reason": "no_position"}

        p = self._resolve_price(
            symbol=sym_u,
            price=price,
            mark_price=(meta or {}).get("mark_price") if isinstance(meta, dict) else None,
            last_price=exit_price,
        )

        if p is None or p <= 0:
            p = float(pos.get("entry_price") or 0.0) or 0.0

        out = self._close_position(
            sym_u,
            float(p),
            reason="INTENT_CLOSE",
            interval=str(interval or pos.get("interval") or ""),
        )

        if out:
            return {
                "status": "closed" if not self.dry_run else "dry_run",
                "symbol": sym_u,
                "price": float(p),
            }

        return {"status": "skip", "reason": "close_failed"}

    async def open_position(self, *args, **kwargs):
        pm = getattr(self, "position_manager", None)
        if pm is not None and hasattr(pm, "open_position"):
            res = pm.open_position(*args, **kwargs)
            try:
                import inspect
                if inspect.isawaitable(res):
                    return await res
            except Exception:
                pass
            return res
        return None

    async def execute_trade(self, *args, **kwargs):
        return await self.open_position(*args, **kwargs)

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

            self.last_snapshot = {
                "ts": datetime.utcnow().isoformat(),
                "symbol": sym_u,
                "interval": interval,
                "signal": signal_u,
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
                    "ensemble_p" if ens is not None else
                    ("p_buy_ema" if pbe is not None else ("p_buy_raw" if pbr is not None else "none"))
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
                self.logger.info(
                    "[EXEC] Signal=HOLD symbol=%s whale_action=%s whale_dir=%s whale_score=%.3f",
                    sym_u,
                    whale_action or "-",
                    whale_dir,
                    whale_score,
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

        try:
            if self._should_block_open_by_whale(side_norm, extra0):
                self.logger.info(
                    "[EXEC][VETO] WHALE_BLOCK | symbol=%s side=%s whale_dir=%s whale_score=%.3f action=%s -> SKIP",
                    sym_u,
                    side_norm,
                    whale_dir,
                    whale_score,
                    whale_action or "-",
                )
                return
        except Exception:
            pass

        live_price = self._resolve_price(
            symbol=sym_u,
            price=price,
            mark_price=extra0.get("mark_price"),
            last_price=extra0.get("last_price"),
        )
        if live_price is None or live_price <= 0:
            try:
                self.logger.warning(
                    "[EXEC] live_price unavailable -> skip | symbol=%s signal=%s",
                    sym_u,
                    signal_u,
                )
            except Exception:
                pass
            return

        try:
            self._check_sl_tp_trailing(symbol=sym_u, price=float(live_price), interval=interval)
        except Exception:
            pass

        raw_notional: Optional[float] = None

        if size is not None and float(size) > 0:
            qty = self._round_qty(float(size))
            raw_notional = float(qty) * float(live_price)
            notional = float(raw_notional)
        else:
            try:
                if "equity_usdt" not in extra0 or not float(extra0.get("equity_usdt") or 0.0) > 0.0:
                    extra0["equity_usdt"] = await self._get_futures_equity_usdt()
            except Exception:
                pass

            raw_notional = float(self._compute_notional(sym_u, side_norm, float(live_price), extra0))
            notional = float(self._apply_whale_open_adjustments(side_norm, float(raw_notional), extra0))
            qty = self._round_qty(float(notional) / float(live_price))

        if qty <= 0:
            try:
                self.logger.info(
                    "[EXEC] qty<=0 -> skip | symbol=%s side=%s raw_notional=%.4f",
                    sym_u,
                    side_norm,
                    float(raw_notional or 0.0),
                )
            except Exception:
                pass
            return

        extra0["whale_action"] = whale_action
        extra0["whale_bias"] = self._whale_bias(side=side_norm, extra=extra0)
        extra0["whale_open_notional_before"] = float(raw_notional or 0.0)
        extra0["whale_open_notional_after"] = float(notional)
        extra0["whale_notional_adjusted"] = bool(
            abs(float(notional) - float(raw_notional or 0.0)) > 1e-12
        )

        cur = self._get_position(sym_u)
        cur_side = str(cur.get("side")).lower() if cur else None

        if cur_side in ("long", "short") and cur_side != side_norm:
            self._close_position(sym_u, float(live_price), reason="FLIP", interval=interval)

        cur2 = self._get_position(sym_u)
        cur2_side = str(cur2.get("side")).lower() if cur2 else None

        if cur2_side == side_norm:
            try:
                self.logger.info(
                    "[EXEC] same-side already open -> skip | symbol=%s side=%s",
                    sym_u,
                    side_norm,
                )
            except Exception:
                pass
            return

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
            self.logger.info(
                "[EXEC][OPEN-CHECK] symbol=%s side=%s price=%.6f raw_notional=%.2f final_notional=%.2f qty=%.10f whale_action=%s whale_bias=%s whale_dir=%s whale_score=%.3f",
                sym_u,
                side_norm,
                float(live_price),
                float(raw_notional or 0.0),
                float(notional),
                float(qty),
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
                    price=float(live_price),
                    reduce_only=False,
                )
            except Exception:
                try:
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
            price=float(live_price),
            qty=float(qty),
            notional=float(notional),
            interval=interval,
            probs=probs,
            extra=extra0,
        )

        self._set_position(sym_u, pos)

        try:
            self.logger.info(
                "[EXEC] OPEN %s | symbol=%s qty=%.10f price=%.6f notional=%.2f interval=%s whale_action=%s whale_bias=%s dry_run=%s",
                side_norm.upper(),
                sym_u,
                float(qty),
                float(live_price),
                float(notional),
                interval,
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
                price=float(live_price),
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
                    price=float(live_price),
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
