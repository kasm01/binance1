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

try:
    # requests.exceptions.JSONDecodeError (requests>=2.27 civarı)
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

        # orchestration knobs (env)
        self.w_min = float(self._clip_float(os.getenv("W_MIN", "0.58"), 0.58) or 0.58)
        self.default_trail_pct = float(self._clip_float(os.getenv("TRAIL_PCT", "0.05"), 0.05) or 0.05)
        self.default_stall_ttl_sec = int(float(self._clip_float(os.getenv("STALL_TTL_SEC", "0"), 0.0) or 0.0))

        # Order debug knobs
        self.order_verify_position = bool(self._truthy_env("ORDER_VERIFY_POSITION", "1"))
        self.order_poll_status = bool(self._truthy_env("ORDER_POLL_STATUS", "1"))
        self.order_poll_wait_s = float(self._clip_float(os.getenv("ORDER_POLL_WAIT_S", "2.0"), 2.0) or 2.0)

        # qty rounding knobs (opsiyonel)
        self.order_qty_round_decimals = int(float(self._clip_float(os.getenv("ORDER_QTY_ROUND_DECIMALS", "0"), 0.0) or 0.0))

        # /status snapshot için
        self.last_snapshot: Dict[str, Any] = {}

        # PositionManager yoksa local fallback
        self._local_positions: Dict[str, Dict[str, Any]] = {}

        # backtest kapanış buffer
        self._closed_buffer: List[Dict[str, Any]] = []

        self._closed = False

        # Best-effort: python-binance requests timeout (bazı sürümlerde client bunu kullanır)
        try:
            if self.client is not None and hasattr(self.client, "__dict__"):
                if not getattr(self.client, "requests_params", None):
                    self.client.requests_params = {"timeout": (3.05, 10)}  # connect, read
        except Exception:
            pass

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
        """
        Normalize -> BUY / SELL / HOLD
        """
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
        """
        Safe CSV append (create header if missing/empty).
        """
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
        """
        BUY/SELL kararlarını logs/trade_decisions.csv içine ekler. (karar anı kaydı)
        """
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
        """
        Opsiyonel qty rounding.
        ORDER_QTY_ROUND_DECIMALS>0 ise round uygular.
        """
        try:
            q = float(qty)
            d = int(self.order_qty_round_decimals or 0)
            if d <= 0:
                return q
            return float(round(q, d))
        except Exception:
            return float(qty)
    # -------------------------
    # async/coro helpers (risk manager vb.)
    # -------------------------
    def _fire_and_forget(self, coro: Any, *, label: str = "task") -> None:
        """
        Sync context'te coroutine dönerse: loop varsa create_task, yoksa yut.
        """
        try:
            if coro is None:
                return
            if not asyncio.iscoroutine(coro):
                return
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(coro)  # fire-and-forget
            except RuntimeError:
                # running loop yok -> burada bloklamıyoruz
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
    # Order debug helpers (latency / summary / clientOrderId)
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
        """
        Binance futures_create_order tipik response alanlarını kısaca çıkarır.
        resp dict değilse stringe düşer.
        """
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
    # Order retry helpers
    # -------------------------
    @staticmethod
    def _is_empty_invalid_response_err(e: Exception) -> bool:
        """
        python-binance bazen response body boş geldiğinde:
          BinanceRequestException: Invalid Response:
          + altında RequestsJSONDecodeError fırlatıyor.
        Bu durumda retry etmek mantıklı.
        """
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
        """
        Order fonksiyonlarını retry ile çağırır.
        Boş response / decode hatalarında retry yapar.
        Diğer hataları direkt yükseltir.
        """
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
        """
        Emir sonrası 'gerçekten position açıldı mı?' kontrolü (best-effort).
        futures_position_information varsa kullanır.
        """
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
        """
        Order status'ü kısa süre poll eder. (FILLED/CANCELED/REJECTED/EXPIRED gelene kadar)
        futures_get_order veya futures_query_order varsa kullanır.
        """
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
                # bazı anlarda boş response geliyor; poll kritik değil -> kısa retry + devam
                if self._is_empty_invalid_response_err(e):
                    time.sleep(0.2)
                    continue
                return {"poll": "error", "err": str(e)[:280], "result": self._summarize_order(last)}

        return {"poll": "ok", "result": self._summarize_order(last)}

    # -------------------------
    # Equity helpers (LIVE cap için)
    # -------------------------
    async def _get_futures_equity_usdt(self) -> float:
        """
        Futures USDT equity (wallet/margin) best-effort.
        DRY_RUN=True iken 0 döner (cap devreye girmesin).
        """
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
        """
        Sync path (open_position_from_signal) için best-effort equity.
        DRY_RUN=True iken 0 döner.
        """
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
        """
        futures_account dict içinden equity seçimi.
        Öncelik: totalWalletBalance -> totalMarginBalance -> availableBalance
        """
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
    # Telegram: SADECE OPEN/CLOSE
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
        """
        ✅ sadece OPEN event
        """
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
        """
        ✅ sadece CLOSE event
        """
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
        """Sync close (idempotent)."""
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
        """Async shutdown (idempotent)."""
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

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual",
        interval: str = "",
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
    # position access via PositionManager
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
        """
        Not: RiskManager on_position_open/on_position_close çağrıları
        SADECE execute_decision/_close_position içinde yapılmalı.
        """
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
    # -------------------------
    # whale bias helpers
    # -------------------------
    def _whale_bias(self, side: str, extra: Dict[str, Any]) -> str:
        """
        Returns: "hold" | "exit" | "neutral"
        """
        try:
            wdir = str(extra.get("whale_dir", "none") or "none").lower()
            ws = float(extra.get("whale_score", 0.0) or 0.0)
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
        """
        hold -> daha gevşek (daha büyük pct)
        exit -> daha sıkı (daha küçük pct)
        """
        bt = float(base_trail or 0.0)
        if bt <= 0:
            return 0.0
        if bias == "hold":
            return max(bt, bt * 1.25)
        if bias == "exit":
            return max(0.001, bt * 0.65)
        return bt

    def _effective_stall_ttl(self, base_ttl: int, bias: str) -> int:
        """
        hold -> TTL uzat
        exit -> TTL kısalt
        """
        t = int(base_ttl or 0)
        if t <= 0:
            return 0
        if bias == "hold":
            return int(max(60, round(t * 1.50)))
        if bias == "exit":
            return int(max(60, round(t * 0.50)))
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

    # -------------------------
    # Exchange order helpers (REAL)
    # -------------------------
    def _exchange_open_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        REAL: Binance Futures market order (best-effort).
        side: "long" -> BUY, "short" -> SELL

        Güçlendirme:
          - clientOrderId ekle
          - latency ölç
          - response summary log
          - optional poll status
          - optional verify position
          - retry/backoff (empty response / decode error)
          - opsiyonel qty rounding (ORDER_QTY_ROUND_DECIMALS)
        """
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
        if q <= 0:
            raise ValueError("qty must be > 0")

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

        # poll
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

        # verify position
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
        """
        side: mevcut pozisyonun side'ı (long/short)
        close için ters yöne market order + reduceOnly
        """
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

        if self.use_atr_sltp and atr_value > 0.0:
            if signal == "long":
                sl_price = price - self.atr_sl_mult * atr_value
                tp_price = price + self.atr_tp_mult * atr_value
            else:
                sl_price = price + self.atr_sl_mult * atr_value
                tp_price = price - self.atr_tp_mult * atr_value
        else:
            if signal == "long":
                sl_price = price * (1.0 - self.sl_pct)
                tp_price = price * (1.0 + self.tp_pct)
            else:
                sl_price = price * (1.0 + self.sl_pct)
                tp_price = price * (1.0 - self.tp_pct)

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

        realized_pnl = self._calc_pnl(
            side=side,
            entry_price=entry_price,
            exit_price=float(price),
            qty=qty,
        )

        # --- exchange close (real) ---
        if self.dry_run:
            try:
                self.logger.info("[EXEC] DRY_RUN=True close emri gönderilmeyecek.")
            except Exception:
                pass
        else:
            try:
                self._exchange_close_market(symbol=str(symbol).upper(), side=str(side), qty=float(qty))
            except Exception:
                # REAL close başarısızsa state'i kapatma!
                try:
                    self.logger.exception(
                        "[EXEC][CLOSE] exchange close failed -> position state korunuyor | symbol=%s",
                        str(symbol).upper(),
                    )
                except Exception:
                    pass
                return None

        # ✅ Telegram: sadece CLOSE
        try:
            self._notify_position_close(
                symbol=str(symbol).upper(),
                interval=str(interval or pos.get("interval") or ""),
                side=str(side),
                qty=float(qty),
                entry_price=float(entry_price),
                exit_price=float(price),
                pnl_usdt=float(realized_pnl),
                reason=str(reason),
            )
        except Exception:
            pass

        # --- RISK ---
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
                    price=float(price),
                    interval=str(interval or ""),
                    realized_pnl=float(realized_pnl),
                    meta=payload_meta,
                )

                # async dönerse loop varsa task'a al
                self._fire_and_forget(out, label="risk_on_close")
        except Exception:
            try:
                if getattr(self, "logger", None):
                    self.logger.exception("[RISK] on_position_close failed")
            except Exception:
                pass

        self._clear_position(symbol)

        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["close_price"] = float(price)
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

        highest = float(pos.get("highest_price", price) or price)
        lowest = float(pos.get("lowest_price", price) or price)

        extra: Dict[str, Any] = {}
        try:
            meta = pos.get("meta") if isinstance(pos.get("meta"), dict) else {}
            extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
        except Exception:
            extra = {}

        bias = self._whale_bias(side=side, extra=extra)
        trailing_pct = self._effective_trailing_pct(trailing_pct_base, bias)

        # --- stall tracking ---
        try:
            entry = float(pos.get("entry_price") or 0.0)
            cur_pnl_pct = float(self._pnl_pct(side, entry, float(price)))
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
                        return self._close_position(symbol, float(price), reason="STALL_EXIT", interval=interval)
        except Exception:
            pass

        # --- SL/TP/TRAIL ---
        if side == "long":
            if sl is not None and price <= sl:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            if tp is not None and price >= tp:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if price > highest:
                    pos["highest_price"] = float(price)
                    self._set_position(symbol, pos)
                    highest = float(price)

                trail_sl = highest * (1.0 - trailing_pct)
                if price <= trail_sl:
                    return self._close_position(symbol, price, reason="TRAILING_STOP_LONG", interval=interval)

        elif side == "short":
            if sl is not None and price >= sl:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            if tp is not None and price <= tp:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if price < lowest:
                    pos["lowest_price"] = float(price)
                    self._set_position(symbol, pos)
                    lowest = float(price)

                trail_sl = lowest * (1.0 + trailing_pct)
                if price >= trail_sl:
                    return self._close_position(symbol, price, reason="TRAILING_STOP_SHORT", interval=interval)

        return None

    def _compute_notional(self, symbol: str, signal: str, price: float, extra: Dict[str, Any]) -> float:
        aggressive_mode = bool(getattr(config, "AGGRESSIVE_MODE", True))
        max_risk_mult = float(getattr(config, "MAX_RISK_MULTIPLIER", 4.0))
        whale_boost_thr = float(getattr(config, "WHALE_STRONG_THR", 0.6))
        whale_veto_thr = float(getattr(config, "WHALE_VETO_THR", 0.6))

        base = float(self.base_order_notional)
        model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)

        whale_dir = str(extra.get("whale_dir", "none") or "none").lower()
        whale_score = float(extra.get("whale_score", 0.0) or 0.0)

        aggr_factor = 1.0

        if aggressive_mode:
            if whale_score > 0.0 and whale_dir in ("long", "short") and signal in ("long", "short"):
                if whale_dir == signal and whale_score >= whale_boost_thr:
                    aggr_factor += self.whale_risk_boost * max(0.0, whale_score - whale_boost_thr)
                elif whale_dir != signal and whale_score >= whale_veto_thr:
                    aggr_factor -= 0.8 * whale_score
                elif whale_dir != signal:
                    aggr_factor -= 0.4 * whale_score

            mc = max(0.0, min(model_conf, 1.0))
            aggr_factor *= (0.5 + 0.5 * mc)

        aggr_factor = max(0.0, min(aggr_factor, max_risk_mult))

        notional = base * aggr_factor
        notional = min(notional, float(self.max_position_notional))
        notional = max(notional, 10.0)

        # -------------------------------------------------
        # EQUITY % CAP (LIVE only)
        # -------------------------------------------------
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
                "[EXEC][NOTIONAL] base=%.2f aggr=%.3f mc=%.3f whale_dir=%s whale_score=%.3f equity=%.2f notional=%.2f",
                base,
                aggr_factor,
                model_conf,
                whale_dir,
                whale_score,
                float(extra.get("equity_usdt", 0.0) or 0.0),
                notional,
            )
        except Exception:
            pass

        return float(notional)
    # -------------------------
    # Orchestration entrypoints (IntentBridge uyumlu)
    # -------------------------
    def open_position_from_signal(
        self,
        symbol: str,
        side: str,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta0 = meta if isinstance(meta, dict) else {}
        sym_u = str(symbol).upper()
        side0 = str(side or "long").strip().lower()
        if side0 not in ("long", "short"):
            side0 = "long"

        price = None
        for k in ("price", "mark_price", "last_price"):
            pv = self._clip_float(meta0.get(k), None)
            if pv is not None and pv > 0:
                price = float(pv)
                break

        if price is None and self.client is not None:
            try:
                fn = getattr(self.client, "get_price", None)
                if callable(fn):
                    out = fn(sym_u)
                    pv = self._clip_float(out, None)
                    if pv and pv > 0:
                        price = float(pv)
            except Exception:
                price = None

        if price is None or price <= 0:
            try:
                self.logger.warning("[EXEC][INTENT] missing price -> skip open | symbol=%s side=%s", sym_u, side0)
            except Exception:
                pass
            return {"status": "skip", "reason": "missing_price"}

        npct = self._clip_float(meta0.get("recommended_notional_pct"), None)
        if npct is None:
            npct = self._clip_float(meta0.get("notional_pct"), None)
        npct = float(npct) if npct is not None else None

        # equity (best effort) -> meta içine koy ki _compute_notional cap'i çalışsın
        try:
            eq_live = self._get_futures_equity_usdt_sync()
            if eq_live > 0:
                meta0["equity_usdt"] = float(eq_live)
        except Exception:
            pass

        eq_fallback = self._clip_float(os.getenv("DEFAULT_EQUITY_USDT", "1000"), 1000.0) or 1000.0

        notional = None
        if npct is not None and npct > 0:
            base_eq = float(meta0.get("equity_usdt") or eq_fallback)
            notional = float(base_eq) * float(npct)

        if notional is None or notional <= 0:
            notional = self._compute_notional(sym_u, side0, float(price), meta0)

        notional = float(min(notional, float(self.max_position_notional)))
        notional = float(max(10.0, notional))
        qty = notional / float(price) if float(price) > 0 else 0.0
        qty = self._round_qty(qty)
        if qty <= 0:
            return {"status": "skip", "reason": "bad_qty"}

        try:
            rm = getattr(self, "risk_manager", None)
            if rm is not None and hasattr(rm, "can_open_new_trade"):
                ok = rm.can_open_new_trade(
                    symbol=sym_u,
                    side=side0,
                    price=float(price),
                    notional=float(notional),
                    interval=str(interval or ""),
                )
                if not ok:
                    return {"status": "skip", "reason": "risk_gate"}
        except Exception:
            pass

        # zaten açık mı?
        cur = self._get_position(sym_u)
        cur_side = str(cur.get("side")).lower() if cur else None
        if cur_side in ("long", "short"):
            if cur_side == side0:
                return {"status": "ok", "reason": "already_open_same_side"}
            # flip -> close then open
            self._close_position(sym_u, float(price), reason="FLIP_INTENT", interval=str(interval or ""))

        # REAL emir (DRY_RUN=False)
        if not self.dry_run:
            try:
                self._exchange_open_market(symbol=sym_u, side=side0, qty=float(qty), reduce_only=False)
            except Exception:
                return {"status": "skip", "reason": "exchange_open_failed"}

        probs: Dict[str, float] = {}
        extra = dict(meta0)
        extra.setdefault("trail_pct", meta0.get("trail_pct", None))
        extra.setdefault("stall_ttl_sec", meta0.get("stall_ttl_sec", None))

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
                "[EXEC][INTENT] OPEN %s | symbol=%s qty=%.10f price=%.6f notional=%.2f npct=%s dry_run=%s",
                side0.upper(),
                sym_u,
                float(qty),
                float(price),
                float(notional),
                ("-" if npct is None else f"{npct:.4f}"),
                self.dry_run,
            )
        except Exception:
            pass

        try:
            self._notify_position_open(sym_u, str(interval or ""), side0, float(qty), float(price), extra)
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
        }

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

        p = self._clip_float(price, None)
        if p is None:
            p = self._clip_float(exit_price, None)

        if p is None or p <= 0:
            p = float(pos.get("entry_price") or 0.0) or 0.0

        out = self._close_position(sym_u, float(p), reason="INTENT_CLOSE", interval=str(interval or pos.get("interval") or ""))
        if out:
            return {"status": "closed" if not self.dry_run else "dry_run", "symbol": sym_u, "price": float(p)}
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
        extra0 = extra if isinstance(extra, dict) else {}
        signal_u = self._signal_u_from_any(signal)
        side_norm = self._normalize_side(signal_u)
        sym_u = str(symbol).upper()

        # snapshot for /status
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
                "whale_dir": extra0.get("whale_dir", "none"),
                "whale_score": extra0.get("whale_score", 0.0),
                "extra": extra0,
            }
        except Exception:
            pass

        # HOLD
        if signal_u == "HOLD":
            try:
                ens = extra0.get("ensemble_p")
                mcf = extra0.get("model_confidence_factor")
                pbe = extra0.get("p_buy_ema")
                pbr = extra0.get("p_buy_raw")

                p_val = ens if ens is not None else (pbe if pbe is not None else pbr)
                p_src = "ensemble_p" if ens is not None else ("p_buy_ema" if pbe is not None else ("p_buy_raw" if pbr is not None else "none"))

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
                self.logger.info("[EXEC] Signal=HOLD symbol=%s", sym_u)
            except Exception:
                pass
            return

        if self._truthy_env("SHADOW_MODE", "0"):
            return
        if training_mode:
            return

        # whale veto (legacy)
        try:
            whale_dir = str(extra0.get("whale_dir", "none") or "none").lower()
            whale_score = float(extra0.get("whale_score", 0.0) or 0.0)
            veto_thr = float(os.getenv("WHALE_VETO_THR", "0.70"))
            if side_norm in ("long", "short") and whale_dir in ("long", "short"):
                if whale_dir != side_norm and whale_score >= veto_thr:
                    self.logger.info(
                        "[EXEC][VETO] WHALE_VETO | side=%s whale_dir=%s whale_score=%.3f thr=%.2f -> SKIP",
                        side_norm,
                        whale_dir,
                        whale_score,
                        veto_thr,
                    )
                    return
        except Exception:
            pass

        # SL/TP/TRAIL/STALL check (mevcut pozisyonu kapatabilir)
        try:
            self._check_sl_tp_trailing(symbol=sym_u, price=float(price), interval=interval)
        except Exception:
            pass

        # qty/notional
        if size is not None and float(size) > 0:
            qty = self._round_qty(float(size))
            notional = qty * float(price)
        else:
            # ✅ LIVE equity cap için equity_usdt doldur
            try:
                if "equity_usdt" not in extra0 or not float(extra0.get("equity_usdt") or 0.0) > 0.0:
                    extra0["equity_usdt"] = await self._get_futures_equity_usdt()
            except Exception:
                pass

            notional = self._compute_notional(sym_u, side_norm, float(price), extra0)
            qty = self._round_qty(notional / float(price) if float(price) > 0 else 0.0)

        if qty <= 0 or float(price) <= 0:
            return

        # open/flip logic
        cur = self._get_position(sym_u)
        cur_side = str(cur.get("side")).lower() if cur else None

        if cur_side in ("long", "short") and cur_side != side_norm:
            self._close_position(sym_u, float(price), reason="FLIP", interval=interval)

        cur2 = self._get_position(sym_u)
        cur2_side = str(cur2.get("side")).lower() if cur2 else None
        if cur2_side == side_norm:
            return

        # BUY/SELL karar anı CSV
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

        # REAL open order (DRY_RUN=False)
        if not self.dry_run:
            try:
                self._exchange_open_market(symbol=sym_u, side=side_norm, qty=float(qty), reduce_only=False)
            except Exception:
                try:
                    self.logger.exception("[EXEC][OPEN] exchange open failed -> state set edilmeyecek | symbol=%s", sym_u)
                except Exception:
                    pass
                return

        pos, _opened_at = self._create_position_dict(
            signal=side_norm,
            symbol=sym_u,
            price=float(price),
            qty=float(qty),
            notional=float(notional),
            interval=interval,
            probs=probs,
            extra=extra0,
        )
        self._set_position(sym_u, pos)

        try:
            self.logger.info(
                "[EXEC] OPEN %s | symbol=%s qty=%.10f price=%.6f notional=%.2f interval=%s dry_run=%s",
                side_norm.upper(),
                sym_u,
                float(qty),
                float(price),
                float(notional),
                interval,
                self.dry_run,
            )
        except Exception:
            pass

        # ✅ Telegram: sadece OPEN
        try:
            self._notify_position_open(
                symbol=sym_u,
                interval=str(interval or ""),
                side=str(side_norm),
                qty=float(qty),
                price=float(price),
                extra=extra0,
            )
        except Exception:
            pass

        # --- RISK on_position_open ---
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
                    price=float(price),
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
