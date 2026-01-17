# core/trade_executor.py
from __future__ import annotations

import asyncio
import logging
import os
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from config import config
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from tg_bot.telegram_bot import TelegramBot


class TradeExecutor:
    """
    TradeExecutor:
      - RiskManager ile entegre
      - PositionManager ile açık pozisyon state'ini yönetir
      - ATR bazlı SL/TP + trailing stop uygular
      - Flip (long -> short, short -> long) durumunda PnL hesaplar
      - DRY_RUN modunda gerçek emir atmadan simüle eder

    Tek tip shutdown kontratı:
      - close() (sync, idempotent)
      - shutdown() (async, idempotent)  <-- main.ShutdownManager bunu çağırabilir

    BACKTEST yardımcı API (eklendi):
      - has_open_position(symbol) -> bool
      - close_position(symbol, price, reason="manual", interval="") -> Optional[dict]
      - pop_closed_trades() -> List[dict]   (backtest trade kayıtları için)
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
        max_leverage: float = 3.0,
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

        self.last_snapshot: Dict[str, Any] = {}
        self._tg_state: Dict[str, Any] = {
            "last_hold_ts": 0.0,
            "last_sig": None,
            "last_sig_ts": 0.0,
        }

        self._local_positions: Dict[str, Dict[str, Any]] = {}

        # backtest trade buffer (NEW)
        self._closed_buffer: List[Dict[str, Any]] = []

        self._closed = False

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
            if getattr(self, "logger", None):
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

        return

    async def aclose(self) -> None:
        return await self.shutdown(reason="close")

    # -------------------------
    # backtest helper API (NEW)
    # -------------------------
    def has_open_position(self, symbol: str) -> bool:
        pos = self._get_position(symbol)
        if not pos:
            return False
        side = str(pos.get("side") or "").lower()
        qty = float(pos.get("qty") or 0.0)
        return (side in ("long", "short")) and (qty > 0)

    def close_position(self, symbol: str, price: float, reason: str = "manual", interval: str = "") -> Optional[Dict[str, Any]]:
        """Backtest / helper close wrapper."""
        return self._close_position(symbol=str(symbol).upper(), price=float(price), reason=str(reason), interval=str(interval or ""))

    def pop_closed_trades(self) -> List[Dict[str, Any]]:
        """Backtest: biriken kapanışları alıp buffer'ı temizler."""
        out = list(self._closed_buffer)
        self._closed_buffer.clear()
        return out

    # -------------------------
    # helpers
    # -------------------------
    def _append_hold_csv(self, row: dict) -> None:
        try:
            path = os.getenv("HOLD_DECISIONS_CSV_PATH", "logs/hold_decisions.csv")
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            header = [
                "timestamp", "symbol", "interval", "signal",
                "p", "p_source", "ensemble_p", "model_confidence_factor", "p_buy_ema", "p_buy_raw",
            ]
            new_file = (not Path(path).exists()) or Path(path).stat().st_size == 0
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if new_file:
                    w.writeheader()
                w.writerow({k: row.get(k, "") for k in header})
        except Exception:
            pass

    def _normalize_side(self, signal: str) -> str:
        s = str(signal or "").strip().lower()
        if s in ("buy", "long", "1", "true"):
            return "long"
        if s in ("sell", "short", "-1", "false"):
            return "short"
        return "hold"

    def _tg_send(self, text: str) -> None:
        try:
            if self.tg_bot is None:
                return
            self.tg_bot.send_message(text)
        except Exception:
            pass

    def _notify_telegram(
        self,
        signal_u: str,
        symbol: str,
        interval: str,
        price: float,
        probs: Dict[str, float],
        extra: Dict[str, Any],
    ) -> None:
        notify_trades = str(os.getenv("TG_NOTIFY_TRADES", "1")).lower() in ("1", "true", "yes", "on")
        notify_hold = str(os.getenv("TG_NOTIFY_HOLD", "0")).lower() in ("1", "true", "yes", "on")
        hold_every = int(os.getenv("HOLD_NOTIFY_EVERY_SEC", "300"))
        dup_cd = int(os.getenv("TG_DUPLICATE_SIGNAL_COOLDOWN_SEC", "20"))

        now = time.time()

        if signal_u == "HOLD":
            if not notify_hold:
                return
            last = float(self._tg_state.get("last_hold_ts") or 0.0)
            if (now - last) < float(hold_every):
                return
            self._tg_state["last_hold_ts"] = now

        if signal_u in ("BUY", "SELL"):
            if not notify_trades:
                return
            last_sig = self._tg_state.get("last_sig")
            last_ts = float(self._tg_state.get("last_sig_ts") or 0.0)
            if last_sig == signal_u and (now - last_ts) < float(dup_cd):
                return
            self._tg_state["last_sig"] = signal_u
            self._tg_state["last_sig_ts"] = now

        p_used = None
        try:
            p_used = extra.get("ensemble_p")
            if p_used is None:
                p_used = extra.get("p_buy_ema")
            if p_used is None:
                p_used = extra.get("p_buy_raw")
            if p_used is None:
                p_used = probs.get("p_used") or probs.get("p_single")
        except Exception:
            p_used = None

        try:
            p_txt = f"{float(p_used):.4f}" if p_used is not None else "?"
        except Exception:
            p_txt = "?"

        whale_dir = str(extra.get("whale_dir", "none") or "none")
        whale_score = float(extra.get("whale_score", 0.0) or 0.0)
        src = str(extra.get("signal_source", extra.get("p_buy_source", "")) or "")

        msg = (
            f"📣 *{signal_u}*  {symbol} `{interval}`\n"
            f"price=`{price}`  p=`{p_txt}` src=`{src}`\n"
            f"🐋 whale=`{whale_dir}` score=`{whale_score:.2f}`  dry_run=`{self.dry_run}`"
        )
        self._tg_send(msg)

    # -------------------------
    # position access via PositionManager
    # -------------------------
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                return self.position_manager.get_position(sym)
            except Exception as e:
                self.logger.warning("[EXEC] PositionManager.get_position hata: %s (local fallback)", e)
        return self._local_positions.get(sym)

    def _set_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                self.position_manager.set_position(sym, pos)

                try:
                    rm = getattr(self, "risk_manager", None)
                    if rm:
                        # symbol/side/qty/notional/price/interval mümkün olduğunca local scope'tan alınır
                        _symbol = str(locals().get("symbol") or locals().get("sym") or "").upper()
                        _side = str(locals().get("side") or locals().get("position_side") or "")
                        _qty = float(locals().get("qty") or locals().get("quantity") or 0.0)
                        _notional = float(locals().get("notional") or locals().get("usdt_notional") or 0.0)
                        _price = locals().get("entry_price", None)
                        if _price is None:
                            _price = locals().get("price", 0.0)
                        _price = float(_price or 0.0)
                        _interval = str(locals().get("interval") or "")
                        _meta = locals().get("meta") if isinstance(locals().get("meta"), dict) else {}
                        rm.on_position_open(
                            symbol=_symbol,
                            side=_side,
                            qty=_qty,
                            notional=_notional,
                            price=_price,
                            interval=_interval,
                            meta={"reason": "EXEC_OPEN", **_meta},
                        )
                except Exception:
                    try:
                        if getattr(self, "logger", None):
                            self.logger.exception("[EXEC] risk_manager.on_position_open failed")
                    except Exception:
                        pass

                return
            except Exception as e:
                self.logger.warning("[EXEC] PositionManager.set_position hata: %s (local fallback)", e)
        self._local_positions[sym] = pos

    def _clear_position(self, symbol: str) -> None:
        sym = str(symbol).upper()
        if self.position_manager is not None:
            try:
                self.position_manager.clear_position(sym)
            except Exception as e:
                self.logger.warning("[EXEC] PositionManager.clear_position hata: %s (local fallback)", e)
        self._local_positions.pop(sym, None)

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
            "trailing_pct": float(self.trailing_pct),
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

        side = str(pos.get("side") or "hold")
        qty = float(pos.get("qty") or 0.0)
        entry_price = float(pos.get("entry_price") or 0.0)
        notional = float(pos.get("notional") or (qty * entry_price))

        realized_pnl = self._calc_pnl(side=side, entry_price=entry_price, exit_price=price, qty=qty)

        if self.dry_run:
            self.logger.info("[EXEC] DRY_RUN=True close emri gönderilmeyecek.")
        else:
            # TODO: gerçek close emri
            pass

        try:
            meta_dict = pos.get("meta") if isinstance(pos.get("meta"), dict) else {}
            probs_dict = meta_dict.get("probs") if isinstance(meta_dict.get("probs"), dict) else {}
            extra_dict = meta_dict.get("extra") if isinstance(meta_dict.get("extra"), dict) else {}

            self.risk_manager.on_position_close(
                symbol=str(symbol).upper(),
                side=side,
                qty=qty,
                notional=notional,
                price=float(price),
                interval=interval,
                realized_pnl=float(realized_pnl),
                meta={
                    "reason": reason,
                    "entry_price": entry_price,
                    "closed_side": side,
                    "interval": interval,
                    "qty": qty,
                    "notional": notional,
                    "probs": probs_dict,
                    "extra": extra_dict,
                },
            )
        except Exception as e:
            self.logger.warning("[RISK] on_position_close hata: %s", e)

        self._clear_position(symbol)

        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["close_price"] = float(price)
        pos["realized_pnl"] = float(realized_pnl)
        pos["close_reason"] = reason

        # backtest buffer push (NEW)
        try:
            self._closed_buffer.append(dict(pos))
        except Exception:
            pass

        return pos

    def _check_sl_tp_trailing(self, symbol: str, price: float, interval: str) -> Optional[Dict[str, Any]]:
        pos = self._get_position(symbol)
        if not pos:
            return None

        side = str(pos.get("side") or "hold")
        sl_price = pos.get("sl_price")
        tp_price = pos.get("tp_price")
        trailing_pct = float(pos.get("trailing_pct") or 0.0)

        sl = float(sl_price) if sl_price is not None else None
        tp = float(tp_price) if tp_price is not None else None

        highest = float(pos.get("highest_price", price) or price)
        lowest = float(pos.get("lowest_price", price) or price)

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

        try:
            self.logger.info(
                "[EXEC][NOTIONAL] base=%.2f aggr=%.3f mc=%.3f whale_dir=%s whale_score=%.3f notional=%.2f",
                base, aggr_factor, model_conf, whale_dir, whale_score, notional
            )
        except Exception:
            pass

        return float(notional)

    # -------------------------
    # Backward-compat open_position / execute_trade
    # -------------------------
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

    # -------------------------
    # MAIN decision entrypoint
    # -------------------------
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

        # signal normalize -> BUY/SELL/HOLD
        try:
            _s = str(signal).strip().lower()
            if _s in ("buy", "long", "1", "true"):
                signal_u = "BUY"
            elif _s in ("sell", "short", "-1", "false"):
                signal_u = "SELL"
            else:
                signal_u = "HOLD"
        except Exception:
            signal_u = "HOLD"

        side_norm = self._normalize_side(signal_u)

        # snapshot for telegram /status
        try:
            self.last_snapshot = {
                "ts": datetime.utcnow().isoformat(),
                "symbol": str(symbol).upper(),
                "interval": interval,
                "signal": signal_u,
                "signal_source": str(extra0.get("signal_source", extra0.get("p_buy_source", ""))),
                "p_used": extra0.get("ensemble_p", probs.get("p_used") if isinstance(probs, dict) else None),
                "p_single": probs.get("p_single") if isinstance(probs, dict) else None,
                "p_buy_raw": extra0.get("p_buy_raw"),
                "p_buy_ema": extra0.get("p_buy_ema"),
                "whale_dir": extra0.get("whale_dir", "none"),
                "whale_score": extra0.get("whale_score", 0.0),
                "extra": extra0,
            }
        except Exception:
            pass

        # HOLD: journal + return
        if signal_u == "HOLD":
            try:
                p_val = None
                p_src = "none"
                ens = extra0.get("ensemble_p")
                mcf = extra0.get("model_confidence_factor")
                pbe = extra0.get("p_buy_ema")
                pbr = extra0.get("p_buy_raw")

                for k, v in (("ensemble_p", ens), ("p_buy_ema", pbe), ("p_buy_raw", pbr), ("model_confidence_factor", mcf)):
                    if v is not None:
                        p_val = v
                        p_src = k
                        break

                try:
                    if p_val is not None:
                        pv = float(p_val)
                        pv = 0.0 if pv < 0.0 else (1.0 if pv > 1.0 else pv)
                        p_val = pv
                except Exception:
                    p_val = None

                self._append_hold_csv({
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": str(symbol).upper(),
                    "interval": interval,
                    "signal": "HOLD",
                    "p": p_val,
                    "p_source": p_src,
                    "ensemble_p": ens,
                    "model_confidence_factor": mcf,
                    "p_buy_ema": pbe,
                    "p_buy_raw": pbr,
                })
            except Exception:
                pass

            try:
                self.logger.info("[EXEC] Signal=HOLD symbol=%s", str(symbol).upper())
            except Exception:
                pass
            return

        # shadow mode: trade yok
        if str(os.getenv("SHADOW_MODE", "false")).lower() in ("1", "true", "yes", "on"):
            return

        # telegram notify
        try:
            self._notify_telegram(signal_u, str(symbol).upper(), interval, float(price), probs, extra0)
        except Exception:
            pass

        # training mode: trade yok
        if training_mode:
            return

        # whale veto
        try:
            whale_dir = str(extra0.get("whale_dir", "none") or "none").lower()
            whale_score = float(extra0.get("whale_score", 0.0) or 0.0)
            veto_thr = float(os.getenv("WHALE_VETO_THR", "0.70"))
            if side_norm in ("long", "short") and whale_dir in ("long", "short"):
                if whale_dir != side_norm and whale_score >= veto_thr:
                    self.logger.info(
                        "[EXEC][VETO] WHALE_VETO | side=%s whale_dir=%s whale_score=%.3f thr=%.2f -> SKIP",
                        side_norm, whale_dir, whale_score, veto_thr
                    )
                    return
        except Exception:
            pass

        # SL/TP/trailing check
        try:
            self._check_sl_tp_trailing(symbol=str(symbol).upper(), price=float(price), interval=interval)
        except Exception:
            pass

        # qty/notional
        if size is not None and float(size) > 0:
            qty = float(size)
            notional = qty * float(price)
        else:
            notional = self._compute_notional(
                symbol=str(symbol).upper(),
                signal=side_norm,
                price=float(price),
                extra=extra0,
            )
            qty = notional / float(price) if float(price) > 0 else 0.0

        if qty <= 0 or float(price) <= 0:
            return

        # open/flip logic:
        cur = self._get_position(str(symbol).upper())
        cur_side = str(cur.get("side")).lower() if cur else None

        if cur_side in ("long", "short") and cur_side != side_norm:
            self._close_position(str(symbol).upper(), float(price), reason="FLIP", interval=interval)

        cur2 = self._get_position(str(symbol).upper())
        cur2_side = str(cur2.get("side")).lower() if cur2 else None
        if cur2_side == side_norm:
            return

        pos, _opened_at = self._create_position_dict(
            signal=side_norm,
            symbol=str(symbol).upper(),
            price=float(price),
            qty=float(qty),
            notional=float(notional),
            interval=interval,
            probs=probs,
            extra=extra0,
        )
        self._set_position(str(symbol).upper(), pos)

        try:
            self.logger.info(
                "[EXEC] OPEN %s | symbol=%s qty=%.6f price=%.4f notional=%.2f interval=%s dry_run=%s",
                side_norm.upper(), str(symbol).upper(), float(qty), float(price), float(notional), interval, self.dry_run
            )

            # --- RISK / Telegram notify (position OPEN) ---
            try:
                rm = getattr(self, "risk_manager", None)
                if rm:
                    _side = str(side_norm).lower() if 'side_norm' in locals() else str(side).lower()
                    if _side in ("buy", "long"):
                        _side = "long"
                    elif _side in ("sell", "short"):
                        _side = "short"
                    rm.on_position_open(
                        symbol=str(symbol).upper(),
                        side=_side,
                        qty=float(qty),
                        notional=float(notional),
                        price=float(price),
                        interval=str(interval or ""),
                        meta={"reason": "EXEC_OPEN", **(meta or {})} if isinstance(meta, dict) else {"reason": "EXEC_OPEN"},
                    )
            except Exception:
                try:
                    if getattr(self, "logger", None):
                        self.logger.exception("[EXEC] risk_manager.on_position_open failed")
                except Exception:
                    pass

                except Exception:
                    try:
                        if getattr(self, "logger", None):
                            self.logger.exception("[EXEC] risk_manager.on_position_open failed")
                    except Exception:
                        pass

        except Exception:
            pass
