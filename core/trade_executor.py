import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from config import config
from core.risk_manager import RiskManager
from core.position_manager import PositionManager

logger = logging.getLogger("system")


class TradeExecutor:
    """
    TradeExecutor:
      - RiskManager ile entegre
      - PositionManager ile açık pozisyon state'ini yönetir
      - ATR bazlı SL/TP + trailing stop uygular
      - Flip (long -> short, short -> long) durumunda PnL hesaplar
      - DRY_RUN modunda gerçek emir atmadan her şeyi simüle eder
    """

    def __init__(
        self,
        client: Optional[Any],
        risk_manager: RiskManager,
        position_manager: Optional[PositionManager] = None,
        logger: Optional[logging.Logger] = None,
        dry_run: bool = True,
        # temel risk parametreleri
        base_order_notional: float = 50.0,
        max_position_notional: float = 500.0,
        max_leverage: float = 3.0,
        # SL/TP & trailing
        sl_pct: float = 0.01,
        tp_pct: float = 0.02,
        trailing_pct: float = 0.01,
        use_atr_sltp: bool = True,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        # whale risk kancası
        whale_risk_boost: float = 2.0,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.position_manager = position_manager
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

        # Eğer PositionManager yoksa en basit fallback olarak in-memory dict
        self._local_positions: Dict[str, Dict[str, Any]] = {}


    # ------------------------------------------------------------------
    #  Low-level position access (PositionManager varsa onu kullan)
    # ------------------------------------------------------------------
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        if self.position_manager is not None:
            try:
                return self.position_manager.get_position(symbol)
            except Exception as e:
                self.logger.warning(
                    "[EXEC] PositionManager.get_position hata: %s, local fallback kullanılacak.",
                    e,
                )
        return self._local_positions.get(symbol)

    def _set_position(self, symbol: str, pos: Dict[str, Any]) -> None:
        if self.position_manager is not None:
            try:
                self.position_manager.set_position(symbol, pos)
            except Exception as e:
                self.logger.warning(
                    "[EXEC] PositionManager.set_position hata: %s, local fallback'e yazılıyor.",
                    e,
                )
                self._local_positions[symbol] = pos
        else:
            self._local_positions[symbol] = pos

    def _clear_position(self, symbol: str) -> None:
        if self.position_manager is not None:
            try:
                self.position_manager.clear_position(symbol)
            except Exception as e:
                self.logger.warning(
                    "[EXEC] PositionManager.clear_position hata: %s, local fallback temizlenecek.",
                    e,
                )
        self._local_positions.pop(symbol, None)

    # ------------------------------------------------------------------
    #  Ortak pozisyon dict oluşturucu (ATR / yüzde SL-TP + meta + entry snapshot)
    # ------------------------------------------------------------------
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

        # --- SL/TP hesaplama ---
        if self.use_atr_sltp and atr_value > 0.0:
            if signal == "long":
                sl_price = price - self.atr_sl_mult * atr_value
                tp_price = price + self.atr_tp_mult * atr_value
            else:  # short
                sl_price = price + self.atr_sl_mult * atr_value
                tp_price = price - self.atr_tp_mult * atr_value
        else:
            if signal == "long":
                sl_price = price * (1.0 - self.sl_pct)
                tp_price = price * (1.0 + self.tp_pct)
            else:
                sl_price = price * (1.0 + self.sl_pct)
                tp_price = price * (1.0 - self.tp_pct)

        # Highest/Lowest (trailing için başlangıç)
        highest_price = price
        lowest_price = price

        pos: Dict[str, Any] = {
            "symbol": symbol,
            "side": signal,
            "qty": qty,
            "entry_price": price,
            "notional": notional,
            "interval": interval,
            "opened_at": opened_at,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "trailing_pct": self.trailing_pct,
            "atr_value": atr_value,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "meta": {
                "probs": probs,
                "extra": extra,
                # journal flatten (entry snapshot)
                "bt_p_buy_raw": (extra.get("ensemble_p") or extra.get("model_confidence_factor") or extra.get("p_buy_ema") or extra.get("p_buy_raw")) if isinstance(extra, dict) else None,
                "bt_p_buy_ema": probs.get("p_single") if isinstance(probs, dict) else None,
                "bt_ema_alpha": None,
            },
        }

        # ------------------------------------------------------------------
        # ENTRY whale snapshot (for analysis / optimization)
        # ------------------------------------------------------------------
        try:
            entry_whale_dir = str(extra.get("whale_dir", "none") or "none")
            entry_whale_score = float(extra.get("whale_score", 0.0) or 0.0)
            entry_whale_thr = float(extra.get("whale_thr", 0.0) or 0.0)
            entry_whale_on = bool(extra.get("whale_on", False))
            entry_whale_alignment = str(
                extra.get("whale_alignment", "no_whale") or "no_whale"
            )
            entry_model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)

            meta = pos.get("meta")
            if not isinstance(meta, dict):
                pos["meta"] = {}
                meta = pos["meta"]

            meta.update(
                {
                    "entry_whale_dir": entry_whale_dir,
                    "entry_whale_score": entry_whale_score,
                    "entry_whale_thr": entry_whale_thr,
                    "entry_whale_on": entry_whale_on,
                    "entry_whale_alignment": entry_whale_alignment,
                    "entry_model_confidence_factor": entry_model_conf,
                }
            )
        except Exception:
            pass

        return pos, opened_at

    # ------------------------------------------------------------------
    #  PnL hesaplayıcı (flip + close için)
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_pnl(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        if qty <= 0:
            return 0.0
        if side == "long":
            return (exit_price - entry_price) * qty
        if side == "short":
            return (entry_price - exit_price) * qty
        return 0.0

    # ------------------------------------------------------------------
    #  Pozisyon kapama helper (PnL + risk manager + state temizleme)
    # ------------------------------------------------------------------
    def _close_position(
        self,
        symbol: str,
        price: float,
        reason: str,
        interval: str,
    ) -> Optional[Dict[str, Any]]:
        pos = self._get_position(symbol)
        if not pos:
            return None

        side = pos["side"]
        qty = float(pos["qty"])
        entry_price = float(pos["entry_price"])
        notional = float(pos.get("notional", qty * entry_price))

        realized_pnl = self._calc_pnl(
            side=side, entry_price=entry_price, exit_price=price, qty=qty
        )


        if self.dry_run:
            self.logger.info("[EXEC] DRY_RUN=True, gerçek close emri gönderilmeyecek.")
        else:
            # TODO: Binance/OKX/KuCoin client ile close emri
            pass

        try:
            meta_dict = pos.get("meta") if isinstance(pos.get("meta"), dict) else {}
            probs_dict = meta_dict.get("probs") if isinstance(meta_dict.get("probs"), dict) else {}
            extra_dict = meta_dict.get("extra") if isinstance(meta_dict.get("extra"), dict) else {}

            # --- bt_* force-set (CLOSE): meta_dict üzerinde kesinleştir ---
            try:
                meta_dict["bt_p_buy_raw"] = extra_dict.get("ensemble_p")
                meta_dict["bt_p_buy_ema"] = probs_dict.get("p_single")
                meta_dict["bt_ema_alpha"] = None
            except Exception:
                pass

            self.risk_manager.on_position_close(
                symbol=symbol,
                side=side,
                qty=qty,
                notional=notional,
                price=price,
                interval=interval,
                realized_pnl=realized_pnl,
                meta={
                    "reason": reason,
                    "entry_price": entry_price,
                    "closed_side": side,
                    "interval": interval,
                    "qty": qty,
                    "notional": notional,
                    "probs": probs_dict,
                    "extra": extra_dict,
                    # --- flatten for journal ---
                    "bt_p_buy_raw": meta_dict.get("bt_p_buy_raw"),
                    "bt_p_buy_ema": meta_dict.get("bt_p_buy_ema"),
                    "bt_ema_alpha": meta_dict.get("bt_ema_alpha"),
                },
            )
        except Exception as e:
            self.logger.warning("[RISK] on_position_close hata: %s", e)

        self._clear_position(symbol)

        pos["closed_at"] = datetime.utcnow().isoformat()
        pos["close_price"] = price
        pos["realized_pnl"] = realized_pnl
        pos["close_reason"] = reason
        return pos


    # ------------------------------------------------------------------
    #  SL/TP + trailing stop kontrolleri
    # ------------------------------------------------------------------
    def _check_sl_tp_trailing(
        self,
        symbol: str,
        price: float,
        interval: str,
    ) -> Optional[Dict[str, Any]]:
        pos = self._get_position(symbol)
        if not pos:
            return None

        side = pos["side"]
        sl_raw = pos.get("sl_price")
        tp_raw = pos.get("tp_price")
        trailing_raw = pos.get("trailing_pct", 0.0)

        sl_price = float(sl_raw) if sl_raw is not None else None
        tp_price = float(tp_raw) if tp_raw is not None else None
        trailing_pct = float(trailing_raw) if trailing_raw not in (None, "") else 0.0
        highest_price = float(pos.get("highest_price", price))
        lowest_price = float(pos.get("lowest_price", price))

        if side == "long":
            if sl_price is not None and price <= sl_price:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            if tp_price is not None and price >= tp_price:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if price > highest_price:
                    pos["highest_price"] = price
                    self._set_position(symbol, pos)
                    highest_price = price

                trail_sl = highest_price * (1.0 - trailing_pct)
                if price <= trail_sl:
                    return self._close_position(
                        symbol, price, reason="TRAILING_STOP_LONG", interval=interval
                    )

        elif side == "short":
            if sl_price is not None and price >= sl_price:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            if tp_price is not None and price <= tp_price:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            if trailing_pct > 0.0:
                if price < lowest_price:
                    pos["lowest_price"] = price
                    self._set_position(symbol, pos)
                    lowest_price = price

                trail_sl = lowest_price * (1.0 + trailing_pct)
                if price >= trail_sl:
                    return self._close_position(
                        symbol, price, reason="TRAILING_STOP_SHORT", interval=interval
                    )

        return None

    # ------------------------------------------------------------------
    #  Notional / qty hesaplama (whale + risk çarpanları dahil)
    # ------------------------------------------------------------------
    def _compute_notional(
        self,
        symbol: str,
        signal: str,
        price: float,
        extra: Dict[str, Any],
    ) -> float:
        """
        Agresif notional hesaplama:
          - base_order_notional
          - model_confidence_factor
          - whale (dir + score)  [direct keys + whale_meta fallback]
          - AGGRESSIVE_MODE / MAX_RISK_MULTIPLIER
          - max_position_notional clamp
        """

        aggressive_mode = bool(getattr(config, "AGGRESSIVE_MODE", True))
        max_risk_mult = float(getattr(config, "MAX_RISK_MULTIPLIER", 4.0))
        whale_boost_thr = float(getattr(config, "WHALE_STRONG_THR", 0.6))
        whale_veto_thr = float(getattr(config, "WHALE_VETO_THR", 0.6))

        base = float(self.base_order_notional)
        model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)

        whale_dir = None
        whale_score = 0.0

        try:
            if isinstance(extra, dict):
                if extra.get("whale_dir") is not None:
                    whale_dir = extra.get("whale_dir")
                if extra.get("whale_score") is not None:
                    whale_score = float(extra.get("whale_score") or 0.0)

            whale_info = extra.get("whale_meta") or extra.get("whale")
            if isinstance(whale_info, dict):
                if whale_dir is None:
                    whale_dir = whale_info.get("direction") or whale_info.get("dir")
                if whale_score == 0.0:
                    whale_score = float(whale_info.get("score") or 0.0)
        except Exception:
            whale_dir = None
            whale_score = 0.0

        whale_dir = "none" if whale_dir is None else str(whale_dir).lower()

        aggr_factor = 1.0

        if aggressive_mode:
            if whale_score > 0.0 and whale_dir in ("long", "short"):
                if whale_dir == signal:
                    if whale_score >= whale_boost_thr:
                        aggr_factor += self.whale_risk_boost * max(
                            0.0, whale_score - whale_boost_thr
                        )
                else:
                    if whale_score >= whale_veto_thr:
                        aggr_factor -= 0.8 * whale_score
                    else:
                        aggr_factor -= 0.4 * whale_score

            mc = max(0.0, min(model_conf, 1.0))
            aggr_factor *= (0.5 + 0.5 * mc)

        aggr_factor = max(0.0, aggr_factor)
        aggr_factor = min(aggr_factor, max_risk_mult)

        notional = base * aggr_factor

        if notional > self.max_position_notional:
            notional = self.max_position_notional
        if notional < 10.0:
            notional = 10.0


        return notional

    # ------------------------------------------------------------------
    #  Ana karar fonksiyonu
    # ------------------------------------------------------------------
    async def _open_or_flip_position(
        self,
        symbol,
        side,
        qty,
        price,
        interval=None,
        meta=None,
        extra=None,
        **kwargs,
    ):
        """Backward-compat helper.

        execute_decision() eski sürümlerde bu metodu çağırabiliyor.
        Bu wrapper:
          - mümkünse ters pozisyonu kapatır
          - sonra yeni pozisyonu açar
        Var olan open/close methodlarını (hangi isimle varsa) kullanır.
        """
        pm = getattr(self, "position_manager", None)

        # Detect current side if possible
        current_side = None
        try:
            if pm is not None and hasattr(pm, "get_open_position"):
                pos = pm.get_open_position(symbol)
                current_side = getattr(pos, "side", None) if pos else None
        except Exception:
            current_side = None

        # Close if opposite
        try:
            if current_side and str(current_side).lower() != str(side).lower():
                if hasattr(self, "_close_position"):
                    await self._close_position(symbol=symbol, price=price, interval=interval, meta=meta, extra=extra)
                elif hasattr(self, "close_position"):
                    res = self.close_position(symbol=symbol, price=price, interval=interval, meta=meta, extra=extra)
                    if hasattr(res, "__await__"):
                        await res
        except Exception:
            pass

        # Open new
        if hasattr(self, "_open_position"):
            return await self._open_position(symbol=symbol, side=side, qty=qty, price=price, interval=interval, meta=meta, extra=extra)
        if hasattr(self, "open_position"):
            res = self.open_position(symbol=symbol, side=side, qty=qty, price=price, interval=interval, meta=meta, extra=extra)
            if hasattr(res, "__await__"):
                return await res
            return res
        if hasattr(self, "execute_trade"):
            res = self.execute_trade(symbol=symbol, signal=side, qty=qty, price=price, interval=interval, meta=meta, extra=extra)
            if hasattr(res, "__await__"):
                return await res
            return res

        raise AttributeError("No open_position/execute_trade method found to open a position.")

    # ------------------------------------------------------------------

    # Backtest/Paper uyumluluk shim'i

    # _open_or_flip_position eski akışta open_position/execute_trade arayabilir.

    # ------------------------------------------------------------------

    async def open_position(self, *args, **kwargs):

        """Backtest/Paper trade uyumluluk katmanı.

        Varsa position_manager.open_position'a delege eder.

        Yoksa DRY_RUN/backtest için no-op yapar.

        """

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
        """Eski isimle alias."""
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
        # --------------------------------------------------
        # BT debug (opsiyonel)
        # --------------------------------------------------
        import os
        bt_debug = os.getenv("BT_DEBUG", "0").strip() == "1"
        try:
            if bt_debug and getattr(self, "logger", None):
                raw_s = str(signal) if signal is not None else ""
                norm = raw_s.strip().lower()
                self.logger.info(
                    "[BT-DBG] execute_decision symbol=%s raw=%s norm=%s price=%s interval=%s size=%s",
                    symbol, raw_s, norm, price, interval, size
                )
        except Exception:
            pass

        # --------------------------------------------------
        # SHADOW MODE: trade yok, sadece log
        # --------------------------------------------------
        # --------------------------------------------------
        # signal mapping (BUY/SELL/HOLD) + long/short uyumu
        # --------------------------------------------------
        try:
            _raw = signal
            _s = str(_raw).strip().lower()
            if _s in ("buy", "long", "1", "true"):
                signal = "BUY"
            elif _s in ("sell", "short", "-1", "false"):
                signal = "SELL"
            else:
                signal = "HOLD"
        except Exception:
            signal = "HOLD"

        shadow = os.getenv("SHADOW_MODE", "false").lower() in ("1", "true", "yes", "on")
        if shadow:
            return

        extra = extra or {}

        if training_mode:
            return

        # SL / TP / trailing kontrolü
        self._check_sl_tp_trailing(symbol=symbol, price=price, interval=interval)

        current_pos = self._get_position(symbol)
        current_side = current_pos["side"] if current_pos else None

        if signal == "hold":
            if self.logger:
                self.logger.info("[EXEC] Signal=HOLD")
            return

        # --------------------------------------------------
        # qty / notional hesaplama
        # --------------------------------------------------
        if size is not None and size > 0:
            qty = float(size)
            notional = qty * price
        else:
            notional = self._compute_notional(
                symbol=symbol,
                signal=signal,
                price=price,
                extra=extra,
            )
            qty = notional / price

        # --------------------------------------------------
        # BT ENSEMBLE SIZING (🔥 DOĞRU YER 🔥)
        # --------------------------------------------------
        if os.getenv("ENABLE_BT_SIZING", "0").lower() in ("1", "true", "yes"):
            try:
                pivot = float(os.getenv("BT_SIZING_PIVOT", "0.75"))
                smin = float(os.getenv("BT_SIZING_MIN", "0.70"))
                smax = float(os.getenv("BT_SIZING_MAX", "1.20"))

                p_ens = None
                if isinstance(extra, dict):
                    p_ens = (
                        extra.get("ensemble_p")
                        or extra.get("p_buy_raw")
                        or extra.get("p_buy_ema")
                        or extra.get("model_confidence_factor")
                    )

                if p_ens is not None:
                    p_ens = float(p_ens)

                    if pivot <= 0:
                        factor = 1.0
                    elif p_ens <= pivot:
                        factor = smin + (p_ens / pivot) * (1.0 - smin)
                    else:
                        denom = (1.0 - pivot) or 1e-9
                        factor = 1.0 + ((p_ens - pivot) / denom) * (smax - 1.0)

                    qty *= factor
                    notional = qty * price

                    if self.logger:
                        self.logger.info(
                            "[EXEC][BT] sizing | p=%.4f factor=%.3f qty=%.6f notional=%.2f",
                            p_ens,
                            factor,
                            qty,
                            notional,
                        )
                else:
                    if self.logger:
                        self.logger.info("[EXEC][BT] sizing skipped | p missing")

            except Exception as e:
                if self.logger:
                    self.logger.warning("[EXEC][BT] sizing error: %s", e)

        # --------------------------------------------------
        # Pozisyon aç / flip
        # --------------------------------------------------
        await self._open_or_flip_position(
            side=signal,
            signal=signal,
            symbol=symbol,
            price=price,
            qty=qty,
            notional=notional,
            interval=interval,
            probs=probs,
            extra=extra,
        )
