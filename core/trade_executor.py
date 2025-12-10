import logging
from datetime import datetime
from typing import Optional, Dict, Any

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
        client: Optional[Any],                 # <-- BURAYA EKLEDİK
        risk_manager: RiskManager,
        position_manager: Optional[PositionManager] = None,
        logger: Optional[logging.Logger] = None,
        dry_run: bool = True,
        # temel risk parametreleri
        base_order_notional: float = 50.0,     # varsayılan pozisyon büyüklüğü (USDT)
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
        self.client = client                  # <-- client referansı
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
    #  Ortak pozisyon dict oluşturucu (ATR / yüzde SL-TP + meta)
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
    ) -> (Dict[str, Any], str):
        opened_at = datetime.utcnow().isoformat()
        atr_value = float(extra.get("atr", 0.0) or 0.0)

        # --- SL/TP hesaplama ---
        if self.use_atr_sltp and atr_value > 0.0:
            # ATR bazlı
            if signal == "long":
                sl_price = price - self.atr_sl_mult * atr_value
                tp_price = price + self.atr_tp_mult * atr_value
            else:  # short
                sl_price = price + self.atr_sl_mult * atr_value
                tp_price = price - self.atr_tp_mult * atr_value
        else:
            # Yüzde bazlı fallback
            if signal == "long":
                sl_price = price * (1.0 - self.sl_pct)
                tp_price = price * (1.0 + self.tp_pct)
            else:
                sl_price = price * (1.0 + self.sl_pct)
                tp_price = price * (1.0 - self.tp_pct)

        # Highest/Lowest (trailing için başlangıç)
        if signal == "long":
            highest_price = price
            lowest_price = price
        else:
            highest_price = price
            lowest_price = price

        pos = {
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
            },
        }
        return pos, opened_at

    # ------------------------------------------------------------------
    #  PnL hesaplayıcı (flip + close için)
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_pnl(side: str, entry_price: float, exit_price: float, qty: float) -> float:
        """
        Basit realized PnL hesaplama (USDT cinsinden):
          long  -> (exit - entry) * qty
          short -> (entry - exit) * qty
        """
        if qty <= 0:
            return 0.0
        if side == "long":
            return (exit_price - entry_price) * qty
        elif side == "short":
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

        realized_pnl = self._calc_pnl(side=side, entry_price=entry_price, exit_price=price, qty=qty)

        self.logger.info(
            "[EXEC] Pozisyon kapatılıyor | symbol=%s side=%s qty=%.4f entry=%.2f exit=%.2f pnl=%.4f reason=%s",
            symbol,
            side,
            qty,
            entry_price,
            price,
            realized_pnl,
            reason,
        )

        # DRY_RUN değilse burada borsada close/market emirleri atılır
        if self.dry_run:
            self.logger.info("[EXEC] DRY_RUN=True, gerçek close emri gönderilmeyecek.")
        else:
            # TODO: Binance/OKX/KuCoin client ile close emri
            pass

        # Risk manager'a bildir
        try:
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
                },
            )
        except Exception as e:
            self.logger.warning("[RISK] on_position_close hata: %s", e)

        # State temizle
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
        sl_price = float(pos["sl_price"])
        tp_price = float(pos["tp_price"])
        trailing_pct = float(pos.get("trailing_pct", 0.0))
        highest_price = float(pos.get("highest_price", price))
        lowest_price = float(pos.get("lowest_price", price))

        # --- SL / TP ---
        if side == "long":
            # SL
            if price <= sl_price:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            # TP
            if price >= tp_price:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            # Trailing
            if trailing_pct > 0.0:
                if price > highest_price:
                    highest_price = price
                    pos["highest_price"] = highest_price
                    self._set_position(symbol, pos)

                trail_sl = highest_price * (1.0 - trailing_pct)
                if price <= trail_sl:
                    return self._close_position(
                        symbol, price, reason="TRAILING_STOP_LONG", interval=interval
                    )

        elif side == "short":
            # SL
            if price >= sl_price:
                return self._close_position(symbol, price, reason="SL_HIT", interval=interval)
            # TP
            if price <= tp_price:
                return self._close_position(symbol, price, reason="TP_HIT", interval=interval)

            # Trailing
            if trailing_pct > 0.0:
                if price < lowest_price:
                    lowest_price = price
                    pos["lowest_price"] = lowest_price
                    self._set_position(symbol, pos)

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
        Basit notional hesaplama:
          - base_order_notional
          - whale / model_conf vs. ile çarpılabilir
          - max_position_notional'a clamp edilir
        """

        notional = self.base_order_notional

        # Model güven faktörü
        model_conf = float(extra.get("model_confidence_factor", 1.0) or 1.0)
        notional *= model_conf

        # Whale bilgisi (varsayılan: yok)
        # main.py içinde extra["whale_meta"] şöyle dolduruluyor:
        # {
        #   "direction": "long" / "short" / "none",
        #   "score": float,
        #   ...
        # }
        # Eski kodla geri uyumluluk için extra["whale"] da fallback olarak okunuyor.
        whale_info = extra.get("whale_meta") or extra.get("whale") or {}
        whale_score = float(whale_info.get("score", 0.0) or 0.0)
        whale_direction = whale_info.get("direction")  # "long" / "short" / "none"

        # Eğer whale_score yüksekse ve sinyal ile aynı yöndeyse notional'ı boost et
        if whale_score > 0 and whale_direction in ("long", "short"):
            if signal == whale_direction:
                notional *= (1.0 + self.whale_risk_boost * whale_score)

        # max_position_notional sınırı
        if notional > self.max_position_notional:
            notional = self.max_position_notional

        # En azından min 10 USDT gibi bir taban (çok küçük olmasın)
        if notional < 10.0:
            notional = 10.0

        self.logger.info(
            "[EXEC] _compute_notional | symbol=%s signal=%s base=%.2f model_conf=%.2f whale_score=%.3f whale_dir=%s final=%.2f",
            symbol,
            signal,
            self.base_order_notional,
            model_conf,
            whale_score,
            whale_direction,
            notional,
        )

        return notional

    # ------------------------------------------------------------------
    #  Ana karar fonksiyonu
    # ------------------------------------------------------------------
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
        """
        main.py içinden çağrılır.
        signal: "long" / "short" / "hold"
        price: son fiyat
        size: istersen dışarıdan qty olarak gönderilebilir; None ise notional/price ile hesaplanır.
        """

        extra = extra or {}
        self.logger.info(
            "[EXEC] execute_decision çağrıldı | signal=%s symbol=%s price=%.2f size=%s interval=%s training_mode=%s hybrid_mode=%s probs=%s extra_keys=%s",
            signal,
            symbol,
            price,
            size,
            interval,
            training_mode,
            hybrid_mode,
            probs,
            list(extra.keys()),
        )

        # Eğitim modunda gerçek trade yok, sadece log
        if training_mode:
            self.logger.info(
                "[EXEC] TRAINING_MODE=True, sadece log. Herhangi bir pozisyon açma/kapama yapılmayacak."
            )
            return

        # Önce mevcut pozisyona SL/TP/trailing uygula
        self._check_sl_tp_trailing(symbol=symbol, price=price, interval=interval)

        # Son güncel pozisyonu tekrar al (SL/TP ile kapanmış olabilir)
        current_pos = self._get_position(symbol)
        current_side = current_pos["side"] if current_pos else None

        if signal == "hold":
            self.logger.info(
                "[EXEC] Sinyal=HOLD, yeni pozisyon açılmayacak / flip edilmeyecek."
            )
            return

        # ------------------------------------------------------------------
        #  Yeni notional / qty hesapla
        # ------------------------------------------------------------------
        if size is not None and size > 0:
            qty = float(size)
            notional = qty * price
        else:
            notional = self._compute_notional(symbol=symbol, signal=signal, price=price, extra=extra)
            qty = notional / price

        # ------------------------------------------------------------------
        #  RiskManager can_open_new_trade kontrolü (yeni pozisyon veya flip)
        # ------------------------------------------------------------------
        allowed = True
        if self.risk_manager is not None:
            try:
                allowed = self.risk_manager.can_open_new_trade(
                    symbol=symbol,
                    side=signal,
                    notional=notional,
                    price=price,
                    interval=interval,
                    meta={"has_position": bool(current_pos)},
                )
            except Exception as e:
                self.logger.warning("[RISK] can_open_new_trade hata: %s, default=True", e)
                allowed = True

        if not allowed:
            self.logger.info(
                "[RISK] can_open_new_trade=False | symbol=%s side=%s notional=%.2f -> trade açılmadı.",
                symbol,
                signal,
                notional,
            )
            return

        # ------------------------------------------------------------------
        #  Pozisyon yok ve sinyal long/short → yeni pozisyon aç
        # ------------------------------------------------------------------
        if current_pos is None:
            if signal in ("long", "short"):
                self.logger.info(
                    "[EXEC] Yeni pozisyon açılıyor | symbol=%s side=%s qty=%.4f notional=%.2f price=%.2f",
                    symbol,
                    signal,
                    qty,
                    notional,
                    price,
                )

                # DRY_RUN değilse burada gerçek emir atılır
                if self.dry_run:
                    self.logger.info(
                        "[EXEC] DRY_RUN=True, gerçek açılış emri gönderilmeyecek (sadece state+log)."
                    )
                else:
                    # TODO: Borsaya market/limit order gönder
                    pass

                pos, opened_at = self._create_position_dict(
                    signal=signal,
                    symbol=symbol,
                    price=price,
                    qty=qty,
                    notional=notional,
                    interval=interval,
                    probs=probs,
                    extra=extra,
                )
                self._set_position(symbol, pos)

                # risk tarafına bildir
                try:
                    self.risk_manager.on_position_open(
                        symbol=symbol,
                        side=signal,
                        qty=qty,
                        notional=notional,
                        price=price,
                        interval=interval,
                        meta={"opened_at": opened_at, "source": "TradeExecutor"},
                    )
                except Exception as e:
                    self.logger.warning("[RISK] on_position_open hata: %s", e)

            return

        # ------------------------------------------------------------------
        #  Zaten pozisyon var
        #  a) Aynı yön sinyal → şimdilik sadece log (istersen scale-in ekleriz)
        #  b) Ters yön sinyal → FLIP (pozisyonu kapat + yeni yön aç)
        # ------------------------------------------------------------------
        if current_side == signal:
            self.logger.info(
                "[EXEC] Mevcut pozisyon sinyal ile aynı yönde (side=%s), scale-in yapılmıyor, sadece HOLD.",
                current_side,
            )
            return

        # --- FLIP ---
        self.logger.info(
            "[EXEC] Flip sinyali | symbol=%s current_side=%s new_side=%s",
            symbol,
            current_side,
            signal,
        )

        # Önce mevcut pozisyonu kapat
        closed_pos = self._close_position(
            symbol=symbol,
            price=price,
            reason="FLIP_CLOSE",
            interval=interval,
        )

        # Sonra yeni yönlü pozisyon aç
        self.logger.info(
            "[EXEC] Flip sonrası yeni pozisyon açılıyor | symbol=%s side=%s qty=%.4f notional=%.2f price=%.2f",
            symbol,
            signal,
            qty,
            notional,
            price,
        )

        if self.dry_run:
            self.logger.info(
                "[EXEC] DRY_RUN=True, flip sonrası gerçek açılış emri gönderilmeyecek (sadece state+log)."
            )
        else:
            # TODO: Borsaya yeni pozisyon emri gönder
            pass

        new_pos, opened_at = self._create_position_dict(
            signal=signal,
            symbol=symbol,
            price=price,
            qty=qty,
            notional=notional,
            interval=interval,
            probs=probs,
            extra=extra,
        )

        # flip sonrası yeni pozisyonu kaydet
        self._set_position(symbol, new_pos)

        # risk tarafına yeni pozisyonu bildir
        try:
            self.risk_manager.on_position_open(
                symbol=symbol,
                side=signal,
                qty=qty,
                notional=notional,
                price=price,
                interval=interval,
                meta={"opened_at": opened_at, "source": "TradeExecutor_flip"},
            )
        except Exception as e:
            self.logger.warning("[RISK] on_position_open (flip) hata: %s", e)

