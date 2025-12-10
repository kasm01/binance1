from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any


@dataclass
class RiskManager:
    """
    RiskManager:
      - Günlük max kayıp (USDT + yüzde)
      - Max consecutive loss
      - Max open trades
      - Pozisyon açma/kapama olaylarından beslenen basit state
      - Son trade metalarını ve basit istatistikleri tutar
    """

    # Konfigürasyon parametreleri
    daily_max_loss_usdt: float = 100.0
    daily_max_loss_pct: float = 0.03
    max_consecutive_losses: int = 5
    max_open_trades: int = 3
    equity_start_of_day: float = 1000.0

    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("RiskManager")
    )

    # Günlük / runtime state
    current_day: date = field(
        default_factory=lambda: datetime.now(timezone.utc).date()
    )
    daily_realized_pnl: float = 0.0
    consecutive_losses: int = 0
    open_trades: int = 0

    # ---- Eklenen meta alanlar (son trade ve istatistikler) ----
    last_open_trade: Optional[Dict[str, Any]] = None
    last_close_trade: Optional[Dict[str, Any]] = None
    last_realized_pnl: float = 0.0
    total_trades: int = 0        # kapanmış (realized) trade sayısı
    total_wins: int = 0
    total_losses: int = 0

    def __post_init__(self) -> None:
        self.daily_max_loss_usdt = float(self.daily_max_loss_usdt)
        self.daily_max_loss_pct = float(self.daily_max_loss_pct)
        self.max_consecutive_losses = int(self.max_consecutive_losses)
        self.max_open_trades = int(self.max_open_trades)
        self.equity_start_of_day = float(self.equity_start_of_day)

        if self.logger:
            self.logger.info(
                "[RISK] Init: daily_max_loss_usdt=%.2f, daily_max_loss_pct=%.4f, "
                "max_consecutive_losses=%d, max_open_trades=%d, equity_start_of_day=%.2f",
                self.daily_max_loss_usdt,
                self.daily_max_loss_pct,
                self.max_consecutive_losses,
                self.max_open_trades,
                self.equity_start_of_day,
            )

    # --------------------------------------------------
    # İç helper: gün değiştiyse günlük metrikleri resetle
    # --------------------------------------------------
    def _maybe_reset_day(self) -> None:
        now_day = datetime.now(timezone.utc).date()
        if now_day != self.current_day:
            if self.logger:
                self.logger.info(
                    "[RISK] Yeni gün tespit edildi (%s -> %s), günlük metrikler resetleniyor. "
                    "Önceki daily_realized_pnl=%.2f, consecutive_losses=%d",
                    self.current_day,
                    now_day,
                    self.daily_realized_pnl,
                    self.consecutive_losses,
                )
            self.current_day = now_day
            self.daily_realized_pnl = 0.0
            self.consecutive_losses = 0
            # open_trades sayısını burada resetlemiyoruz; aktif pozisyonlar
            # PositionManager tarafından yönetiliyor, biz sadece sayacı tutuyoruz.

    # --------------------------------------------------
    # Pozisyon açılabilir mi?
    # TradeExecutor.execute_decision şuradan çağırıyor:
    #   can_open_new_trade(symbol=..., side=..., price=..., notional=..., interval=...)
    # --------------------------------------------------
    def can_open_new_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        notional: float,
        interval: str,
        **kwargs: Any,
    ) -> bool:
        """
        Yeni pozisyon açmadan önce risk limitlerini kontrol eder.
        Tüm parametreler şu an sadece log için kullanılıyor, ama gelecekte
        position-size dinamikleştirmede de kullanılabilir.
        """
        self._maybe_reset_day()

        # 1) Günlük mutlak kayıp limiti
        if self.daily_realized_pnl <= -self.daily_max_loss_usdt:
            if self.logger:
                self.logger.warning(
                    "[RISK] Günlük max kayıp USDT limiti aşıldı: daily_realized_pnl=%.2f <= -%.2f, "
                    "yeni trade reddedildi (symbol=%s side=%s interval=%s)",
                    self.daily_realized_pnl,
                    self.daily_max_loss_usdt,
                    symbol,
                    side,
                    interval,
                )
            return False

        # 2) Günlük yüzde bazlı kayıp limiti
        equity_now = self.equity_start_of_day + self.daily_realized_pnl
        equity_drawdown_pct = (
            0.0
            if self.equity_start_of_day <= 0
            else (self.equity_start_of_day - equity_now) / self.equity_start_of_day
        )

        if equity_drawdown_pct >= self.daily_max_loss_pct:
            if self.logger:
                self.logger.warning(
                    "[RISK] Günlük max loss %% limiti aşıldı: drawdown=%.4f >= %.4f, "
                    "yeni trade reddedildi (equity_now=%.2f, symbol=%s, side=%s, interval=%s)",
                    equity_drawdown_pct,
                    self.daily_max_loss_pct,
                    equity_now,
                    symbol,
                    side,
                    interval,
                )
            return False

        # 3) Max consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            if self.logger:
                self.logger.warning(
                    "[RISK] Max consecutive loss limiti aşıldı: consecutive_losses=%d >= %d, "
                    "yeni trade reddedildi (symbol=%s, side=%s, interval=%s)",
                    self.consecutive_losses,
                    self.max_consecutive_losses,
                    symbol,
                    side,
                    interval,
                )
            return False

        # 4) Max open trades
        if self.open_trades >= self.max_open_trades:
            if self.logger:
                self.logger.warning(
                    "[RISK] Max open trades limiti aşıldı: open_trades=%d >= %d, "
                    "yeni trade reddedildi (symbol=%s, side=%s, interval=%s)",
                    self.open_trades,
                    self.max_open_trades,
                    symbol,
                    side,
                    interval,
                )
            return False

        # Hepsi geçti -> trade açılabilir
        if self.logger:
            self.logger.info(
                "[RISK] can_open_new_trade=TRUE | symbol=%s side=%s price=%.2f notional=%.2f "
                "daily_realized_pnl=%.2f consecutive_losses=%d open_trades=%d",
                symbol,
                side,
                price,
                notional,
                self.daily_realized_pnl,
                self.consecutive_losses,
                self.open_trades,
            )

        return True

    # --------------------------------------------------
    # Pozisyon açıldığında TradeExecutor tarafından çağrılır
    # --------------------------------------------------
    def on_position_open(
        self,
        symbol: str,
        side: str,
        qty: float,
        notional: float,
        price: float,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._maybe_reset_day()
        self.open_trades += 1

        ts = datetime.now(timezone.utc).isoformat()
        self.last_open_trade = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "notional": float(notional),
            "price": float(price),
            "interval": interval,
            "meta": meta or {},
        }

        if self.logger:
            self.logger.info(
                "[RISK] on_position_open | symbol=%s side=%s qty=%.6f notional=%.2f "
                "price=%.2f interval=%s open_trades=%d meta=%s",
                symbol,
                side,
                qty,
                notional,
                price,
                interval,
                self.open_trades,
                meta,
            )

    # --------------------------------------------------
    # Pozisyon kapandığında / flip olduğunda TradeExecutor tarafından çağrılır
    # --------------------------------------------------
    def on_position_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        notional: float,
        price: float,
        interval: str,
        realized_pnl: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        realized_pnl: USDT cinsinden, pozisyonun toplam kar/zararı
        price      : kapatma fiyatı (exit)
        meta       : TradeExecutor._close_position içinden gelir
                     örn: {"reason": "...", "entry_price": ...}
        """
        self._maybe_reset_day()

        # Açık pozisyon sayısını azalt
        self.open_trades = max(0, self.open_trades - 1)

        # Günlük PnL ve son PnL
        self.daily_realized_pnl += float(realized_pnl)
        self.last_realized_pnl = float(realized_pnl)

        # Toplam trade sayaçları
        self.total_trades += 1
        if realized_pnl > 0:
            self.total_wins += 1
        elif realized_pnl < 0:
            self.total_losses += 1

        # Consecutive losses
        if realized_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Son close metası
        ts = datetime.now(timezone.utc).isoformat()
        m = meta or {}
        self.last_close_trade = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "notional": float(notional),
            "exit_price": float(price),
            "entry_price": float(m.get("entry_price", 0.0)),
            "interval": interval,
            "realized_pnl": float(realized_pnl),
            "reason": m.get("reason"),
            "meta": m,
        }

        if self.logger:
            self.logger.info(
                "[RISK] on_position_close | symbol=%s side=%s qty=%.6f notional=%.2f "
                "price=%.2f interval=%s realized_pnl=%.2f "
                "daily_realized_pnl=%.2f consecutive_losses=%d open_trades=%d "
                "total_trades=%d wins=%d losses=%d meta=%s",
                symbol,
                side,
                qty,
                notional,
                price,
                interval,
                realized_pnl,
                self.daily_realized_pnl,
                self.consecutive_losses,
                self.open_trades,
                self.total_trades,
                self.total_wins,
                self.total_losses,
                meta,
            )
