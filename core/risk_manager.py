from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any

try:
    # opsiyonel: varsa kullanırız
    from core.trade_journal import TradeJournal
except Exception:
    TradeJournal = None  # type: ignore


@dataclass
class RiskManager:
    """
    RiskManager:
      - Günlük max kayıp (USDT + yüzde)
      - Max consecutive loss
      - Max open trades
      - Pozisyon açma/kapama olaylarından beslenen basit state
      - Son trade metalarını ve basit istatistikleri tutar
      - Intraday reset + new-day auto reset destekler
      - TelegramBot enjekte edilirse pozisyon aç/kapa otomatik bildirim yollar
    """

    # -----------------------------
    # Konfigürasyon parametreleri
    # -----------------------------
    daily_max_loss_usdt: float = 100.0
    daily_max_loss_pct: float = 0.03
    max_consecutive_losses: int = 5
    max_open_trades: int = 3
    equity_start_of_day: float = 1000.0

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("RiskManager"))

    # -----------------------------
    # Günlük / runtime state
    # -----------------------------
    current_day: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
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

    # ---- Entegrasyonlar ----
    tg_bot: Any = field(default=None, init=False, repr=False)      # TelegramBot injector
    journal: Any = field(default=None, init=False, repr=False)     # TradeJournal (opsiyonel)

    def set_telegram_bot(self, tg_bot) -> None:
        """Inject TelegramBot instance for auto notifications."""
        self.tg_bot = tg_bot

    def set_trade_journal(self, journal) -> None:
        """Inject TradeJournal if you want to persist trades."""
        self.journal = journal

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
    # Telegram helper
    # --------------------------------------------------
    def _tg_notify(self, text: str) -> None:
        tg = getattr(self, "tg_bot", None)
        if not tg:
            return
        try:
            tg.send_message(text, parse_mode=None)
        except Exception:
            try:
                if self.logger:
                    self.logger.exception("[RISK] Telegram notify failed")
            except Exception:
                pass

    # --------------------------------------------------
    # Debug helper: limitleri görünür yap
    # --------------------------------------------------
    def _debug_limits(self, symbol: str = "", interval: str = "") -> None:
        """Günlük limit / stop nedenlerini görünür yapmak için."""
        try:
            if self.logger:
                self.logger.warning(
                    "[RISK-DBG] symbol=%s interval=%s daily_realized_pnl=%s max_daily_loss_usdt=%s "
                    "daily_max_loss_pct=%s equity_start_of_day=%s consecutive_losses=%s "
                    "max_consecutive_losses=%s open_trades=%s max_open_trades=%s",
                    symbol,
                    interval,
                    getattr(self, "daily_realized_pnl", None),
                    getattr(self, "daily_max_loss_usdt", None),
                    getattr(self, "daily_max_loss_pct", None),
                    getattr(self, "equity_start_of_day", None),
                    getattr(self, "consecutive_losses", None),
                    getattr(self, "max_consecutive_losses", None),
                    getattr(self, "open_trades", None),
                    getattr(self, "max_open_trades", None),
                )
        except Exception:
            pass

    # --------------------------------------------------
    # Intraday reset API
    # --------------------------------------------------
    def reset_intraday(self, new_equity_start: Optional[float] = None, reset_open_trades: bool = False) -> None:
        """
        Gün içi reset: daily pnl, consecutive losses vb.
        - reset_open_trades=False önerilir (PositionManager gerçek pozisyonları yönetiyor)
        """
        if new_equity_start is not None:
            try:
                self.equity_start_of_day = float(new_equity_start)
            except Exception:
                pass

        self.daily_realized_pnl = 0.0
        self.consecutive_losses = 0
        self.last_realized_pnl = 0.0

        if reset_open_trades:
            self.open_trades = 0

        if self.logger:
            self.logger.info(
                "[RISK] Intraday reset applied | equity_start_of_day=%.2f reset_open_trades=%s",
                float(self.equity_start_of_day),
                str(bool(reset_open_trades)),
            )

    def reset_if_new_day(self, now_utc: Optional[datetime] = None, new_equity_start: Optional[float] = None) -> bool:
        """
        Dışarıdan çağrılabilen güvenli 'yeni gün' kontrolü.
        Yeni günse reset uygular ve True döner.
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        d = now_utc.date()
        if d != self.current_day:
            if self.logger:
                self.logger.info("[RISK] New day detected via reset_if_new_day (%s -> %s)", self.current_day, d)
            self.current_day = d
            self.reset_intraday(new_equity_start=new_equity_start, reset_open_trades=False)
            return True
        return False

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
            # open_trades resetlenmez

    def tick(self) -> None:
        """Loop başında güvenle çağrılabilir."""
        self._maybe_reset_day()

    # --------------------------------------------------
    # Pozisyon açılabilir mi?
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
                self._debug_limits(symbol, interval)
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
                self._debug_limits(symbol, interval)
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
                self._debug_limits(symbol, interval)
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
                self._debug_limits(symbol, interval)
            return False

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
    # Pozisyon açıldığında çağrılır
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

        # Telegram notify
        self._tg_notify(
            "🚀 POSITION OPENED\n"
            f"symbol={symbol} side={side} qty={qty}\n"
            f"price={price} interval={interval}"
        )

    # --------------------------------------------------
    # Pozisyon kapandığında çağrılır
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
        self._maybe_reset_day()

        # open_trades azalt (negatife düşmesin)
        self.open_trades = max(0, int(self.open_trades) - 1)

        r = float(realized_pnl)
        self.last_realized_pnl = r
        self.daily_realized_pnl += r

        self.total_trades += 1
        if r >= 0:
            self.total_wins += 1
            self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1

        ts = datetime.now(timezone.utc).isoformat()
        self.last_close_trade = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "notional": float(notional),
            "price": float(price),
            "interval": interval,
            "realized_pnl": r,
            "daily_realized_pnl": float(self.daily_realized_pnl),
            "meta": meta or {},
        }

        if self.logger:
            self.logger.info(
                "[RISK] on_position_close | symbol=%s side=%s qty=%.6f notional=%.2f price=%.2f interval=%s "
                "realized_pnl=%.4f daily_realized_pnl=%.4f consecutive_losses=%d open_trades=%d "
                "total_trades=%d wins=%d losses=%d meta=%s",
                symbol,
                side,
                qty,
                notional,
                price,
                interval,
                r,
                self.daily_realized_pnl,
                self.consecutive_losses,
                self.open_trades,
                self.total_trades,
                self.total_wins,
                self.total_losses,
                meta,
            )

        # Telegram notify
        self._tg_notify(
            "✅ POSITION CLOSED\n"
            f"symbol={symbol} side={side} qty={qty}\n"
            f"price={price} interval={interval}\n"
            f"realized_pnl={r:.4f} daily_realized_pnl={float(self.daily_realized_pnl):.4f}"
        )

        # Journal (opsiyonel)
        try:
            j = getattr(self, "journal", None)
            if j is not None:
                # TradeJournal API'si projene göre değişebilir; kırmamak için try/except
                try:
                    j.log_close(self.last_close_trade)
                except Exception:
                    pass
        except Exception:
            pass
