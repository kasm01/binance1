import os
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any
from core.trade_journal import TradeJournal


class RiskManager:
    """
    Merkezi risk kontrolü:

    - Günlük max kayıp (USDT ve/veya %)
    - Maksimum ardışık (consecutive) kayıp sayısı
    - Maksimum açık pozisyon sayısı

    Notlar:
    -------
    * Günlük kayıp ve consecutive loss kuralları, trade kapandığında
      `register_trade_result(...)` veya `on_position_close(...)` ile beslendiğinde
      tam anlamıyla çalışır.
    * Şu anda tek sembol + tek bot için tasarlandı, ama `open_positions` dict'i ile
      multi-symbol'e de genişletilebilir.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("system")

        # ------------------------------------------------------------
        # ENV tabanlı konfigürasyon
        # ------------------------------------------------------------

        # Günlük max kayıp (USDT cinsinden)
        self.daily_max_loss_usdt = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))

        # Günlük max kayıp (yüzde). Örn: 0.03 → gün başı equity'in %3'ü
        self.daily_max_loss_pct = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.03"))

        # Maksimum ardışık kayıp sayısı (örn: 5 üstü trade yok)
        self.max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

        # Maksimum açık pozisyon sayısı
        self.max_open_trades = int(os.getenv("MAX_OPEN_TRADES", "3"))

        # Gün başı equity (yüzdesel loss hesaplamak için)
        # Şimdilik ENV'den alıyoruz; daha sonra borsa API'si ile güncellenebilir.
        self.equity_start_of_day = float(
            os.getenv("EQUITY_START_OF_DAY", os.getenv("ACCOUNT_BALANCE_USDT", "1000"))
        )

        # Dahili state
        self._reset_daily_state()
        self.open_positions: Dict[str, Dict[str, Any]] = {}

        self.logger.info(
            "[RISK_INIT] daily_max_loss_usdt=%.2f daily_max_loss_pct=%.2f "
            "max_consecutive_losses=%d max_open_trades=%d equity_start_of_day=%.2f",
            self.daily_max_loss_usdt,
            self.daily_max_loss_pct,
            self.max_consecutive_losses,
            self.max_open_trades,
            self.equity_start_of_day,
        )

        self.trade_journal = TradeJournal(logger=self.logger)

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------
    def _reset_daily_state(self) -> None:
        self.current_day: date = date.today()
        self.daily_realized_pnl_usdt: float = 0.0
        self.daily_trade_count: int = 0
        self.consecutive_losses: int = 0

    def _ensure_daily_reset(self) -> None:
        today = date.today()
        if today != self.current_day:
            self.logger.info(
                "[RISK] New day detected (%s → %s), resetting daily stats.",
                self.current_day,
                today,
            )
            self._reset_daily_state()

    # ------------------------------------------------------------------
    # Dışarıya açık yardımcılar
    # ------------------------------------------------------------------
    def set_equity_start_of_day(self, equity_usdt: float) -> None:
        """
        Gün başındaki equity'i dışarıdan güncellemek için.
        Örn: her gün 00:01'de borsa API'sinden çekip buraya set edebilirsin.
        """
        self.equity_start_of_day = max(float(equity_usdt), 1.0)
        self.logger.info(
            "[RISK] equity_start_of_day güncellendi: %.2f", self.equity_start_of_day
        )

    def get_current_equity(self) -> float:
        """
        TradeExecutor, equity'i bilmek istediğinde kullanabilir.
        Şimdilik sadece equity_start_of_day + günlük PnL üzerinden pribliyoruz.
        """
        return max(self.equity_start_of_day + self.daily_realized_pnl_usdt, 1.0)

    # ------------------------------------------------------------------
    # Trade açmadan ÖNCE çağrılacak ana gate fonksiyon
    # ------------------------------------------------------------------
    def can_open_new_trade(
        self,
        *,
        symbol: str,
        side: str,
        notional: float,
        price: float,
        interval: str,
    ) -> bool:
        """
        Yeni bir trade açmadan önce çağır:

        True  -> trade açılmasına izin ver
        False -> risk limitlerinden dolayı trade reddedildi

        Burada kontrol edilenler:
        - Günlük max kayıp (USDT ve/veya %)
        - Maksimum consecutive loss
        - Maksimum açık pozisyon sayısı
        """
        self._ensure_daily_reset()

        side = side.lower()
        notional = float(notional)
        price = float(price)

        # ------------------------------------------------------------
        # 1) Günlük max kayıp (USDT)
        # ------------------------------------------------------------
        if self.daily_realized_pnl_usdt <= -self.daily_max_loss_usdt:
            self.logger.warning(
                "[RISK_BLOCK] daily_realized_pnl=%.2f USDT, limit=%.2f USDT → "
                "günlük max kayıp aşıldı, yeni trade açılmayacak.",
                self.daily_realized_pnl_usdt,
                self.daily_max_loss_usdt,
            )
            return False

        # ------------------------------------------------------------
        # 2) Günlük max kayıp (yüzde)
        # ------------------------------------------------------------
        if self.equity_start_of_day > 0:
            loss_pct = -self.daily_realized_pnl_usdt / self.equity_start_of_day
            if loss_pct >= self.daily_max_loss_pct:
                self.logger.warning(
                    "[RISK_BLOCK] daily_realized_pnl=%.2f USDT, start_eq=%.2f → "
                    "loss_pct=%.2f >= max_daily_loss_pct=%.2f, yeni trade açılmayacak.",
                    self.daily_realized_pnl_usdt,
                    self.equity_start_of_day,
                    loss_pct,
                    self.daily_max_loss_pct,
                )
                return False

        # ------------------------------------------------------------
        # 3) Max consecutive losses
        # ------------------------------------------------------------
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.warning(
                "[RISK_BLOCK] consecutive_losses=%d >= max_consecutive_losses=%d → "
                "yeni trade açılmayacak.",
                self.consecutive_losses,
                self.max_consecutive_losses,
            )
            return False

        # ------------------------------------------------------------
        # 4) Max open trades
        # ------------------------------------------------------------
        open_trades_count = len(self.open_positions)
        if open_trades_count >= self.max_open_trades:
            self.logger.warning(
                "[RISK_BLOCK] open_trades=%d >= max_open_trades=%d → "
                "yeni trade açılmayacak.",
                open_trades_count,
                self.max_open_trades,
            )
            return False

        # Eğer buraya kadar geldiysek, risk açısından trade açılabilir
        self.logger.info(
            "[RISK_OK] Trade'e izin verildi: symbol=%s side=%s notional=%.2f price=%.2f "
            "interval=%s daily_pnl=%.2f cons_losses=%d open_trades=%d",
            symbol,
            side,
            notional,
            price,
            interval,
            self.daily_realized_pnl_usdt,
            self.consecutive_losses,
            open_trades_count,
        )
        return True

    # ------------------------------------------------------------------
    # Trade / pozisyon event'leri (gelecekte PositionManager ile besleyeceksin)
    # ------------------------------------------------------------------
    def on_position_open(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        notional: float,
        price: float,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Pozisyon açıldığında çağrılabilir.
        Şu anda basitçe open_positions'a kayıt düşüyor.
        """
        self._ensure_daily_reset()

        self.open_positions[symbol] = {
            "side": side.lower(),
            "qty": float(qty),
            "notional": float(notional),
            "entry_price": float(price),
            "interval": interval,
            "opened_at": datetime.utcnow().isoformat(),
            "meta": meta or {},
        }

        self.logger.info(
            "[RISK_POS_OPEN] symbol=%s side=%s qty=%.6f notional=%.2f price=%.2f "
            "open_trades=%d",
            symbol,
            side,
            qty,
            notional,
            price,
            len(self.open_positions),
        )

    def on_position_close(
        self,
        *,
        symbol: str,
        exit_price: float,
        pnl_usdt: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Pozisyon kapandığında çağrılabilir.
        Burada:
        - Günlük realized PnL güncellenir
        - Consecutive loss sayısı güncellenir
        - open_positions'tan silinir
        """
        self._ensure_daily_reset()

        exit_price = float(exit_price)
        pnl_usdt = float(pnl_usdt)

        pos = self.open_positions.pop(symbol, None)

        self.daily_realized_pnl_usdt += pnl_usdt
        self.daily_trade_count += 1

        if pnl_usdt < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.logger.info(
            "[RISK_POS_CLOSE] symbol=%s exit_price=%.2f pnl=%.2f daily_pnl=%.2f "
            "consecutive_losses=%d open_trades=%d meta=%s",
            symbol,
            exit_price,
            pnl_usdt,
            self.daily_realized_pnl_usdt,
            self.consecutive_losses,
            len(self.open_positions),
            meta,
        )

    
        # Son olarak trade journal
        self.trade_journal.log_trade_from_close(
            symbol=symbol,
            exit_price=exit_price,
            pnl_usdt=pnl_usdt,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # TradeExecutor ile uyumlu basit kayıt fonksiyonu
    # ------------------------------------------------------------------
    def register_trade(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        notional: float,
        price: float,
        interval: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Şu anda TradeExecutor, trade açıldığında burayı çağırıyor.
        Burayı "pozisyon açıldı" gibi düşünebilirsin.

        Eğer tam PnL takibi yapmak istersen:
        - Buradan `on_position_open(...)` çağır
        - Pozisyon kapandığında PositionManager üzerinden `on_position_close(...)`
          veya `register_trade_result(...)` gibi bir fonksiyona haber ver.
        """
        self._ensure_daily_reset()

        self.logger.info(
            "[RISK_REGISTER] symbol=%s side=%s qty=%.6f notional=%.2f price=%.2f "
            "interval=%s meta=%s",
            symbol,
            side,
            qty,
            notional,
            price,
            interval,
            meta,
        )

        # Tek sembol + tek pozisyon senaryosunda direkt on_position_open çağırmak isteyebilirsin.
        # Şimdilik sadece log atıyoruz; ister burayı aktif edersin ister dışarıdan çağırırsın:
        #
        # self.on_position_open(
        #     symbol=symbol,
        #     side=side,
        #     qty=qty,
        #     notional=notional,
        #     price=price,
        #     interval=interval,
        #     meta=meta,
        # )

    # ------------------------------------------------------------------
    # İsteğe bağlı: manuel sonuc kaydı (PnL)
    # ------------------------------------------------------------------
    def register_trade_result(
        self,
        *,
        symbol: str,
        pnl_usdt: float,
        closed_at: Optional[datetime] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Eğer tam pozisyon takibi yapmayıp sadece sonuç PnL'lerini biliyorsan,
        bu fonksiyonla günlük PnL ve consecutive loss'ları güncelleyebilirsin.
        """
        self._ensure_daily_reset()

        pnl_usdt = float(pnl_usdt)
        self.daily_realized_pnl_usdt += pnl_usdt
        self.daily_trade_count += 1

        if pnl_usdt < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.logger.info(
            "[RISK_RESULT] symbol=%s pnl=%.2f daily_pnl=%.2f cons_losses=%d closed_at=%s meta=%s",
            symbol,
            pnl_usdt,
            self.daily_realized_pnl_usdt,
            self.consecutive_losses,
            (closed_at or datetime.utcnow()).isoformat(),
            meta,
        )
