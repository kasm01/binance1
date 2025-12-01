# trading/risk_manager.py

import json
import os
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Tuple

from core.logger import system_logger


@dataclass
class DailyRiskState:
    """
    Günlük realized PnL ve trading halt durumunu tutar.

    trade_date: 'YYYY-MM-DD' (bugünün tarihi)
    realized_pnl: sadece KAPANAN pozisyonlardan gelen kar/zarar (USDT)
    trading_halted: günlük zarar limiti aşıldıysa True
    """
    trade_date: str
    realized_pnl: float = 0.0
    trading_halted: bool = False


class RiskManager:
    """
    - max_risk_per_trade: özsermayedeki risk oranı (ör: 0.01 = %1)
    - max_daily_loss_pct: günlük max zarar (ör: 0.05 = %5)
    - state_file: günlük risk durumunu sakladığımız json

    Not:
      * Günlük zarar limiti, gün başındaki equity'ye göre hesaplanır.
      * Eğer equity <= 0 ise, günlük zarar limiti ENFORCE edilmez
        (max_daily_loss_abs None kalır, sadece log basılır).
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.01,
        max_daily_loss_pct: float = 0.05,
        state_file: str = "logs/risk_state.json",
    ):
        self.max_risk_per_trade = float(max_risk_per_trade)
        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.state_file = state_file

        # Bugüne ait state
        self._state: DailyRiskState = self._load_state()
        # Gün başı equity (yüzdelik zarar limitini buna göre hesaplıyoruz)
        self._start_of_day_equity: Optional[float] = None

        # 🔥 Model performansına göre risk çarpanı (AUC'den beslenecek)
        # 1.0 = nötr, >1 agresif, <1 defansif
        self.model_confidence_factor: float = 1.0

    # ──────────────── internal helpers ────────────────

    def _today_str(self) -> str:
        # Cloud Run için de yeterli: UTC tarih
        return date.today().isoformat()

    def _load_state(self) -> DailyRiskState:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                system_logger.info(
                    f"[RISK] Loaded state from {self.state_file}: {data}"
                )
                return DailyRiskState(**data)
        except Exception as e:
            system_logger.exception(f"[RISK] Failed to load state: {e}")

        # Dosya yoksa ya da okunamıyorsa bugünün tarihini içeren temiz state ile başla
        return DailyRiskState(trade_date=self._today_str())

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self._state), f, ensure_ascii=False, indent=2)
        except Exception as e:
            system_logger.exception(f"[RISK] Failed to save state: {e}")

    def _reset_if_new_day(self) -> None:
        today = self._today_str()
        if self._state.trade_date != today:
            system_logger.info(
                f"[RISK] New day detected. Resetting risk state "
                f"({self._state.trade_date} -> {today})"
            )
            self._state = DailyRiskState(trade_date=today)
            self._start_of_day_equity = None
            self._save_state()

    # ──────────────── public API ────────────────

    def set_start_of_day_equity(self, equity: float) -> None:
        """
        Günün başındaki özsermaye. Günlük zarar limitini bu değer üzerinden hesaplıyoruz.
        Bunu main loop'ta günde bir kez (veya instance restart olduğunda) çağır.

        ÖNEMLİ:
          - Eğer equity <= 0 ise, günlük zarar limiti ENFORCE edilmez.
          - Böyle bir durumda _start_of_day_equity set edilmez ve
            max_daily_loss_abs property’si None döner.
        """
        self._reset_if_new_day()

        # ÖNEMLİ: Equity 0 veya negatifse günlük limit hesaplamayalım
        if equity is None or equity <= 0:
            system_logger.warning(
                f"[RISK] Attempted to set start-of-day equity <= 0 (equity={equity}). "
                f"Daily loss limit will NOT be enforced until a positive equity is seen."
            )
            return

        if self._start_of_day_equity is None:
            self._start_of_day_equity = float(equity)
            system_logger.info(f"[RISK] Start-of-day equity set to {equity:.2f} USDT")

    @property
    def max_daily_loss_abs(self) -> Optional[float]:
        """
        Günlük max zarar (USDT cinsinden). start_of_day_equity set edilmediyse None döner.
        """
        if self._start_of_day_equity is None:
            return None
        return self._start_of_day_equity * self.max_daily_loss_pct

    @property
    def realized_pnl(self) -> float:
        return self._state.realized_pnl

    @property
    def trading_halted(self) -> bool:
        return self._state.trading_halted

    def can_open_new_trade(self, current_equity: float) -> Tuple[bool, str]:
        """
        Yeni trade açılabilir mi? Günlük zarar limiti ve halt durumuna göre kontrol eder.
        Returns: (allowed: bool, reason: str)
        """
        self._reset_if_new_day()

        if self._state.trading_halted:
            return False, "Trading halted for today (daily loss limit reached)."

        if self._start_of_day_equity is None:
            # İlk trade öncesi equity set et (equity <= 0 ise set_start_of_day_equity
            # içinden sadece uyarı basıp geri dönecek)
            self.set_start_of_day_equity(current_equity)

        max_loss_abs = self.max_daily_loss_abs
        if max_loss_abs is None:
            # Her ihtimale karşı allow et ama logla
            system_logger.warning(
                "[RISK] max_daily_loss_abs is None. Daily loss limit not enforced."
            )
            return True, "OK (daily limit not set yet)"

        # realized_pnl negatif ise zarar. Örn: realized_pnl = -55, max_loss_abs=50 → halt
        if -self._state.realized_pnl >= max_loss_abs:
            self._state.trading_halted = True
            self._save_state()
            system_logger.warning(
                f"[RISK] Daily loss limit hit. realized_pnl={self._state.realized_pnl:.2f}, "
                f"max_daily_loss_abs={max_loss_abs:.2f}. Trading halted for today."
            )
            return False, "Daily loss limit reached."

        return True, "OK"

    def update_model_confidence(self, auc: float) -> None:
        """
        Model performansına göre risk çarpanını ayarla.
        Örnek strateji:
        - AUC >= 0.70 -> 1.5x pozisyon (agresif)
        - 0.65 <= AUC < 0.70 -> 1.2x
        - 0.60 <= AUC < 0.65 -> 1.0x (nötr)
        - AUC < 0.60 -> 0.7x (defansif)
        """
        try:
            auc = float(auc)
        except (TypeError, ValueError):
            system_logger.warning(
                f"[RISK] Invalid AUC value for model_confidence: {auc!r}. "
                "Falling back to 1.0."
            )
            self.model_confidence_factor = 1.0
            return

        if auc >= 0.70:
            self.model_confidence_factor = 1.5
        elif auc >= 0.65:
            self.model_confidence_factor = 1.2
        elif auc >= 0.60:
            self.model_confidence_factor = 1.0
        else:
            self.model_confidence_factor = 0.7

        system_logger.info(
            f"[RISK] update_model_confidence: auc={auc:.4f} -> "
            f"model_confidence_factor={self.model_confidence_factor:.2f}"
        )

    def compute_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        equity: float,
        stop_loss_pct: float,
        leverage: float = 1.0,
    ) -> float:
        """
        Risk'e göre pozisyon büyüklüğü hesabı.

        risk_amount = equity * max_risk_per_trade
        risk_per_unit = entry_price * stop_loss_pct * leverage
        qty = risk_amount / risk_per_unit

        Burada qty daha sonra model_confidence_factor ile ölçeklenir:
        - AUC yüksekse (factor > 1) -> daha büyük pozisyon
        - AUC düşükse (factor < 1) -> daha küçük pozisyon
        """
        risk_amount = equity * self.max_risk_per_trade
        risk_per_unit = entry_price * stop_loss_pct * leverage

        if risk_per_unit <= 0:
            system_logger.warning(
                f"[RISK] Invalid risk_per_unit for {symbol}. entry={entry_price}, "
                f"sl_pct={stop_loss_pct}, lev={leverage}. Using qty=0."
            )
            return 0.0

        # Temel qty
        qty = risk_amount / risk_per_unit
        # 🔥 Model güvenine göre ölçekle
        qty = qty * self.model_confidence_factor

        system_logger.info(
            f"[RISK] Position size computed for {symbol} {side}: "
            f"equity={equity:.2f}, risk_amount={risk_amount:.2f}, "
            f"entry={entry_price:.2f}, sl_pct={stop_loss_pct:.4f}, lev={leverage}, "
            f"model_conf_factor={self.model_confidence_factor:.2f}, qty={qty:.6f}"
        )
        return qty

    def register_closed_trade(self, pnl: float) -> None:
        """
        Pozisyon kapandıktan sonra realized PnL'i risk state'e işler.
        pnl: USDT cinsinden kar/zarar (+/-)
        """
        self._reset_if_new_day()
        self._state.realized_pnl += float(pnl)
        self._save_state()
        system_logger.info(
            f"[RISK] Closed trade registered. pnl={pnl:.2f}, "
            f"daily_realized_pnl={self._state.realized_pnl:.2f}"
        )

    def get_status_summary(self) -> dict:
        """
        Telegram /status veya monitoring için özet.
        """
        self._reset_if_new_day()
        return {
            "trade_date": self._state.trade_date,
            "realized_pnl": self._state.realized_pnl,
            "trading_halted": self._state.trading_halted,
            "max_daily_loss_abs": self.max_daily_loss_abs,
            "model_confidence_factor": self.model_confidence_factor,
        }

