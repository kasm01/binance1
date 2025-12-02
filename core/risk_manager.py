# core/risk_manager.py

import json
import os
from datetime import datetime


class RiskManager:
    """
    Basit ama genişletilebilir risk yönetimi sistemi.
    - Günlük kayıp limiti (max_daily_loss)
    - Günlük trade limiti
    - Model güven katsayısı (model_confidence_factor)
    """

    def __init__(
        self,
        state_file: str = "logs/risk_state.json",
        max_daily_loss: float = -500.0,     # Örn: -50 USDT
        max_daily_trades: int = 50,        # Günlük limit
    ):
        self.state_file = state_file
        self.max_daily_loss = max_daily_loss
        self.max_daily_trades = max_daily_trades

        # Bugün kaç trade oldu, günlük PnL
        self.daily_pnl = 0.0
        self.daily_trades = 0

        # Model güvenirlik katsayısı
        # 1.0 = nötr
        self.model_confidence_factor: float = 1.0

        self.current_date = datetime.utcnow().date()

        self.load_state()

    # -------------------------------------------------------------
    # STATE LOAD / SAVE
    # -------------------------------------------------------------
    def load_state(self):
        """Risk durumunu JSON'dan yükle."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                saved_date = datetime.strptime(data["date"], "%Y-%m-%d").date()

                # Gün değiştiyse resetle
                if saved_date != self.current_date:
                    self.reset_daily_state()
                else:
                    self.daily_pnl = data.get("daily_pnl", 0.0)
                    self.daily_trades = data.get("daily_trades", 0)

        except Exception:
            self.reset_daily_state()

    def save_state(self):
        """Risk durumunu JSON'a yaz."""
        data = {
            "date": self.current_date.strftime("%Y-%m-%d"),
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=4)

    # -------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------
    def reset_daily_state(self):
        """Yeni gün başladığında değerleri sıfırla."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.save_state()

    # -------------------------------------------------------------
    # MODEL CONFIDENCE — (Senin eklemek istediğin mekanizma)
    # -------------------------------------------------------------
    def update_model_confidence(self, auc: float) -> None:
        """
        Model performansına göre risk çarpanını ayarla.
        - AUC >= 0.70 → agresif (1.5x)
        - 0.65–0.70 → yarı agresif (1.2x)
        - 0.60–0.65 → nötr (1.0x)
        - AUC < 0.60 → defansif (0.7x)
        """
        if auc >= 0.70:
            self.model_confidence_factor = 1.5
        elif auc >= 0.65:
            self.model_confidence_factor = 1.2
        elif auc >= 0.60:
            self.model_confidence_factor = 1.0
        else:
            self.model_confidence_factor = 0.7

    # -------------------------------------------------------------
    # RISK CHECKS
    # -------------------------------------------------------------
    def allow_trade(self) -> bool:
        """Günlük limitlere göre ticarete izin verilsin mi?"""
        if self.daily_pnl <= self.max_daily_loss:
            return False
        if self.daily_trades >= self.max_daily_trades:
            return False
        return True

    # -------------------------------------------------------------
    # TRADE LOGGING
    # -------------------------------------------------------------
    def register_trade(self, pnl: float):
        """İşlem yapıldığında PnL ve trade sayısını güncelle."""
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.save_state()

    # -------------------------------------------------------------
    # POSITION SIZE MULTIPLIER
    # -------------------------------------------------------------
    def get_position_multiplier(self) -> float:
        """
        Pozisyon boyutunu model güvenine göre artırır/azaltır.
        Örn:
        base_size * model_confidence_factor
        """
        return self.model_confidence_factor
