import logging
import json
from typing import Optional

from core.risk_manager import RiskManager

trade_logger = logging.getLogger('trade_logger')
trade_logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs/trades.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
trade_logger.addHandler(fh)

def log_trade(trade_data):
    trade_logger.info(trade_data)

def log_risk_snapshot(risk_manager: RiskManager, logger: Optional[logging.Logger] = None) -> None:
    """
    RiskManager state'ini tek satır JSON olarak loglar.

    Örnek kullanım:
        from monitoring.trade_logger import log_risk_snapshot
        log_risk_snapshot(risk_manager)
    """
    lg = logger or logging.getLogger("risk_snapshot")

    snap = {
        "current_day": str(risk_manager.current_day),
        "daily_realized_pnl": float(risk_manager.daily_realized_pnl),
        "consecutive_losses": int(risk_manager.consecutive_losses),
        "open_trades": int(risk_manager.open_trades),
        "daily_max_loss_usdt": float(risk_manager.daily_max_loss_usdt),
        "daily_max_loss_pct": float(risk_manager.daily_max_loss_pct),
        "equity_start_of_day": float(risk_manager.equity_start_of_day),

        # son trade istatistikleri
        "last_realized_pnl": float(risk_manager.last_realized_pnl),
        "total_trades": int(risk_manager.total_trades),
        "total_wins": int(risk_manager.total_wins),
        "total_losses": int(risk_manager.total_losses),

        # son açılan / kapanan trade meta (dict olarak, None olabilir)
        "last_open_trade": risk_manager.last_open_trade,
        "last_close_trade": risk_manager.last_close_trade,
    }

    lg.info("[RISK-SNAPSHOT] %s", json.dumps(snap, ensure_ascii=False))
