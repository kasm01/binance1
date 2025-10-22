from core.logger import system_logger

class PerformanceTracker:
    def __init__(self):
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.pnl = 0

    def record_trade(self, pnl, success=True):
        self.total_trades += 1
        if success:
            self.successful_trades += 1
        else:
            self.failed_trades += 1
        self.pnl += pnl
        system_logger.info(f"Trade recorded: PnL={pnl}, Total PnL={self.pnl}")

    def get_summary(self):
        return {
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "total_pnl": self.pnl
        }
