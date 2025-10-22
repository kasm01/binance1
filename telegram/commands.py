from telegram import Update
from telegram.ext import CallbackContext
from core.logger import system_logger
from monitoring.performance_tracker import PerformanceTracker

performance_tracker = PerformanceTracker()

def start_command(update: Update, context: CallbackContext):
    update.message.reply_text("Binance1-Pro botu aktif! 🚀")
    system_logger.info("Telegram: /start command used")

def status_command(update: Update, context: CallbackContext):
    summary = performance_tracker.get_summary()
    msg = f"Total Trades: {summary['total_trades']}\nSuccessful: {summary['successful_trades']}\nFailed: {summary['failed_trades']}\nPnL: {summary['total_pnl']}"
    update.message.reply_text(msg)
    system_logger.info("Telegram: /status command used")

def trades_command(update: Update, context: CallbackContext):
    update.message.reply_text("Trade geçmişi: Yakında eklenecek özellik")
    system_logger.info("Telegram: /trades command used")
