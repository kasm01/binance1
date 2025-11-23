# tg_bot/commands.py
from telegram import Update
from telegram.ext import CallbackContext

from core.logger import system_logger
from monitoring.performance_tracker import PerformanceTracker

# Tek bir global tracker
performance_tracker = PerformanceTracker()


def start_command(update: Update, context: CallbackContext) -> None:
    """
    /start komutu: Botun aktif olduğunu bildirir.
    """
    update.message.reply_text("Binance1-Pro botu aktif! 🚀")
    system_logger.info("Telegram: /start command used")


def status_command(update: Update, context: CallbackContext) -> None:
    """
    /status komutu: Performans özetini gösterir.
    """
    summary = performance_tracker.get_summary()
    msg = (
        "📊 Bot Durumu\n"
        f"Toplam İşlem: {summary['total_trades']}\n"
        f"Başarılı: {summary['successful_trades']}\n"
        f"Başarısız: {summary['failed_trades']}\n"
        f"Toplam PnL: {summary['total_pnl']:.4f}"
    )
    update.message.reply_text(msg)
    system_logger.info("Telegram: /status command used")


def trades_command(update: Update, context: CallbackContext) -> None:
    """
    /trades komutu: Şimdilik placeholder.
    İleride gerçek trade geçmişi buraya bağlanacak.
    """
    update.message.reply_text("Trade geçmişi: Yakında eklenecek özellik 🛠")
    system_logger.info("Telegram: /trades command used")

