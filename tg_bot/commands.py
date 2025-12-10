# tg_bot/commands.py

from telegram import Update
from telegram.ext import CallbackContext, CommandHandler
from telegram.parsemode import ParseMode

from core.logger import system_logger
from core.risk_manager import RiskManager
from monitoring.performance_tracker import PerformanceTracker
from tg_bot.message_formatter import format_risk_status

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


def cmd_risk(update: Update, context: CallbackContext) -> None:
    """
    /risk → RiskManager state'ini gösterir.
    RiskManager instance'ı context.bot_data['risk_manager'] içinde bekliyoruz.
    """
    rm: RiskManager = context.bot_data.get("risk_manager")  # type: ignore

    if rm is None:
        update.message.reply_text("RiskManager henüz init edilmemiş.")
        system_logger.warning("Telegram: /risk command used but RiskManager is None")
        return

    text = format_risk_status(rm)
    update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /risk command used")


def register_handlers(dispatcher) -> None:
    """
    Telegram dispatcher için tüm komut handler'larını register eder.
    telegram_bot.py içinde örnek kullanım:
        from tg_bot.commands import register_handlers
        register_handlers(dispatcher)
    """
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("status", status_command))
    dispatcher.add_handler(CommandHandler("trades", trades_command))
    dispatcher.add_handler(CommandHandler("risk", cmd_risk))
