from typing import Optional

from telegram import Bot
from telegram.parsemode import ParseMode
from telegram.ext import Updater

from core.logger import system_logger, error_logger
from core.risk_manager import RiskManager
from config.credentials import Credentials
from tg_bot.commands import register_handlers


class TelegramBot:
    """
    Basit Telegram bot wrapper'ı.
    - /start, /status, /trades, /risk gibi komutlar tg_bot.commands.register_handlers ile eklenir.
    """

    def __init__(self) -> None:
        self.token: Optional[str] = getattr(Credentials, "TELEGRAM_BOT_TOKEN", None)
        self.default_chat_id: Optional[str] = getattr(Credentials, "TELEGRAM_CHAT_ID", None)

        if not self.token:
            system_logger.warning("[TelegramBot] TELEGRAM_BOT_TOKEN bulunamadı. Telegram bot devre dışı.")
            self.bot = None
            self.updater = None
            self.dispatcher = None
            return

        self.bot: Bot = Bot(token=self.token)

        self.updater: Updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher

        register_handlers(self.dispatcher)

        system_logger.info("[TelegramBot] Telegram bot başarıyla initialize edildi.")

    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        if not self.dispatcher:
            system_logger.warning("[TelegramBot] set_risk_manager çağrıldı ama dispatcher yok.")
            return

        self.dispatcher.bot_data["risk_manager"] = risk_manager  # type: ignore
        system_logger.info("[TelegramBot] RiskManager instance dispatcher.bot_data['risk_manager'] içine set edildi.")

    def start_polling(self) -> None:
        if not self.updater:
            system_logger.warning("[TelegramBot] Updater yok, muhtemelen TELEGRAM_BOT_TOKEN ayarlı değil.")
            return

        system_logger.info("[TelegramBot] Telegram polling başlatılıyor...")

        try:
            self.updater.start_polling(stop_signals=[])
        except TypeError:
            self.updater.start_polling()

        # Thread içinde idle() çağırma.

    def stop_polling(self) -> None:
        """
        Graceful stop: polling thread’i dursa bile safe.
        """
        if not self.updater:
            return
        try:
            self.updater.stop()
            system_logger.info("[TelegramBot] Telegram polling durduruldu.")
        except Exception as e:
            system_logger.warning("[TelegramBot] stop_polling hata: %s", e)

    def send_message(self, message: str, *, parse_mode: Optional[str] = None) -> None:
        if not self.bot:
            error_logger.error("[TelegramBot] send_message çağrıldı ama bot initialize edilmemiş.")
            return

        if not self.default_chat_id:
            error_logger.error("[TelegramBot] TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilemiyor.")
            return

        try:
            pm = parse_mode or ParseMode.MARKDOWN
            self.bot.send_message(chat_id=self.default_chat_id, text=message, parse_mode=pm)
            system_logger.info("[TelegramBot] Mesaj gönderildi.")
        except Exception as e:
            error_logger.error(f"[TelegramBot] Mesaj gönderilemedi: {e}")
