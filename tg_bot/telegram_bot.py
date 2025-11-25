from typing import Optional

from telegram import Bot
from telegram.ext import Updater, CommandHandler

from core.logger import system_logger, error_logger
from config.credentials import Credentials
from tg_bot.commands import start_command, status_command, trades_command


class TelegramBot:
    """
    Basit Telegram bot wrapper'ı.
    """

    def __init__(self) -> None:
        # Credentials içinden token ve varsayılan chat_id çekiyoruz
        self.token: Optional[str] = getattr(Credentials, "TELEGRAM_BOT_TOKEN", None)
        self.default_chat_id: Optional[str] = getattr(
            Credentials, "TELEGRAM_CHAT_ID", None
        )

        if not self.token:
            system_logger.warning(
                "[TelegramBot] TELEGRAM_BOT_TOKEN bulunamadı. Telegram bot devre dışı."
            )
            self.bot = None
            self.updater = None
            self.dispatcher = None
            return

        # Bot instance
        self.bot: Bot = Bot(token=self.token)

        # Updater/Dispatcher komutlar için (local kullanım için ideal)
        self.updater: Updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher

        self.register_commands()

        system_logger.info("[TelegramBot] Telegram bot başarıyla initialize edildi.")

    def register_commands(self) -> None:
        if not self.dispatcher:
            return

        self.dispatcher.add_handler(CommandHandler("start", start_command))
        self.dispatcher.add_handler(CommandHandler("status", status_command))
        self.dispatcher.add_handler(CommandHandler("trades", trades_command))

        system_logger.info("[TelegramBot] Telegram komutları dispatcher'a eklendi.")

    def start_polling(self) -> None:
        if not self.updater:
            system_logger.warning(
                "[TelegramBot] Updater yok, muhtemelen TELEGRAM_BOT_TOKEN ayarlı değil."
            )
            return

        system_logger.info("[TelegramBot] Telegram polling başlatılıyor...")
        self.updater.start_polling()
        self.updater.idle()

    def send_message(self, message: str) -> None:
        if not self.bot:
            error_logger.error(
                "[TelegramBot] send_message çağrıldı ama bot initialize edilmemiş."
            )
            return

        if not self.default_chat_id:
            error_logger.error(
                "[TelegramBot] TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilemiyor."
            )
            return

        try:
            self.bot.send_message(chat_id=self.default_chat_id, text=message)
            system_logger.info("[TelegramBot] Mesaj gönderildi.")
        except Exception as e:
            error_logger.error(f"[TelegramBot] Mesaj gönderilemedi: {e}")

