# tg_bot/telegram_bot.py
from typing import Optional

from telegram import Bot
from telegram.ext import Updater, CommandHandler, CallbackContext

from core.logger import system_logger, error_logger
from config.credentials import Credentials
from .commands import start_command, status_command, trades_command


class TelegramBot:
    """
    Basit Telegram bot wrapper'ı.
    - python-telegram-bot (v13 civarı) Updater/Dispatcher API'sini kullanır.
    - Notifier ile uyumlu olacak şekilde send_message(message: str) metoduna sahiptir.
    """

    def __init__(self) -> None:
        # Credentials içinden token ve varsayılan chat_id çekiyoruz
        self.token: Optional[str] = getattr(Credentials, "TELEGRAM_TOKEN", None)
        self.default_chat_id: Optional[str] = getattr(
            Credentials, "TELEGRAM_CHAT_ID", None
        )

        if not self.token:
            system_logger.warning(
                "[TelegramBot] TELEGRAM_TOKEN bulunamadı. Telegram bot devre dışı."
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

    # ------------------------------------------------------------------ #
    # Komut kayıtları
    # ------------------------------------------------------------------ #
    def register_commands(self) -> None:
        """
        /start, /status, /trades komutlarını dispatcher'a kayıt eder.
        """
        if not self.dispatcher:
            return

        self.dispatcher.add_handler(CommandHandler("start", start_command))
        self.dispatcher.add_handler(CommandHandler("status", status_command))
        self.dispatcher.add_handler(CommandHandler("trades", trades_command))

        system_logger.info("[TelegramBot] Telegram komutları dispatcher'a eklendi.")

    # ------------------------------------------------------------------ #
    # Polling başlatma (local kullanım için)
    # Cloud Run içinde genelde kullanmayacağız.
    # ------------------------------------------------------------------ #
    def start_polling(self) -> None:
        """
        Local ortamda botu çalıştırmak için (Cloud Run yerine kendi makinen).
        """
        if not self.updater:
            system_logger.warning(
                "[TelegramBot] Updater yok, muhtemelen TELEGRAM_TOKEN ayarlı değil."
            )
            return

        system_logger.info("[TelegramBot] Telegram polling başlatılıyor...")
        self.updater.start_polling()
        self.updater.idle()

    # ------------------------------------------------------------------ #
    # Notifier ile uyumlu mesaj gönderme
    # ------------------------------------------------------------------ #
    def send_message(self, message: str) -> None:
        """
        Notifier tarafından çağrılacak basit mesaj gönderme fonksiyonu.
        TELEGRAM_CHAT_ID tanımlı değilse log'a hata basar.
        """
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

