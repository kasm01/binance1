# tg_bot/telegram_bot.py

from typing import Optional

from telegram import Bot
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

        # Updater/Dispatcher komutlar için (python-telegram-bot v13)
        self.updater: Updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher

        # Tüm komutları (start/status/trades/risk) kaydet
        register_handlers(self.dispatcher)

        system_logger.info("[TelegramBot] Telegram bot başarıyla initialize edildi.")

    # ------------------------------------------------------------------
    # RiskManager enjeksiyonu (/risk komutu için)
    # ------------------------------------------------------------------
    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        """
        main.py içinde RiskManager instance'ını Telegram bot'a takmak için:
            tg_bot = TelegramBot()
            tg_bot.set_risk_manager(risk_manager)
        Böylece /risk komutu context.bot_data['risk_manager'] üzerinden erişebilir.
        """
        if not self.dispatcher:
            system_logger.warning(
                "[TelegramBot] set_risk_manager çağrıldı ama dispatcher yok."
            )
            return

        self.dispatcher.bot_data["risk_manager"] = risk_manager  # type: ignore
        system_logger.info(
            "[TelegramBot] RiskManager instance dispatcher.bot_data['risk_manager'] içine set edildi."
        )

    # ------------------------------------------------------------------
    # Polling başlatma
    # ------------------------------------------------------------------
    def start_polling(self) -> None:
        if not self.updater:
            system_logger.warning(
                "[TelegramBot] Updater yok, muhtemelen TELEGRAM_BOT_TOKEN ayarlı değil."
            )
            return

        system_logger.info("[TelegramBot] Telegram polling başlatılıyor...")

        # Thread içinde signal kurmasın diye stop_signals boş veriyoruz
        try:
            self.updater.start_polling(stop_signals=[])
        except TypeError:
            # Eski sürümlerde stop_signals paramı yoksa fallback
            self.updater.start_polling()

        # !!! ÖNEMLİ: Thread içinde idle() çağırma (signal hatası verir)
        # self.updater.idle()
    # ------------------------------------------------------------------
    # Basit mesaj gönderme helper'ı
    # ------------------------------------------------------------------
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
