from typing import Optional

from core.logger import system_logger, error_logger


class Notifier:
    """
    Sistem + hata bildirimleri.
    Telegram entegrasyonu ileride buraya bağlanacak.
    """

    def __init__(self, telegram_bot: Optional[object] = None) -> None:
        self.telegram_bot = telegram_bot

    def notify_system(self, message: str) -> None:
        system_logger.info(message)
        if self.telegram_bot:
            try:
                self.telegram_bot.send_message(message)
            except Exception as e:
                error_logger.error(f"Telegram system notify failed: {e}")

    def notify_error(self, message: str) -> None:
        error_logger.error(message)
        if self.telegram_bot:
            try:
                self.telegram_bot.send_message(f"ERROR: {message}")
            except Exception as e:
                error_logger.error(f"Telegram error notify failed: {e}")
