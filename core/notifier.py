from typing import Optional
import threading

from core.logger import system_logger, error_logger


class Notifier:
    """
    Sistem + hata bildirimleri.
    Telegram entegrasyonu safe-thread ile çalışır.
    Event loop closed / timeout hatasını ana bot akışına taşımaz.
    """

    def __init__(self, telegram_bot: Optional[object] = None) -> None:
        self.telegram_bot = telegram_bot

    def _safe_telegram_send(self, message: str) -> None:
        bot = self.telegram_bot
        if not bot:
            return

        def _worker() -> None:
            try:
                bot.send_message(message)
            except Exception as e:
                try:
                    error_logger.warning(f"Telegram notify skipped: {e}")
                except Exception:
                    pass

        try:
            threading.Thread(target=_worker, daemon=True).start()
        except Exception as e:
            try:
                error_logger.warning(f"Telegram thread start failed: {e}")
            except Exception:
                pass

    def notify_system(self, message: str) -> None:
        system_logger.info(message)
        self._safe_telegram_send(message)

    def notify_error(self, message: str) -> None:
        error_logger.error(message)
        self._safe_telegram_send(f"ERROR: {message}")
