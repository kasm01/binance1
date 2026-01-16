import os
import re
from typing import Optional

from telegram import Bot
from telegram.parsemode import ParseMode
from telegram.ext import Updater
from telegram.error import BadRequest

from core.logger import system_logger, error_logger
from core.risk_manager import RiskManager
from config.credentials import Credentials
from tg_bot.commands import register_handlers


class TelegramBot:
    """
    Basit Telegram bot wrapper'ı.
    - /start, /status, /trades, /risk gibi komutlar tg_bot.commands.register_handlers ile eklenir.
    """


    def _safe_send(self, chat_id: str, text: str, **kwargs):
        """Send message safely. If Telegram Markdown/HTML parse fails, retry as plain text."""
        try:
            return self._safe_send(chat_id=chat_id, text=text, **kwargs)
        except BadRequest as e:
            # Parse hatası vb. olursa parse_mode'u düşürüp plain text tekrar dene
            msg = str(e)
            if "Can't parse entities" in msg or "parse" in msg.lower():
                kwargs.pop("parse_mode", None)
                try:
                    return self._safe_send(chat_id=chat_id, text=text, **kwargs)
                except Exception:
                    raise
            raise

    def __init__(self) -> None:
        self.token: Optional[str] = getattr(Credentials, "TELEGRAM_BOT_TOKEN", None)
        self.default_chat_id: Optional[str] = getattr(Credentials, "TELEGRAM_CHAT_ID", None)
        if not self.default_chat_id:
            _allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or ""
            _parts = [x.strip() for x in re.split(r"[,\s]+", _allowed) if x.strip()]
            if _parts:
                self.default_chat_id = _parts[0]

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


        # default_chat_id fallback (no .env chat_id; use allowed list)

        try:

            if not getattr(self, "default_chat_id", None):

                _allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or ""

                _parts = [x.strip() for x in re.split(r"[,\s]+", _allowed) if x.strip()]

                if _parts:

                    self.default_chat_id = _parts[0]

        except Exception:

            pass

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
        """Send message safely.
        Markdown/HTML parse hatasında plain-text retry yapar.
        """
        if not self.default_chat_id:
            error_logger.error("[TelegramBot] TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilemiyor.")
            return

        try:
            # İlk deneme: parse_mode varsa kullan
            self.bot.send_message(
                chat_id=self.default_chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return

        except BadRequest as e:
            msg = str(e)
            # Markdown / entity parse hatası → plain text retry
            if "Can't parse entities" in msg or "parse" in msg.lower():
                try:
                    self.bot.send_message(
                        chat_id=self.default_chat_id,
                        text=message,
                    )
                    return
                except Exception as e2:
                    error_logger.error(f"[TelegramBot] Plain-text retry başarısız: {e2}")
                    return

            error_logger.error(f"[TelegramBot] Mesaj gönderilemedi: {e}")

        except Exception as e:
            error_logger.error(f"[TelegramBot] Mesaj gönderilemedi: {e}")
