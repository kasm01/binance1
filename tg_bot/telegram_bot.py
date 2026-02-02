import os
import re
import logging
import traceback
from typing import Optional

from telegram import Bot
from telegram.error import BadRequest
from telegram.ext import Updater

from core.logger import system_logger, error_logger
from core.risk_manager import RiskManager
from tg_bot.commands import register_handlers


class TelegramBot:
    """
    Basit Telegram bot wrapper'ı.
    - /start, /status, /trades, /risk gibi komutlar tg_bot.commands.register_handlers ile eklenir.
    """

    def __init__(self) -> None:
        # ENV'den oku (Credentials import-time cache sorununu engeller)
        self.token: Optional[str] = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or None
        self.default_chat_id: Optional[str] = (os.getenv("TELEGRAM_CHAT_ID") or "").strip() or None

        if not self.default_chat_id:
            allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or ""
            parts = [x.strip() for x in re.split(r"[,\s]+", allowed) if x.strip()]
            if parts:
                self.default_chat_id = parts[0]

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
        system_logger.info(
            "[TelegramBot] chat_id=%s allowed=%s",
            "SET" if self.default_chat_id else "MISSING",
            "SET" if (os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or "").strip() else "MISSING",
        )

    def _safe_send(self, chat_id: str, text: str, *, parse_mode: Optional[str] = None) -> None:
        """Parse hatasında plain-text retry yapan güvenli gönderim."""
        if not getattr(self, "bot", None):
            return

        try:
            self.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
            return
        except BadRequest as e:
            msg = str(e)
            # Markdown/HTML entity parse hatası → plain text retry
            if "Can't parse entities" in msg or "parse" in msg.lower():
                try:
                    self.bot.send_message(chat_id=chat_id, text=text)
                    return
                except Exception as e2:
                    error_logger.error("[TelegramBot] Plain-text retry başarısız: %s", e2)
                    return
            error_logger.error("[TelegramBot] Mesaj gönderilemedi (BadRequest): %s", e)
        except Exception as e:
            error_logger.error("[TelegramBot] Mesaj gönderilemedi: %s", e)

    def set_risk_manager(self, risk_manager: RiskManager) -> None:
        if not getattr(self, "dispatcher", None):
            system_logger.warning("[TelegramBot] set_risk_manager çağrıldı ama dispatcher yok.")
            return
        self.dispatcher.bot_data["risk_manager"] = risk_manager  # type: ignore
        system_logger.info("[TelegramBot] RiskManager instance dispatcher.bot_data['risk_manager'] içine set edildi.")

    def start_polling(self) -> None:
        if not getattr(self, "updater", None):
            system_logger.warning("[TelegramBot] Updater yok, muhtemelen TELEGRAM_BOT_TOKEN ayarlı değil.")
            return

        system_logger.info("[TelegramBot] Telegram polling başlatılıyor...")
        try:
            self.updater.start_polling(stop_signals=[])
        except TypeError:
            # python-telegram-bot sürümü farklıysa
            self.updater.start_polling()

    def stop_polling(self) -> None:
        if not getattr(self, "updater", None):
            return
        try:
            self.updater.stop()
            system_logger.info("[TelegramBot] Telegram polling durduruldu.")
        except Exception as e:
            system_logger.warning("[TelegramBot] stop_polling hata: %s", e)

    def send_message(self, message: str, *, parse_mode: Optional[str] = None) -> None:
        """
        Varsayılan: plain text.
        İstersen parse_mode="Markdown" vererek çağırabilirsin.
        Parse hatasında otomatik plain text retry yapılır.
        """
        # Debug: mesajı kimin gönderdiğini stack trace ile logla
        try:
            if os.getenv("TG_DEBUG_CALLER", "0").strip().lower() in ("1", "true", "yes", "on"):
                tb = "".join(traceback.format_stack(limit=12))
                logging.getLogger("system").info("[TG][CALLER]\n%s", tb)
        except Exception:
            pass

        if not getattr(self, "bot", None):
            return

        if not self.default_chat_id:
            error_logger.error(
                "[TelegramBot] TELEGRAM_CHAT_ID/TELEGRAM_ALLOWED_CHAT_IDS tanımlı değil, mesaj gönderilemiyor."
            )
            return

        self._safe_send(
            chat_id=str(self.default_chat_id),
            text=str(message),
            parse_mode=parse_mode,
        )

