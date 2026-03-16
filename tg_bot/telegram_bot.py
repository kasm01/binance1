# -*- coding: utf-8 -*-
import os
from typing import Optional

from telegram import Bot

from core.logger import system_logger, error_logger


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class TelegramBot:
    def __init__(self) -> None:
        self.token = str(os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
        self.default_chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "") or "").strip()

        self.enabled = bool(self.token and self.default_chat_id)
        self.polling_enabled = _env_bool("TELEGRAM_POLLING_ENABLED", False)

        self.bot: Optional[Bot] = None
        self.dispatcher = None
        self.risk_manager = None

        if not self.enabled:
            system_logger.warning(
                "[TelegramBot] disabled: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing"
            )
            return

        try:
            self.bot = Bot(token=self.token)
            system_logger.info(
                "[TelegramBot] init ok | chat_id=%s polling_enabled=%s",
                self.default_chat_id,
                self.polling_enabled,
            )
        except Exception as e:
            self.bot = None
            error_logger.exception("[TelegramBot] init failed")
    def set_risk_manager(self, risk_manager) -> None:
        self.risk_manager = risk_manager
        try:
            system_logger.info("[TelegramBot] risk_manager attached")
        except Exception:
            pass

    def start_polling(self) -> None:
        """
        Send-only mode.
        Polling bilinçli olarak kapalı bırakılıyor.
        """
        if not self.enabled or self.bot is None:
            system_logger.warning("[TelegramBot] start_polling skipped: bot not ready")
            return

        if not self.polling_enabled:
            system_logger.info("[TelegramBot] polling disabled -> send-only mode")
            return

        system_logger.warning(
            "[TelegramBot] polling requested but implementation is send-only; skipping"
        )
        return

    def send_message(self, message: str, *, parse_mode: Optional[str] = None) -> None:
        if not self.enabled or self.bot is None:
            error_logger.error("[TelegramBot] send_message called but bot not initialized")
            return

        text = str(message or "").strip()
        if not text:
            return

        try:
            kwargs = {
                "chat_id": self.default_chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                kwargs["parse_mode"] = parse_mode

            self.bot.send_message(**kwargs)
        except Exception:
            try:
                error_logger.exception("[TelegramBot] send_message failed")
            except Exception:
                pass
