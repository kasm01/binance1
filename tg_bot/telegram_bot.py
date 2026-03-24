# -*- coding: utf-8 -*-
import asyncio
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
        self.token = ""
        self.default_chat_id = ""

        self.enabled = False
        self.polling_enabled = _env_bool("TELEGRAM_POLLING_ENABLED", False)

        self.bot: Optional[Bot] = None
        self.dispatcher = None
        self.risk_manager = None

        self._reload_env()
        self._ensure_bot(log_prefix="[TelegramBot] init")

    def _reload_env(self) -> None:
        self.token = str(os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
        self.default_chat_id = str(os.getenv("TELEGRAM_CHAT_ID", "") or "").strip()
        self.enabled = bool(self.token and self.default_chat_id)
        self.polling_enabled = _env_bool("TELEGRAM_POLLING_ENABLED", False)

    def _ensure_bot(self, log_prefix: str = "[TelegramBot] ensure_bot") -> bool:
        self._reload_env()

        if not self.enabled:
            self.bot = None
            try:
                system_logger.warning(
                    "%s skipped: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing",
                    log_prefix,
                )
            except Exception:
                pass
            return False

        if self.bot is not None:
            return True

        try:
            self.bot = Bot(token=self.token)
            system_logger.info(
                "%s ok | chat_id=%s polling_enabled=%s",
                log_prefix,
                self.default_chat_id,
                self.polling_enabled,
            )
            return True
        except Exception:
            self.bot = None
            try:
                error_logger.exception("%s failed", log_prefix)
            except Exception:
                pass
            return False

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
        if not self._ensure_bot(log_prefix="[TelegramBot] start_polling"):
            system_logger.warning("[TelegramBot] start_polling skipped: bot not ready")
            return

        if not self.polling_enabled:
            system_logger.info("[TelegramBot] polling disabled -> send-only mode")
            return

        system_logger.warning(
            "[TelegramBot] polling requested but implementation is send-only; skipping"
        )

    def send_message(self, message: str, *, parse_mode: Optional[str] = None) -> None:
        text = str(message or "").strip()
        if not text:
            return

        if not self._ensure_bot(log_prefix="[TelegramBot] send_message ensure_bot"):
            error_logger.error("[TelegramBot] send_message skipped: bot not ready")
            return

        try:
            kwargs = {
                "chat_id": self.default_chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                kwargs["parse_mode"] = parse_mode

            result = self.bot.send_message(**kwargs)

            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    asyncio.run(result)

            try:
                system_logger.info(
                    "[TelegramBot] send_message queued | chat_id=%s len=%s",
                    self.default_chat_id,
                    len(text),
                )
            except Exception:
                pass

        except Exception:
            try:
                error_logger.exception("[TelegramBot] send_message failed")
            except Exception:
                pass
