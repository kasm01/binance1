# tg_bot/guards.py
import os
import time
from functools import wraps
from typing import Callable, Any

from telegram import Update
from telegram.ext import CallbackContext

def _allowed_chat_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or os.getenv("TELEGRAM_CHAT_ID") or ""
    ids = set()
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            ids.add(int(x))
        except Exception:
            pass
    return ids

def require_auth(fn: Callable[[Update, CallbackContext], Any]):
    @wraps(fn)
    def wrapper(update: Update, context: CallbackContext):
        allowed = _allowed_chat_ids()
        chat_id = update.effective_chat.id if update.effective_chat else None
        if allowed and (chat_id is None or int(chat_id) not in allowed):
            # Sessiz reddetmek yerine kısa cevap
            if update.message:
                update.message.reply_text("⛔ Bu bot için yetkin yok.")
            return
        return fn(update, context)
    return wrapper

def rate_limit(min_seconds: float = 1.0):
    """
    Basit per-chat rate limit. context.chat_data içinde timestamp tutar.
    """
    def deco(fn: Callable[[Update, CallbackContext], Any]):
        @wraps(fn)
        def wrapper(update: Update, context: CallbackContext):
            now = time.time()
            last = context.chat_data.get("_last_cmd_ts", 0.0)
            if now - float(last) < float(min_seconds):
                return
            context.chat_data["_last_cmd_ts"] = now
            return fn(update, context)
        return wrapper
    return deco
