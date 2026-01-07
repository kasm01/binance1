# tg_bot/commands.py

from __future__ import annotations

import os
import time
from functools import wraps
from typing import Any, Dict, Optional, Set

from telegram import Update
from telegram.ext import CallbackContext, CommandHandler
from telegram.parsemode import ParseMode

from core.logger import system_logger
from core.risk_manager import RiskManager
from monitoring.performance_tracker import PerformanceTracker
from tg_bot.message_formatter import format_risk_status

# Tek bir global tracker
performance_tracker = PerformanceTracker()


# =========================================================
# Guards: Auth + Rate Limit
# =========================================================
def _allowed_chat_ids() -> Set[int]:
    """
    Öncelik:
      1) TELEGRAM_ALLOWED_CHAT_IDS="id1,id2"
      2) TELEGRAM_CHAT_ID="id"
    """
    raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or os.getenv("TELEGRAM_CHAT_ID") or ""
    ids: Set[int] = set()
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            ids.add(int(x))
        except Exception:
            pass
    return ids


def require_auth(fn):
    @wraps(fn)
    def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
        allowed = _allowed_chat_ids()
        chat_id = update.effective_chat.id if update.effective_chat else None
        if allowed and (chat_id is None or int(chat_id) not in allowed):
            if update.message:
                update.message.reply_text("⛔ Bu bot için yetkin yok.")
            return None
        return fn(update, context, *args, **kwargs)

    return wrapper


def rate_limit(min_seconds: float = 0.8):
    """
    Basit per-chat rate limit. context.chat_data içinde ts tutar.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
            now = time.time()
            last = float(context.chat_data.get("_last_cmd_ts", 0.0))
            if now - last < float(min_seconds):
                return None
            context.chat_data["_last_cmd_ts"] = now
            return fn(update, context, *args, **kwargs)

        return wrapper

    return deco


# =========================================================
# Helpers: Status formatting
# =========================================================
def _fmt_status_from_snapshot(snapshot: Dict[str, Any]) -> str:
    """
    main loop snapshot örneği (opsiyonel):
      context.bot_data["status_snapshot"] = {
          "symbol": "BTCUSDT",
          "signal": "HOLD",
          "ensemble_p": 0.4038,
          "intervals": ["1m","3m","5m","15m","30m","1h"],
          "aucs": {"1m":0.6213, "3m":0.5160, ...},
          "last_price": 42350.1,
          "why": "mtf ensemble",
          "ts": "2026-01-05T12:48:55"
      }
    """
    symbol = snapshot.get("symbol", "N/A")
    signal = str(snapshot.get("signal", "N/A")).upper()
    ens_p = snapshot.get("ensemble_p", None)
    intervals = snapshot.get("intervals", []) or []
    aucs = snapshot.get("aucs", {}) or {}

    emoji = {"BUY": "✅", "SELL": "🟣", "HOLD": "⏸"}.get(signal, "❔")

    lines = []
    lines.append(f"{emoji} *{symbol}*")
    lines.append(f"• *Signal:* `{signal}`")

    if ens_p is not None:
        try:
            lines.append(f"• *Ensemble p:* `{float(ens_p):.4f}`")
        except Exception:
            lines.append(f"• *Ensemble p:* `{ens_p}`")

    last_price = snapshot.get("last_price", None)
    if last_price is not None:
        try:
            lines.append(f"• *Price:* `{float(last_price):.4f}`")
        except Exception:
            lines.append(f"• *Price:* `{last_price}`")

    if intervals:
        lines.append(f"• *MTF:* `{','.join([str(x) for x in intervals])}`")

    if isinstance(aucs, dict) and aucs:
        # kısa göster (ilk 6)
        keys = list(aucs.keys())[:6]
        pairs = []
        for k in keys:
            v = aucs.get(k)
            if isinstance(v, (int, float)):
                pairs.append(f"{k}:{v:.3f}")
            else:
                pairs.append(f"{k}:{v}")
        lines.append(f"• *AUC:* `{', '.join(pairs)}`")

    why = snapshot.get("why", None)
    if why:
        lines.append(f"• *Why:* `{why}`")

    ts = snapshot.get("ts", None)
    if ts:
        lines.append(f"• *TS:* `{ts}`")

    return "\n".join(lines)


def _fmt_status_from_perf() -> str:
    summary = performance_tracker.get_summary()
    return (
        "📊 *Bot Durumu*\n"
        f"• *Toplam İşlem:* `{summary.get('total_trades', 0)}`\n"
        f"• *Başarılı:* `{summary.get('successful_trades', 0)}`\n"
        f"• *Başarısız:* `{summary.get('failed_trades', 0)}`\n"
        f"• *Toplam PnL:* `{float(summary.get('total_pnl', 0.0)):.4f}`"
    )


# =========================================================
# Commands
# =========================================================
@require_auth
@rate_limit(0.8)
def start_command(update: Update, context: CallbackContext) -> None:
    """/start: Botun aktif olduğunu bildirir."""
    msg = (
        "Binance1-Pro botu aktif! 🚀\n\n"
        "Komutlar için: /help"
    )
    update.message.reply_text(msg)
    system_logger.info("Telegram: /start command used")


@require_auth
@rate_limit(0.8)
def help_command(update: Update, context: CallbackContext) -> None:
    msg = (
        "📌 *Komutlar*\n"
        "• /start - bot tanıtım\n"
        "• /status - durum özeti (snapshot varsa zengin)\n"
        "• /trades - trade geçmişi (placeholder)\n"
        "• /risk - risk durumu\n"
        "• /ping - canlılık\n"
        "• /whoami - chat bilgisi\n"
        "• /help - bu menü\n"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


@require_auth
@rate_limit(0.8)
def ping_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("🏓 pong")


@require_auth
@rate_limit(0.8)
def whoami_command(update: Update, context: CallbackContext) -> None:
    u = update.effective_user
    c = update.effective_chat
    msg = (
        "👤 *WhoAmI*\n"
        f"• *user:* `{u.username if u else None}`\n"
        f"• *name:* `{u.full_name if u else None}`\n"
        f"• *chat_id:* `{c.id if c else None}`\n"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


@require_auth
@rate_limit(0.8)
def status_command(update: Update, context: CallbackContext) -> None:
    """
    /status komutu: Anlık bot snapshot + performans özeti.
    Snapshot main loop içinde dispatcher.bot_data['status_snapshot'] olarak set ediliyor.
    """
    snap = None
    try:
        snap = context.bot_data.get("status_snapshot")  # type: ignore
    except Exception:
        snap = None

    lines = ["📊 *Bot Durumu*"]

    # 1) Snapshot (varsa)
    if isinstance(snap, dict) and snap:
        try:
            symbol = snap.get("symbol", "N/A")
            signal = snap.get("signal", "N/A")
            p_used = snap.get("ensemble_p", None)
            last_price = snap.get("last_price", None)
            why = snap.get("why", "")

            lines.append(f"• *Symbol:* `{symbol}`")
            lines.append(f"• *Signal:* `{signal}`")

            if p_used is not None:
                try:
                    lines.append(f"• *p_used:* `{float(p_used):.4f}`")
                except Exception:
                    lines.append(f"• *p_used:* `{p_used}`")

            if last_price is not None:
                try:
                    lines.append(f"• *Price:* `{float(last_price):.4f}`")
                except Exception:
                    lines.append(f"• *Price:* `{last_price}`")

            if why:
                lines.append(f"• *Source:* `{why}`")

            itvs = snap.get("intervals") or []
            if isinstance(itvs, list) and itvs:
                lines.append(f"• *MTF:* `{', '.join([str(x) for x in itvs])}`")

            aucs = snap.get("aucs") or {}
            if isinstance(aucs, dict) and aucs:
                # kompakt AUC satırı
                parts = []
                for k in sorted(aucs.keys(), key=lambda x: str(x)):
                    v = aucs.get(k)
                    try:
                        parts.append(f"{k}:{float(v):.3f}")
                    except Exception:
                        parts.append(f"{k}:{v}")
                lines.append("• *AUC:* `" + " | ".join(parts) + "`")

        except Exception as e:
            lines.append(f"⚠️ Snapshot okunamadı: `{e}`")
    else:
        lines.append("• Snapshot: `henüz yok (ilk loop bekleniyor)`")

    # 2) PerformanceTracker (fallback/ek bilgi)
    try:
        summary = performance_tracker.get_summary()
        lines.append("")
        lines.append("🧾 *Performans (tracker)*")
        lines.append(f"• Toplam İşlem: `{summary.get('total_trades', 0)}`")
        lines.append(f"• Başarılı: `{summary.get('successful_trades', 0)}`")
        lines.append(f"• Başarısız: `{summary.get('failed_trades', 0)}`")
        try:
            lines.append(f"• Toplam PnL: `{float(summary.get('total_pnl', 0.0)):.4f}`")
        except Exception:
            lines.append(f"• Toplam PnL: `{summary.get('total_pnl')}`")
    except Exception:
        pass

    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /status command used")


@require_auth
@rate_limit(0.8)
def trades_command(update: Update, context: CallbackContext) -> None:
    """/trades: placeholder (ileride PG/Redis bağlanır)."""
    update.message.reply_text("Trade geçmişi: Yakında eklenecek özellik 🛠")
    system_logger.info("Telegram: /trades command used")


@require_auth
@rate_limit(0.8)
def cmd_risk(update: Update, context: CallbackContext) -> None:
    """
    /risk → RiskManager state'ini gösterir.
    RiskManager instance'ı context.bot_data['risk_manager'] içinde bekliyoruz.
    """
    rm: Optional[RiskManager] = context.bot_data.get("risk_manager")  # type: ignore

    if rm is None:
        update.message.reply_text("RiskManager henüz init edilmemiş.")
        system_logger.warning("Telegram: /risk command used but RiskManager is None")
        return

    text = format_risk_status(rm)
    update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /risk command used")


def register_handlers(dispatcher) -> None:
    """
    Telegram dispatcher için tüm komut handler'larını register eder.
    telegram_bot.py içinde:
        from tg_bot.commands import register_handlers
        register_handlers(dispatcher)
    """
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("ping", ping_command))
    dispatcher.add_handler(CommandHandler("whoami", whoami_command))

    dispatcher.add_handler(CommandHandler("status", status_command))
    dispatcher.add_handler(CommandHandler("trades", trades_command))
    dispatcher.add_handler(CommandHandler("risk", cmd_risk))

    system_logger.info("[TG] Handlers registered: start/help/ping/whoami/status/trades/risk")
