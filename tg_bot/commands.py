from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List

from telegram import Update
from telegram.ext import CallbackContext, CommandHandler
from telegram.parsemode import ParseMode

from core.logger import system_logger
from core.risk_manager import RiskManager
from core.position_manager import PositionManager


def _fmt_ts(s: Any) -> str:
    try:
        if s is None:
            return "-"
        ss = str(s).replace("T", " ")
        return ss[:19]
    except Exception:
        return "-"


def _tail_csv(path: str, n: int = 10) -> List[List[str]]:
    p = Path(path)
    if (not p.exists()) or p.stat().st_size == 0:
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) <= 1:
        return []
    data = lines[1:]  # skip header
    n = max(1, min(int(n), 30))
    out: List[List[str]] = []
    for line in data[-n:]:
        out.append([c.strip() for c in line.split(",")])
    return out


def start_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Binance1-Pro botu aktif! 🚀\n/help ile komutları görebilirsin.")
    system_logger.info("Telegram: /start command used")


def help_command(update: Update, context: CallbackContext) -> None:
    msg = (
        "🧭 *Komutlar*\n"
        "/status - bot özeti (son snapshot)\n"
        "/trades [N] - son N trade kararı (logs/trade_decisions.csv)\n"
        "/positions - açık pozisyon\n"
        "/signal - son sinyal + p_used + MTF ağırlıkları\n"
        "/risk - risk durumu\n"
        "/ping - bot canlı mı\n"
        "/whoami - chat/user\n"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


def ping_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("pong ✅")


def whoami_command(update: Update, context: CallbackContext) -> None:
    u = update.effective_user
    c = update.effective_chat
    msg = (
        f"👤 user=`{getattr(u,'id','?')}` @{getattr(u,'username','-')}\n"
        f"💬 chat=`{getattr(c,'id','?')}` type=`{getattr(c,'type','-')}`"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


def status_command(update: Update, context: CallbackContext) -> None:
    te = context.bot_data.get("trade_executor")  # type: ignore
    snap = getattr(te, "last_snapshot", None) if te is not None else None

    if isinstance(snap, dict) and snap:
        msg = (
            "📊 *Durum*\n"
            f"symbol=`{snap.get('symbol','?')}` interval=`{snap.get('interval','?')}`\n"
            f"signal=*{snap.get('signal','?')}* src=`{snap.get('signal_source','?')}`\n"
            f"p_used=`{snap.get('p_used','?')}` p_single=`{snap.get('p_single','?')}`\n"
            f"ts=`{_fmt_ts(snap.get('ts'))}`"
        )
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    else:
        update.message.reply_text("📊 Bot çalışıyor. (son snapshot henüz oluşmadı)")
    system_logger.info("Telegram: /status command used")


def trades_command(update: Update, context: CallbackContext) -> None:
    n = 10
    try:
        if context.args and len(context.args) >= 1:
            n = int(context.args[0])
    except Exception:
        n = 10

    path = os.getenv("TRADE_DECISIONS_CSV_PATH", "logs/trade_decisions.csv")
    rows = _tail_csv(path, n=n)
    if not rows:
        update.message.reply_text("Henüz trade kaydı yok. (logs/trade_decisions.csv boş/yok)")
        system_logger.info("Telegram: /trades command used (no rows)")
        return

    lines = ["📜 *Son Trade Kararları*"]
    for i, r in enumerate(rows, 1):
        ts = r[0] if len(r) > 0 else "-"
        sym = r[1] if len(r) > 1 else "?"
        itv = r[2] if len(r) > 2 else "?"
        sig = r[3] if len(r) > 3 else "?"
        p = r[4] if len(r) > 4 else "?"
        src = r[5] if len(r) > 5 else "?"
        lines.append(f"{i}) `{_fmt_ts(ts)}` | *{sym}* `{itv}` | *{sig}* | p=`{p}` ({src})")

    update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /trades command used")


def positions_command(update: Update, context: CallbackContext) -> None:
    pm: PositionManager = context.bot_data.get("position_manager")  # type: ignore
    if pm is None:
        update.message.reply_text("PositionManager bot_data içinde yok. (main.py enjeksiyon eksik)")
        return

    symbol = context.bot_data.get("symbol") or os.getenv("SYMBOL", "BTCUSDT")  # type: ignore
    try:
        pos = pm.get_position(symbol)
    except Exception as e:
        update.message.reply_text(f"PositionManager get_position hata: {e}")
        return

    if not pos:
        update.message.reply_text("Açık pozisyon yok ✅")
        return

    side = str(pos.get("side", "?")).upper()
    msg = (
        "📌 *Açık Pozisyon*\n"
        f"symbol=`{pos.get('symbol', symbol)}` side=*{side}*\n"
        f"qty=`{pos.get('qty','?')}` entry=`{pos.get('entry_price','?')}`\n"
        f"SL=`{pos.get('sl_price','-')}` TP=`{pos.get('tp_price','-')}`\n"
        f"opened_at=`{_fmt_ts(pos.get('opened_at'))}`"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /positions command used")


def _mtf_summary(extra: Dict[str, Any]) -> str:
    md = extra.get("mtf_debug")
    if not isinstance(md, dict):
        return "MTF: (yok)"
    ens = md.get("ensemble_p", extra.get("ensemble_p"))
    itvs = md.get("intervals_used") or []
    w = md.get("weights_norm") or []
    per = md.get("per_interval") if isinstance(md.get("per_interval"), dict) else {}

    try:
        ens_s = f"{float(ens):.4f}" if ens is not None else "?"
    except Exception:
        ens_s = "?"

    weights_bits = []
    if itvs and w and len(itvs) == len(w):
        for itv, wi in zip(itvs, w):
            try:
                weights_bits.append(f"{itv}:{float(wi):.3f}")
            except Exception:
                pass

    per_bits = []
    for itv in itvs:
        d = per.get(itv, {}) if isinstance(per.get(itv), dict) else {}
        p_last = d.get("p_last")
        auc = d.get("auc_used")
        try:
            ps = f"{float(p_last):.3f}" if p_last is not None else "?"
        except Exception:
            ps = "?"
        try:
            as_ = f"{float(auc):.3f}" if auc is not None else "?"
        except Exception:
            as_ = "?"
        per_bits.append(f"{itv}(p={ps},auc={as_})")

    return (
        f"*MTF* ensemble_p=`{ens_s}`\n"
        f"weights: `{', '.join(weights_bits) if weights_bits else '-'}`\n"
        f"per: `{'; '.join(per_bits) if per_bits else '-'}`"
    )


def signal_command(update: Update, context: CallbackContext) -> None:
    te = context.bot_data.get("trade_executor")  # type: ignore
    snap = getattr(te, "last_snapshot", None) if te is not None else None
    if not isinstance(snap, dict) or not snap:
        update.message.reply_text("Henüz sinyal snapshot yok. (ilk loop sonrası oluşur)")
        return

    extra = snap.get("extra", {}) if isinstance(snap.get("extra"), dict) else {}
    msg = (
        "🧠 *Son Sinyal*\n"
        f"symbol=`{snap.get('symbol','?')}` interval=`{snap.get('interval','?')}`\n"
        f"signal=*{snap.get('signal','?')}* src=`{snap.get('signal_source','?')}`\n"
        f"p_used=`{snap.get('p_used','?')}` p_single=`{snap.get('p_single','?')}`\n"
        f"p_buy_raw=`{snap.get('p_buy_raw','?')}` p_buy_ema=`{snap.get('p_buy_ema','?')}`\n"
        f"whale=`{snap.get('whale_dir','none')}` score=`{snap.get('whale_score',0.0)}`\n"
        f"ts=`{_fmt_ts(snap.get('ts'))}`\n\n"
        f"{_mtf_summary(extra)}"
    )
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    system_logger.info("Telegram: /signal command used")


def cmd_risk(update: Update, context: CallbackContext) -> None:
    rm: RiskManager = context.bot_data.get("risk_manager")  # type: ignore
    if rm is None:
        update.message.reply_text("RiskManager henüz init edilmemiş.")
        system_logger.warning("Telegram: /risk command used but RiskManager is None")
        return

    try:
        from tg_bot.message_formatter import format_risk_status
        text = format_risk_status(rm)
        update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        msg = (
            "🛡️ *Risk*\n"
            f"daily_max_loss_usdt=`{getattr(rm,'daily_max_loss_usdt','?')}`\n"
            f"daily_max_loss_pct=`{getattr(rm,'daily_max_loss_pct','?')}`\n"
            f"max_open_trades=`{getattr(rm,'max_open_trades','?')}`"
        )
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    system_logger.info("Telegram: /risk command used")


def register_handlers(dispatcher) -> None:
    dispatcher.add_handler(CommandHandler("start", start_command))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("ping", ping_command))
    dispatcher.add_handler(CommandHandler("whoami", whoami_command))

    dispatcher.add_handler(CommandHandler("status", status_command))
    dispatcher.add_handler(CommandHandler("trades", trades_command))
    dispatcher.add_handler(CommandHandler("positions", positions_command))
    dispatcher.add_handler(CommandHandler("signal", signal_command))
    dispatcher.add_handler(CommandHandler("risk", cmd_risk))

    system_logger.info("[TG] Handlers registered: start/help/ping/whoami/status/trades/positions/signal/risk")
