from __future__ import annotations

import os
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List
from collections import deque
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


def _parse_dt_any(s: str) -> Optional[datetime]:
    """
    CSV içindeki timestamp'i olabildiğince esnek parse eder.
    - ISO (2026-02-02T20:34:10)
    - "YYYY-MM-DD HH:MM:SS"
    - "YYYY/MM/DD HH:MM:SS"
    Sonuç: UTC tz-aware datetime döndürür.
    """
    s = (s or "").strip()
    if not s:
        return None

    # ISO ve benzeri
    try:
        ss = s.replace("T", " ")
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Klasik formatlar
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def _tail_csv(path: str, n: int = 10) -> List[List[str]]:
    """
    Robust tail reader for CSV:
    - Uses csv.reader (handles edge cases better than split(','))
    - Keeps only last N data rows (header excluded)
    """
    p = Path(path)
    if (not p.exists()) or p.stat().st_size == 0:
        return []

    n = max(1, min(int(n), 30))

    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            if not header:
                return []

            buf = deque(maxlen=n)
            for row in reader:
                # row boş satırsa geç
                if not row or all((c is None or str(c).strip() == "") for c in row):
                    continue
                buf.append([str(c).strip() for c in row])

            return list(buf)

    except Exception as e:
        try:
            system_logger.exception("[TG] _tail_csv failed path=%s err=%s", path, e)
        except Exception:
            pass
        return []

def _file_diag_csv(path: str) -> Dict[str, Any]:
    p = Path(path)
    diag: Dict[str, Any] = {
        "exists": False,
        "abs": str(p.resolve()),
        "size": 0,
        "mtime": "-",
        "lines": 0,
        "head": [],
        "tail": [],
    }

    try:
        if not p.exists():
            return diag
        diag["exists"] = True
        st = p.stat()
        diag["size"] = int(st.st_size)
        diag["mtime"] = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    try:
        # küçük dosya: komple oku (trade_decisions.csv genelde küçük)
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        diag["lines"] = len(lines)
        diag["head"] = lines[:2]
        diag["tail"] = lines[-2:] if len(lines) >= 2 else lines
    except Exception:
        pass

    return diag

def start_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("Binance1-Pro botu aktif! 🚀\n/help ile komutları görebilirsin.")
    system_logger.info("Telegram: /start command used")


def help_command(update: Update, context: CallbackContext) -> None:
    msg = (
        "🧭 *Komutlar*\n"
        "/status - bot özeti (son snapshot)\n"
        "/trades [N] [scope] - son N kayıt (default all)\n"
        "    scope: all | 24h | 6h | 1h | Nh (örn 12h)\n"
        "/positions - açık pozisyon\n"
        "/signal - son sinyal + p_used + MTF ağırlıkları\n"
        "/risk - risk durumu\n"
        "/ping - bot canlı mı\n"
        "/whoami - chat/user\n"
    )
    update.message.reply_text(msg)


def ping_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text("pong ✅")


def whoami_command(update: Update, context: CallbackContext) -> None:
    u = update.effective_user
    c = update.effective_chat
    msg = (
        f"👤 user=`{getattr(u,'id','?')}` @{getattr(u,'username','-')}\n"
        f"💬 chat=`{getattr(c,'id','?')}` type=`{getattr(c,'type','-')}`"
    )
    # TG_MD_FALLBACK: Markdown parse hatasında plain-text fallback
    try:
        update.message.reply_text(msg)
    except Exception as e:
        try:
            context.bot.logger.warning("[TG] Markdown parse error -> fallback plain. err=%s", e)
        except Exception:
            pass
        update.message.reply_text(msg)


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
        update.message.reply_text(msg)
    else:
        update.message.reply_text("📊 Bot çalışıyor. (son snapshot henüz oluşmadı)")
    system_logger.info("Telegram: /status command used")


def trades_command(update: Update, context: CallbackContext) -> None:
    """
    /trades [N] [scope]
      - N: kaç kayıt (default 10, max 30)
      - scope: all | 24h | 6h | 1h | Nh (örn 12h)
    """
    n = 10
    scope = "all"

    try:
        if context.args and len(context.args) >= 1:
            n = int(context.args[0])
    except Exception:
        n = 10

    try:
        if context.args and len(context.args) >= 2:
            scope = str(context.args[1]).strip().lower()
    except Exception:
        scope = "all"

    n = max(1, min(int(n), 30))

    path = os.getenv("TRADE_DECISIONS_CSV_PATH", "logs/trade_decisions.csv")
    diag = _file_diag_csv(path)

    # dosya yok/boş
    if not diag.get("exists") or int(diag.get("size") or 0) == 0:
        update.message.reply_text(
            "Trade kaydı yok.\n"
            f"- path=`{path}`\n"
            f"- abs=`{diag.get('abs','-')}`\n"
            f"- exists=`{diag.get('exists')}` size=`{diag.get('size')}` mtime=`{diag.get('mtime')}`"
        )
        system_logger.info("Telegram: /trades command used (missing/empty file)")
        return

    # ham tail al (son N satır)
    rows = _tail_csv(path, n=n)

    # Burada artık “header var ama satır yok” demeden önce diag’ı yazdırıyoruz
    if not rows:
        # Dosya dolu ama parse ile rows çıkaramadık -> parse/okuma sorunu
        sample_head = (diag.get("head") or ["-","-"])
        sample_tail = (diag.get("tail") or ["-","-"])
        update.message.reply_text(
            "Trade kaydı okunamadı (dosya dolu görünüyor) → CSV parse/okuma sorunu.\n"
            f"- path=`{path}`\n"
            f"- abs=`{diag.get('abs','-')}`\n"
            f"- size=`{diag.get('size')}` mtime=`{diag.get('mtime')}` lines=`{diag.get('lines')}`\n"
            f"- head=`{sample_head[0]}`\n"
            f"- tail=`{sample_tail[-1]}`"
        )
        system_logger.info("Telegram: /trades command used (parse failed) diag=%s", diag)
        return

    # scope filtresi
    lookback_sec: Optional[int] = None
    if scope in ("all", "0", "none", "nofilter"):
        lookback_sec = None
    elif scope.endswith("h"):
        try:
            hours = int(scope[:-1])
            lookback_sec = max(1, hours) * 3600
        except Exception:
            lookback_sec = 24 * 3600
    elif scope in ("24h", "24"):
        lookback_sec = 24 * 3600
    elif scope in ("6h", "6"):
        lookback_sec = 6 * 3600
    elif scope in ("1h", "1"):
        lookback_sec = 1 * 3600
    else:
        lookback_sec = None

    now = datetime.now()
    filtered = rows

    if lookback_sec is not None:
        cut = now.timestamp() - lookback_sec
        tmp = []
        for r in rows:
            ts_raw = r[0] if len(r) > 0 else None
            try:
                ts = str(ts_raw).replace("T", " ")
                dt = datetime.fromisoformat(ts)
                if dt.timestamp() >= cut:
                    tmp.append(r)
            except Exception:
                pass
        filtered = tmp

    last_ts = _fmt_ts(rows[-1][0] if rows and rows[-1] else None)

    if not filtered:
        update.message.reply_text(
            "Filtre kapsamında trade kaydı yok.\n"
            f"- scope=`{scope}` n=`{n}`\n"
            f"- file_mtime=`{diag.get('mtime')}` last_row_ts=`{last_ts}`\n"
            f"- abs=`{diag.get('abs')}`"
        )
        system_logger.info("Telegram: /trades command used (filtered empty) scope=%s", scope)
        return

    lines = [f"📜 *Trade Kararları* scope=`{scope}` n=`{n}`"]
    lines.append(f"_file_mtime={diag.get('mtime')} last_row_ts={last_ts}_")
    lines.append(f"_abs={diag.get('abs')}_")

    for i, r in enumerate(filtered, 1):
        ts = r[0] if len(r) > 0 else "-"
        sym = r[1] if len(r) > 1 else "?"
        itv = r[2] if len(r) > 2 else "?"
        sig = r[3] if len(r) > 3 else "?"
        pval = r[4] if len(r) > 4 else "?"
        src = r[5] if len(r) > 5 else "?"
        lines.append(f"{i}) `{_fmt_ts(ts)}` | *{sym}* `{itv}` | *{sig}* | p=`{pval}` ({src})")

    update.message.reply_text("\n".join(lines))
    system_logger.info("Telegram: /trades command used (scope=%s n=%s)", scope, n)


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
    update.message.reply_text(msg)
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
    update.message.reply_text(msg)
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
        update.message.reply_text(text)
    except Exception:
        msg = (
            "🛡️ *Risk*\n"
            f"daily_max_loss_usdt=`{getattr(rm,'daily_max_loss_usdt','?')}`\n"
            f"daily_max_loss_pct=`{getattr(rm,'daily_max_loss_pct','?')}`\n"
            f"max_open_trades=`{getattr(rm,'max_open_trades','?')}`"
        )
        update.message.reply_text(msg)

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

