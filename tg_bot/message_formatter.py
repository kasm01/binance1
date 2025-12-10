# tg_bot/message_formatter.py
from datetime import datetime
from typing import Dict, Any
from core.risk_manager import RiskManager


def format_trade_message(trade: Dict[str, Any]) -> str:
    """
    Trade sözlüğünü okunabilir Telegram mesajına çevirir.
    Beklenen trade dict alanları: symbol, side, qty, price, status
    """
    msg = (
        "📈 Trade Bilgisi\n"
        f"Coin: {trade.get('symbol')}\n"
        f"Tip: {trade.get('side')}\n"
        f"Miktar: {trade.get('qty')}\n"
        f"Fiyat: {trade.get('price')}\n"
        f"Durum: {trade.get('status')}\n"
        f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return msg


def format_alert_message(alert_type: str, details: str) -> str:
    """
    Genel amaçlı uyarı mesaj formatlayıcı.
    """
    msg = (
        f"⚠️ ALERT: {alert_type}\n"
        f"Details: {details}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return msg

def format_risk_status(rm: RiskManager) -> str:
    """
    RiskManager state'ini Telegram için okunaklı string'e çevirir.
    """
    last_open = rm.last_open_trade or {}
    last_close = rm.last_close_trade or {}

    def _fmt_trade(tr):
        if not tr:
            return "—"
        ts = tr.get("timestamp", "—")
        symbol = tr.get("symbol", "—")
        side = tr.get("side", "—")
        qty = tr.get("qty", "—")
        price = tr.get("price") or tr.get("entry_price") or tr.get("exit_price")
        interval = tr.get("interval", "—")
        reason = tr.get("reason") or tr.get("meta", {}).get("reason")
        return f"{ts}\n  {symbol} {interval} | {side} | qty={qty:.4f} @ {price:.2f if price else 0} | reason={reason}"

    lines = []
    lines.append("*Risk Durumu*")
    lines.append(f"- Günlük PnL: `{rm.daily_realized_pnl:.2f} USDT`")
    lines.append(f"- Günlük max loss (USDT): `{rm.daily_max_loss_usdt:.2f}`")
    lines.append(f"- Günlük max loss (%): `{rm.daily_max_loss_pct:.4f}`")
    lines.append(f"- Günlük consecutive losses: `{rm.consecutive_losses}`")
    lines.append(f"- Açık pozisyon sayısı: `{rm.open_trades}`")
    lines.append("")
    lines.append("*Toplam İstatistikler*")
    lines.append(f"- Toplam trade: `{rm.total_trades}`")
    lines.append(f"- Kazanan: `{rm.total_wins}`")
    lines.append(f"- Kaybeden: `{rm.total_losses}`")
    if rm.total_trades > 0:
        win_rate = rm.total_wins / rm.total_trades * 100
        lines.append(f"- Win rate: `{win_rate:.2f}%`")
    lines.append(f"- Son realized PnL: `{rm.last_realized_pnl:.2f} USDT`")
    lines.append("")
    lines.append("*Son Açılan Pozisyon*")
    if not last_open:
        lines.append("—")
    else:
        lines.append(f"`{last_open}`")  # istersen burada daha detaylı formatlama yapabilirsin
    lines.append("")
    lines.append("*Son Kapanan Pozisyon*")
    if not last_close:
        lines.append("—")
    else:
        lines.append(f"`{last_close}`")

    return "\n".join(lines)
