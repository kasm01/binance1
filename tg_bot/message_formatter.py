# tg_bot/message_formatter.py
from datetime import datetime
from typing import Dict, Any


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

