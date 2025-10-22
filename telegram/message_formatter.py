from datetime import datetime

def format_trade_message(trade):
    msg = (
        f"📈 Trade Bilgisi\n"
        f"Coin: {trade['symbol']}\n"
        f"Tip: {trade['side']}\n"
        f"Miktar: {trade['qty']}\n"
        f"Fiyat: {trade['price']}\n"
        f"Durum: {trade['status']}\n"
        f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return msg

def format_alert_message(alert_type, details):
    msg = f"⚠️ ALERT: {alert_type}\nDetails: {details}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return msg
