import os
import logging
import time
import numpy as np
import pandas as pd
import requests
import websockets
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
from kucoin.client import Client as KucoinClient
from okx.Client import Client as okxClient  # ✅ OKX API Client Eklendi
from web3 import Web3
from stable_baselines3 import PPO
import gym
from sklearn.preprocessing import MinMaxScaler
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import telebot

# 📌 **Google Drive Bağlantısı**
auth.authenticate_user()
drive_service = build('drive', 'v3')

# 📌 **API Anahtarlarını Yükle**
load_dotenv()
API_KEYS = {
    "binance": {"key": os.getenv("BINANCE_API_KEY"), "secret": os.getenv("BINANCE_API_SECRET")},
    "kucoin": {"key": os.getenv("KUCOIN_API_KEY"), "secret": os.getenv("KUCOIN_API_SECRET"), "passphrase": os.getenv("KUCOIN_API_PASSPHRASE")},
    "okx": {"key": os.getenv("OKX_API_KEY"), "secret": os.getenv("OKX_API_SECRET"), "passphrase": os.getenv("OKX_API_PASSPHRASE")},
    "telegram": {"token": os.getenv("TELEGRAM_BOT_TOKEN"), "chat_id": os.getenv("TELEGRAM_CHAT_ID")},
    "google_drive_folder": os.getenv("GOOGLE_DRIVE_FOLDER_ID")
}

# 📌 **Borsa Bağlantıları**
clients = {
    "binance": BinanceClient(API_KEYS["binance"]["key"], API_KEYS["binance"]["secret"]),
    "kucoin": KucoinClient(API_KEYS["kucoin"]["key"], API_KEYS["kucoin"]["secret"], API_KEYS["kucoin"]["passphrase"]),
    "okx": okxClient(API_KEYS["okx"]["key"], API_KEYS["okx"]["secret"], API_KEYS["okx"]["passphrase"]),  # ✅ OKX API Client Tanımlandı
}

# 📌 **Telegram Bildirimi**
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{API_KEYS['telegram']['token']}/sendMessage"
    data = {"chat_id": API_KEYS["telegram"]["chat_id"], "text": message}
    requests.post(url, data=data)

# 📌 **PnL Bazlı Hedge Mekanizması**
def dynamic_hedging(pnl, balance, leverage, volatility):
    """ Hedge işlemini dinamik olarak ayarla. """
    MAX_HEDGE_RATIO = 0.5
    risk_factor = abs(pnl) / balance
    hedge_size = min(risk_factor * leverage * volatility, balance * MAX_HEDGE_RATIO)
    return hedge_size if pnl < -5 else 0

# 📌 **Order Book Analizi (Binance, KuCoin, OKX)**
def analyze_order_book(symbol="BTCUSDT", exchange="binance"):
    """ Binance, KuCoin ve OKX borsaları için Order Book analizi. """
    if exchange == "binance":
        order_book = clients["binance"].get_order_book(symbol=symbol, limit=100)
    elif exchange == "kucoin":
        response = requests.get(f"https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol={symbol}")
        order_book = response.json()
    elif exchange == "okx":
        order_book = clients["okx"].get_orderbook(symbol)  # ✅ OKX API Client ile Order Book Analizi

    bid_volumes = np.array([float(order[1]) for order in order_book["bids"]])
    ask_volumes = np.array([float(order[1]) for order in order_book["asks"]])
    return (bid_volumes.sum() - ask_volumes.sum()) / (bid_volumes.sum() + ask_volumes.sum())

# 📌 **Trade Kararı**
def determine_trade(symbol):
    """ Order book analizine ve piyasa verilerine göre trade sinyali üret. """
    order_book_imbalance = analyze_order_book(symbol)
    if order_book_imbalance > 0.2:
        return "BUY"
    elif order_book_imbalance < -0.2:
        return "SELL"
    return "HOLD"

# 📌 **Trade Büyüklüğünü Dinamik Hesaplama**
def dynamic_trade_size(balance, volatility):
    """ Piyasa volatilitesine göre işlem büyüklüğünü ayarla. """
    risk_factor = 0.02
    return balance * risk_factor * (1 + volatility)

# 📌 **KuCoin ve OKX’de İşlem Açma**
def execute_trade(symbol, side, quantity, exchange):
    """ OKX ve KuCoin borsalarında işlem açma fonksiyonu. """
    if exchange == "kucoin":
        clients["kucoin"].create_market_order(symbol, side, size=quantity)
    elif exchange == "okx":
        clients["okx"].create_order(symbol, side, "market", quantity)  # ✅ OKX API Client ile Market Order

# 📌 **Google Drive'a İşlem Kayıtlarını Kaydetme**
def save_trade_to_drive(data):
    """ İşlem kayıtlarını Google Drive'a kaydeder. """
    file_path = "trading_data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["timestamp", "symbol", "action", "exchange"])

    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(file_path, index=False)

    file_metadata = {"name": "trading_data.csv", "parents": [API_KEYS["google_drive_folder"]]}
    media = MediaFileUpload(file_path, mimetype="text/csv")
    drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

# 📌 **Ana Çalıştırma**
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    max_trades = 8
    trade_count = 0

    while trade_count < max_trades:
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
            trade_signal = determine_trade(symbol)
            balance = 1000
            volatility = np.random.uniform(0.01, 0.05)
            position_size = dynamic_trade_size(balance, volatility)

            if trade_signal == "BUY":
                execute_trade(symbol, "buy", position_size, "kucoin")
                execute_trade(symbol, "buy", position_size, "okx")
                send_telegram_alert(f"🚀 {symbol} için ALIM işlemi gerçekleşti!")
                save_trade_to_drive({"timestamp": time.time(), "symbol": symbol, "action": "buy", "exchange": "kucoin"})
                save_trade_to_drive({"timestamp": time.time(), "symbol": symbol, "action": "buy", "exchange": "okx"})

            elif trade_signal == "SELL":
                execute_trade(symbol, "sell", position_size, "kucoin")
                execute_trade(symbol, "sell", position_size, "okx")
                send_telegram_alert(f"📉 {symbol} için SATIŞ işlemi gerçekleşti!")
                save_trade_to_drive({"timestamp": time.time(), "symbol": symbol, "action": "sell", "exchange": "kucoin"})
                save_trade_to_drive({"timestamp": time.time(), "symbol": symbol, "action": "sell", "exchange": "okx"})

            trade_count += 1
            if trade_count >= max_trades:
                break 

        time.sleep(10)
