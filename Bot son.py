import os
import sys
import io
import logging
import time
import numpy as np
import pandas as pd
import requests
import hmac
import hashlib
import base64
import torch
import lightgbm as lgb
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
from kucoin.client import Trade, Market
from okx import MarketData, Trade
from web3 import Web3
from stable_baselines3 import PPO
import gym
from sklearn.preprocessing import MinMaxScaler
import telebot

# ✅ Google Colab için event loop düzeltmesi
nest_asyncio.apply()

# ✅ Çıkışları UTF-8 formatına getir
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# 📌 **API Anahtarlarını Yükle**
load_dotenv()

API_KEYS = {
    "binance": {
        "key": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET")
    },
    "kucoin": {
        "key": os.getenv("KUCOIN_API_KEY"),
        "secret": os.getenv("KUCOIN_API_SECRET"),
        "passphrase": os.getenv("KUCOIN_API_PASSPHRASE")
    },
    "okx": {
        "key": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "passphrase": os.getenv("OKX_API_PASSPHRASE")
    },
    "telegram": {
        "token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID")
    }
}

# ✅ **Borsa Bağlantıları**
clients = {}

try:
    # ✅ Binance API Bağlantısı
    clients["binance"] = BinanceClient(API_KEYS["binance"]["key"], API_KEYS["binance"]["secret"])

    # ✅ KuCoin API Bağlantısı
    clients["kucoin_trade"] = Trade(
        key=API_KEYS["kucoin"]["key"],
        secret=API_KEYS["kucoin"]["secret"],
        passphrase=API_KEYS["kucoin"]["passphrase"]
    )
    clients["kucoin_market"] = Market()

except Exception as e:
    logging.error(f"API Bağlantı Hatası: {str(e)}")

# ✅ **OKX API Bağlantısı**
OKX_API_URL = "https://www.okx.com"

# 📌 **OKX API için imzalama fonksiyonu**
def sign_request(method, endpoint, body=""):
    """OKX API için HMAC-SHA256 imzalama işlemi"""
    timestamp = str(time.time())

    message = timestamp + method + endpoint + body
    signature = base64.b64encode(
        hmac.new(
            API_KEYS["okx"]["secret"].encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode()

    headers = {
        "OK-ACCESS-KEY": API_KEYS["okx"]["key"],
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": API_KEYS["okx"]["passphrase"],
        "Content-Type": "application/json",
    }
    return headers

# 📌 **Telegram Bildirimi**
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{API_KEYS['telegram']['token']}/sendMessage"
    data = {"chat_id": API_KEYS["telegram"]["chat_id"], "text": message}
    requests.post(url, data=data)

# 📌 **Order Book Analizi (Binance, KuCoin, OKX)**
def analyze_order_book(symbol="BTCUSDT", exchange="binance"):
    """ Binance, KuCoin ve OKX borsaları için Order Book analizi. """
    try:
        if exchange == "binance":
            order_book = clients["binance"].get_order_book(symbol=symbol, limit=100)
        elif exchange == "kucoin":
            order_book = clients["kucoin_market"].get_order_book(symbol, limit=20)
        elif exchange == "okx":
            headers = sign_request("GET", f"/api/v5/market/books?instId={symbol}")
            response = requests.get(f"{OKX_API_URL}/api/v5/market/books?instId={symbol}", headers=headers)
            order_book = response.json()

        bid_volumes = np.array([float(order[1]) for order in order_book["bids"]])
        ask_volumes = np.array([float(order[1]) for order in order_book["asks"]])
        return (bid_volumes.sum() - ask_volumes.sum()) / (bid_volumes.sum() + ask_volumes.sum())
    
    except Exception as e:
        logging.error(f"Order Book Hatası: {str(e)}")
        return 0

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
    try:
        if exchange == "kucoin":
            clients["kucoin_trade"].create_market_order(symbol, side, size=quantity)
        elif exchange == "okx":
            headers = sign_request("POST", "/api/v5/trade/order")
            data = {"instId": symbol, "tdMode": "cross", "side": side, "ordType": "market", "sz": str(quantity)}
            requests.post(f"{OKX_API_URL}/api/v5/trade/order", json=data, headers=headers)
    
    except Exception as e:
        logging.error(f"İşlem Açma Hatası: {str(e)}")

# 📌 **Ana Çalıştırma (Google Colab Uyumlu)**
async def main():
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

            elif trade_signal == "SELL":
                execute_trade(symbol, "sell", position_size, "kucoin")
                execute_trade(symbol, "sell", position_size, "okx")
                send_telegram_alert(f"📉 {symbol} için SATIŞ işlemi gerçekleşti!")

            trade_count += 1
            if trade_count >= max_trades:
                break

        await asyncio.sleep(10)

# ✅ Google Colab Uyumlu Çalıştır
loop = asyncio.get_running_loop()
task = loop.create_task(main())
