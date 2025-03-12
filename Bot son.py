# 📌 Gerekli Kütüphaneleri Yükleme
!pip install numpy pandas torch torchvision tensorflow optuna scikit-learn joblib google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client requests geopy stable-baselines3 ccxt websockets ta textblob vaderSentiment web3

import os
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import requests
import asyncio
import websockets
import ccxt
import ta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from web3 import Web3
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
from kucoin.client import Trade, Market
from okx import Trade
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from stable_baselines3 import PPO
import gym
import telebot

# 📌 Google Drive API Kimlik Doğrulama
auth.authenticate_user()
drive_service = build('drive', 'v3')

# 📌 API Anahtarlarını Yükleme
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

# 📌 Binance sadece izleme modunda
binance = BinanceClient(API_KEYS["binance"]["key"], API_KEYS["binance"]["secret"])
kucoin_trade = Trade(API_KEYS["kucoin"]["key"], API_KEYS["kucoin"]["secret"], API_KEYS["kucoin"]["passphrase"])
okx_headers = {
    "OK-ACCESS-KEY": API_KEYS["okx"]["key"],
    "OK-ACCESS-SIGN": API_KEYS["okx"]["secret"],
    "OK-ACCESS-PASSPHRASE": API_KEYS["okx"]["passphrase"],
    "Content-Type": "application/json",
}

# 📌 WebSocket Bağlantıları
async def binance_websocket():
    uri = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            logging.info(f"Binance WebSocket Veri: {data}")

async def okx_websocket():
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            logging.info(f"OKX WebSocket Veri: {data}")

async def kucoin_websocket():
    uri = "wss://ws-api.kucoin.com/endpoint"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            logging.info(f"KuCoin WebSocket Veri: {data}")

# 📌 AI Destekli Kaldıraç Yönetimi (1x - 20x)
def determine_leverage(risk_level):
    if risk_level < 0.3:
        return np.random.randint(1, 5)
    elif 0.3 <= risk_level < 0.7:
        return np.random.randint(5, 15)
    else:
        return np.random.randint(15, 20)

# 📌 AI Destekli Alım-Satım (Sadece OKX ve KuCoin)
def open_futures_positions():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        price = float(binance.get_symbol_ticker(symbol=symbol)["price"])
        leverage = determine_leverage(price / 100000)

        # ✅ OKX işlemi
        requests.post("https://www.okx.com/api/v5/trade/order", headers=okx_headers, json={
            "instId": symbol, "tdMode": "cross", "side": "buy", "ordType": "market", "sz": "0.01"
        })
        
        # ✅ KuCoin işlemi
        kucoin_trade.create_market_order(symbol, "buy", size=0.01)
        
        logging.info(f"✅ {symbol} için işlem açıldı (OKX & KuCoin)")

# 📌 Telegram Botu ile Kontrol
bot = telebot.TeleBot(API_KEYS["telegram"]["token"])

@bot.message_handler(commands=['status'])
def send_pnl_status(message):
    bot.send_message(message.chat.id, "📊 Güncel PnL bilgisi yakında eklenecek.")

@bot.message_handler(commands=['open_trade'])
def manual_trade(message):
    bot.send_message(message.chat.id, "📈 Manuel işlem açılıyor...")
    open_futures_positions()

@bot.message_handler(commands=['start'])
def start_bot(message):
    bot.send_message(message.chat.id, "Bot çalışmaya başladı!")
    asyncio.run(open_futures_positions())

@bot.message_handler(commands=['stop'])
def stop_bot(message):
    bot.send_message(message.chat.id, "Bot durduruldu!")
    os._exit(0)

# 📌 Ana Çalıştırma
if __name__ == "__main__":
    bot.polling(none_stop=True)
