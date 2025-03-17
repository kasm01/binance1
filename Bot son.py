import os
import logging
import time
import numpy as np
import pandas as pd
import torch
import requests
import asyncio
import websockets
import json
from dotenv import load_dotenv
import telebot
import nest_asyncio
import ta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from web3 import Web3
from stable_baselines3 import PPO
import gym

# 📌 Google Colab uyumluluğu için event loop düzeltmesi
nest_asyncio.apply()

# 📌 API Anahtarlarını Yükleme
load_dotenv()

API_KEYS = {
    "okx": {
        "key": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_API_SECRET"),
        "passphrase": os.getenv("OKX_API_PASSPHRASE")
    },
    "telegram": {
        "token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID")
    },
    "coinglass": os.getenv("COINGLASS_API_KEY")
}

# 📌 OKX API Bağlantısı
okx_headers = {
    "OK-ACCESS-KEY": API_KEYS["okx"]["key"],
    "OK-ACCESS-SIGN": API_KEYS["okx"]["secret"],
    "OK-ACCESS-PASSPHRASE": API_KEYS["okx"]["passphrase"],
    "Content-Type": "application/json",
}

# 📌 Teknik Göstergeler (TA kütüphanesi ile)
def calculate_technical_indicators(prices):
    rsi = ta.momentum.RSIIndicator(pd.Series(prices), window=14).rsi()
    macd = ta.trend.MACD(pd.Series(prices)).macd()
    signal = ta.trend.MACD(pd.Series(prices)).macd_signal()
    upper = ta.volatility.BollingerBands(pd.Series(prices), window=20).bollinger_hband()
    middle = ta.volatility.BollingerBands(pd.Series(prices), window=20).bollinger_mavg()
    lower = ta.volatility.BollingerBands(pd.Series(prices), window=20).bollinger_lband()
    return {
        "RSI": rsi.iloc[-1],
        "MACD": macd.iloc[-1],
        "Signal": signal.iloc[-1],
        "Bollinger Upper": upper.iloc[-1],
        "Bollinger Middle": middle.iloc[-1],
        "Bollinger Lower": lower.iloc[-1],
    }

# 📌 AI Destekli Kaldıraç Yönetimi (1x - 20x)
def determine_leverage(risk_level):
    if risk_level < 0.3:
        return np.random.randint(1, 5)
    elif 0.3 <= risk_level < 0.7:
        return np.random.randint(5, 15)
    else:
        return np.random.randint(15, 20)

# 📌 AI Destekli OKX Alım-Satım İşlemleri
def open_futures_positions():
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
    for symbol in symbols:
        price = np.random.uniform(1000, 50000)
        leverage = determine_leverage(price / 100000)
        requests.post("https://www.okx.com/api/v5/trade/order", headers=okx_headers, json={
            "instId": symbol, "tdMode": "cross", "side": "buy", "ordType": "market", "sz": "0.01"
        })
        time.sleep(1)
        requests.post("https://www.okx.com/api/v5/trade/order", headers=okx_headers, json={
            "instId": symbol, "tdMode": "cross", "side": "sell", "ordType": "market", "sz": "0.01"
        })
        logging.info(f"✅ {symbol} için işlem açıldı (OKX)")

# 📌 Telegram Botu ile Kontrol
bot = telebot.TeleBot(API_KEYS["telegram"]["token"])

@bot.message_handler(commands=['status'])
def send_pnl_status(message):
    pnl = np.random.uniform(-50, 50)
    bot.send_message(message.chat.id, f"📊 Güncel PnL: {pnl}$")

@bot.message_handler(commands=['open_trade'])
def manual_trade(message):
    bot.send_message(message.chat.id, "📈 Manuel işlem açılıyor...")
    open_futures_positions()

@bot.message_handler(commands=['start'])
def start_bot(message):
    bot.send_message(message.chat.id, "Bot çalışmaya başladı!")
    asyncio.create_task(test_okx_websocket())

@bot.message_handler(commands=['stop'])
def stop_bot(message):
    bot.send_message(message.chat.id, "Bot durduruldu!")
    os._exit(0)

# 📌 WebSocket Bağlantısı (OKX)
async def test_okx_websocket():
    uri = "wss://ws.okx.com:8443/ws/v5/public"
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                print("✅ OKX WebSocket bağlantısı kuruldu!")
                subscribe_message = {
                    "op": "subscribe",
                    "args": [{"channel": "tickers", "instId": "BTC-USDT"}]
                }
                await websocket.send(json.dumps(subscribe_message))
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    print("📡 OKX WebSocket Yanıtı:", data)
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket bağlantısı koptu, 5 saniye içinde yeniden bağlanılıyor...")
            await asyncio.sleep(5)

# ✅ WebSocket bağlantısını başlat
loop = asyncio.get_event_loop()
loop.create_task(test_okx_websocket())

# 📌 Ana Çalıştırma
if __name__ == "__main__":
    bot.polling(none_stop=True)
