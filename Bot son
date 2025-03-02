import os
import logging
import time
import numpy as np
import pandas as pd
import threading
import asyncio
import requests
import websockets
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
from kucoin.client import Client as KucoinClient
from okx import MarketData, Trade
from web3 import Web3
from stable_baselines3 import PPO
import gym
from sklearn.preprocessing import MinMaxScaler

# 📌 API Anahtarlarını Yükle
load_dotenv()
API_KEYS = {
    "binance": {"key": os.getenv("BINANCE_API_KEY"), "secret": os.getenv("BINANCE_API_SECRET")},
    "kucoin": {"key": os.getenv("KUCOIN_API_KEY"), "secret": os.getenv("KUCOIN_API_SECRET")},
    "okx": {"key": os.getenv("OKX_API_KEY"), "secret": os.getenv("OKX_API_SECRET")},
    "alchemy": os.getenv("ALCHEMY_API_KEY"),
    "infura": os.getenv("INFURA_API_KEY"),
    "the_graph": os.getenv("THE_GRAPH_API_URL"),
}

# 📌 Borsa Bağlantıları
clients = {
    "binance": BinanceClient(API_KEYS["binance"]["key"], API_KEYS["binance"]["secret"]),
    "kucoin": KucoinClient(API_KEYS["kucoin"]["key"], API_KEYS["kucoin"]["secret"]),
    "okx_market": MarketData(API_KEYS["okx"]["key"], API_KEYS["okx"]["secret"]),
    "okx_trade": Trade(API_KEYS["okx"]["key"], API_KEYS["okx"]["secret"]),
}

# 📌 Blockchain API Bağlantıları
w3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{API_KEYS['infura']}"))

# 📌 **PnL Bazlı Hedge Mekanizması**
def dynamic_hedging(pnl, balance, leverage):
    MAX_HEDGE_RATIO = 0.5
    if pnl < -5:
        hedge_size = np.clip(abs(pnl) * (leverage / 10), 0.01, balance * MAX_HEDGE_RATIO)
        logging.warning(f"📉 PnL kötüleşti! {hedge_size} büyüklüğünde hedge işlemi açılıyor.")
        return hedge_size
    return 0

# 📌 **WebSocket Veri Yönetimi**
class WebSocketManager:
    def __init__(self):
        self.data = {}

    async def websocket_listener(self, exchange, symbol):
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                self.data[symbol] = float(data["p"])
                logging.info(f"📡 {exchange.upper()} | {symbol}: {self.data[symbol]}")

websocket_manager = WebSocketManager()

# 📌 **AI Modeli: LSTM + Transformers**
class AITradingModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_layer_size, nhead=8, num_encoder_layers=2)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer(lstm_out, lstm_out)
        return self.linear(transformer_out[:, -1])

# 📌 **Reinforcement Learning: PPO ile Strateji Optimizasyonu**
def train_rl_model():
    env = gym.make("TradingEnv-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("ppo_trading_model")
    return model

# 📌 **LightGBM Modeli ile Alım-Satım Kararı**
def train_lightgbm():
    data = pd.read_csv("market_data.csv")
    scaler = MinMaxScaler()
    features = ["open", "high", "low", "close", "volume"]
    data[features] = scaler.fit_transform(data[features])

    X = data[features]
    y = np.where(data["close"].shift(-1) > data["close"], 1, 0)

    model = lgb.LGBMClassifier(boosting_type="gbdt", num_leaves=31, learning_rate=0.005, n_estimators=200)
    model.fit(X, y)
    return model, scaler

def determine_trade():
    model, scaler = train_lightgbm()
    current_price = np.array([[50000, 50500, 49500, 50000, 10000]])
    X_scaled = scaler.transform(current_price)

    prediction = model.predict(X_scaled)

    if prediction == 1:
        leverage = np.random.randint(3, 10)
        position_size = np.random.uniform(0.1, 1)
        return f"BUY | Leverage: {leverage}x | Position Size: {position_size} BTC"
    return "SELL"

# 📌 **Blockchain ve On-Chain Analiz**
def get_blockchain_data():
    if w3.is_connected():
        logging.info("✅ Blockchain verisi çekiliyor...")
        return w3.eth.get_block("latest")
    return None

def get_graph_data():
    query = {"query": "{ transfers(first: 5) { id from to value } }"}
    response = requests.post(API_KEYS["the_graph"], json=query)
    return response.json() if response.status_code == 200 else None

# 📌 **WebSocket ile Gerçek Zamanlı Veri Akışı**
async def websocket_listener(symbol):
    uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(f"📡 {symbol}: {message}")

# 📌 **Ana Çalıştırma**
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # WebSocket Bağlantıları
    threading.Thread(target=lambda: asyncio.run(websocket_listener("BTCUSDT")), daemon=True).start()

    # Trade Kararı
    while True:
        trade_signal = determine_trade()
        if "BUY" in trade_signal:
            logging.info(f"🚀 Alım sinyali: {trade_signal}")
        elif "SELL" in trade_signal:
            logging.info(f"📉 Satış sinyali: {trade_signal}")
        else:
            logging.info("⏳ Beklemede...")

        time.sleep(60)  # 1 dakika sonra tekrar kontrol et
