import asyncio
import logging

# ------------------------------
# Core & Config
# ------------------------------
from config.load_env import load_environment_variables
from config.validate_env import validate_env
from core.logger import setup_logger
from core.exceptions import GlobalExceptionHandler

# ------------------------------
# Data Modules
# ------------------------------
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner
from data.batch_learning import BatchLearner

# ------------------------------
# Models
# ------------------------------
from models.ensemble_model import EnsembleModel
from models.hyperparameter_tuner import HyperparameterTuner
from models.fallback_model import FallbackModel

# ------------------------------
# Trading
# ------------------------------
from trading.position_manager import PositionManager
from trading.capital_manager import CapitalManager
from trading.risk_manager import RiskManager
from trading.strategy_engine import StrategyEngine
from trading.trade_executor import TradeExecutor
from trading.multi_trade_engine import MultiTradeEngine
from trading.fallback_trade import FallbackTrade

# ------------------------------
# Websocket & Monitoring
# ------------------------------
from websocket.binance_ws import BinanceWebSocket
from monitoring.performance_tracker import PerformanceTracker
from monitoring.system_health import SystemHealth
from monitoring.trade_logger import TradeLogger
from monitoring.alert_system import AlertSystem

# ------------------------------
# Telegram
# ------------------------------
from telegram.telegram_bot import TelegramBot

# ------------------------------
# Main Execution
# ------------------------------

def main():
    # 1️⃣ Çevresel değişkenleri yükle ve doğrula
    validate_env()
    env_vars = load_environment_variables()
    
    # 2️⃣ Logger kurulumu
    logger = setup_logger("binance1_pro_bot")

    # 3️⃣ Global exception handler
    GlobalExceptionHandler.register()
    
    # 4️⃣ Data loader & feature engineering
    data_loader = DataLoader(env_vars)
    raw_data = data_loader.load_recent_data()
    
    feature_engineer = FeatureEngineer(raw_data)
    features = feature_engineer.transform()
    
    # 5️⃣ Anomali tespiti
    anomaly_detector = AnomalyDetector(features)
    clean_features = anomaly_detector.remove_anomalies()
    
    # 6️⃣ Batch + Online learning
    batch_learner = BatchLearner(clean_features)
    batch_model = batch_learner.train()
    
    online_learner = OnlineLearner(batch_model)
    
    # 7️⃣ Model ensemble
    ensemble_model = EnsembleModel(online_learner)
    fallback_model = FallbackModel()
    
    # 8️⃣ Trading setup
    capital_manager = CapitalManager(env_vars)
    position_manager = PositionManager()
    risk_manager = RiskManager()
    strategy_engine = StrategyEngine(ensemble_model, fallback_model, risk_manager)
    trade_executor = TradeExecutor(env_vars)
    multi_trade_engine = MultiTradeEngine(trade_executor, position_manager)
    
    # 9️⃣ Websocket
    ws = BinanceWebSocket(env_vars, multi_trade_engine)
    
    # 10️⃣ Monitoring
    performance_tracker = PerformanceTracker()
    system_health = SystemHealth()
    trade_logger = TradeLogger()
    alert_system = AlertSystem(env_vars)
    
    # 11️⃣ Telegram
    telegram_bot = TelegramBot(env_vars)
    
    # 12️⃣ Başlat
    logger.info("Starting Binance1-Pro Bot...")
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(ws.connect())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually.")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
