import os
import asyncio
import logging

from aiohttp import web

# ------------------------------
# Core & Config
# ------------------------------
from config.credentials import Credentials
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


logger = logging.getLogger("binance1_pro_main")


# ------------------------------
# Bot Ana Döngüsü
# ------------------------------

async def run_bot():
    """
    Binance1-Pro botunun ana akışı.
    Cloud Run içinde arka planda çalışan, sonsuz döngülü görev.
    Hata aldığında bekleyip tekrar dener.
    """

    setup_logger("binance1_pro_bot")
    GlobalExceptionHandler.register()

    while True:
        try:
            logger.info("🔄 [BOT] Credentials doğrulanıyor...")
            Credentials.validate()

            # Env bilgilerini basitçe os.environ'dan alıyoruz
            env_vars = dict(os.environ)

            logger.info("✅ [BOT] Bileşenler başlatılıyor...")

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

            # Hyperparameter tuning (isteğe bağlı)
            tuner = HyperparameterTuner()
            tuned_model = tuner.tune(batch_model, clean_features)

            online_learner = OnlineLearner(tuned_model)

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

            # 9️⃣ WebSocket
            ws = BinanceWebSocket(env_vars, multi_trade_engine)

            # 10️⃣ Monitoring
            performance_tracker = PerformanceTracker()
            system_health = SystemHealth()
            trade_logger = TradeLogger()
            alert_system = AlertSystem(env_vars)

            # 11️⃣ Telegram
            telegram_bot = TelegramBot(env_vars)

            logger.info("🚀 [BOT] Binance1-Pro Bot WebSocket'e bağlanıyor...")
            await ws.connect()

            # Eğer ws.connect() dönerse (disconnect vs.), biraz bekleyip yeniden dene
            logger.warning("⚠️ [BOT] WebSocket bağlantısı sona erdi. 10 sn sonra tekrar denenecek.")
            await asyncio.sleep(10)

        except Exception as e:
            logger.exception(f"💥 [BOT] Kritik hata: {e}. 15 sn sonra tekrar denenecek.")
            await asyncio.sleep(15)


# ------------------------------
# Health / HTTP Server (Cloud Run için)
# ------------------------------

async def health_handler(request):
    """
    Cloud Run health check endpoint.
    Sadece 'OK' döner.
    """
    return web.Response(text="OK")


async def ready_handler(request):
    """
    Opsiyonel readiness endpoint.
    Şimdilik basit 'READY' cevabı döner.
    """
    return web.Response(text="READY")


async def start_background_bot(app: web.Application):
    """
    App startup sırasında bot görevini başlatır.
    """
    logger.info("🔁 [MAIN] Background bot görevi başlatılıyor...")
    app["bot_task"] = asyncio.create_task(run_bot())


async def stop_background_bot(app: web.Application):
    """
    App cleanup sırasında bot görevini durdurur.
    """
    logger.info("🛑 [MAIN] Background bot görevi durduruluyor...")
    bot_task = app.get("bot_task")
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            logger.info("✅ [MAIN] Bot görevi düzgün şekilde iptal edildi.")


async def create_app() -> web.Application:
    """
    Hem health endpoint'lerini hem de background bot'u yöneten aiohttp uygulaması.
    """
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(start_background_bot)
    app.on_cleanup.append(stop_background_bot)

    return app


def main():
    """
    Cloud Run için entry point.

    - PORT env değişkenini alır (Cloud Run bunu otomatik set eder, default: 8080)
    - aiohttp HTTP server'ı başlatır
    - Binance1-Pro botunu background task olarak çalıştırır
    """
    setup_logger("binance1_pro_entry")

    port = int(os.environ.get("PORT", "8080"))
    logger.info(f"🌐 [MAIN] HTTP server 0.0.0.0:{port} üzerinde başlatılıyor...")

    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

