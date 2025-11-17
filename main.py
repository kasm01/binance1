import os
import asyncio
import logging

from aiohttp import web
import pandas as pd

from config.credentials import Credentials
from config.settings import Settings
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector


logger = logging.getLogger("binance1_pro_main")


async def run_data_pipeline(symbol: str, interval: str = "1m", limit: int = 500):
    """
    Bloklayıcı data + feature + anomali pipeline'ını ayrı fonksiyonda topladık.
    asyncio.to_thread ile çağıracağız ki event loop kilitlenmesin.
    """
    try:
        system_logger.info(f"[DATA] Fetching {limit} klines from Binance for {symbol} ({interval})")

        # DataLoader sync çalışıyor, o yüzden bu fonksiyon sync.
        data_loader = DataLoader(api_keys={})
        raw_df = data_loader.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)

        if raw_df is None or raw_df.empty:
            system_logger.warning("[DATA] Empty DataFrame returned from Binance.")
            return

        # Kolonları numeric tipe çevir
        for col in ["open", "high", "low", "close", "volume"]:
            raw_df[col] = raw_df[col].astype(float)

        # Zaman kolonu → datetime index (opsiyonel ama güzel)
        try:
            raw_df["open_time"] = pd.to_datetime(raw_df["open_time"], unit="ms")
            raw_df.set_index("open_time", inplace=True)
        except Exception as e:
            logger.warning(f"[DATA] Failed to set datetime index: {e}")

        system_logger.info(f"[DATA] Raw DF shape: {raw_df.shape}")

        # Feature engineering
        fe = FeatureEngineer(raw_data=raw_df)
        features_df = fe.transform()
        system_logger.info(f"[FE] Features DF shape: {features_df.shape}")

        # Anomali tespiti
        detector = AnomalyDetector(features_df=features_df)
        clean_df = detector.remove_anomalies()
        system_logger.info(
            f"[ANOM] Clean DF shape: {clean_df.shape} "
            f"(removed {len(features_df) - len(clean_df)} rows)"
        )

    except Exception as e:
        logger.exception(f"[PIPELINE] Error in data/feature/anomaly pipeline for {symbol}: {e}")


async def bot_loop():
    """
    Binance1-Pro botunun çekirdek döngüsü.
    Şu an:
      - Belirli aralıklarla Binance'ten veri çekiyor
      - Feature üretiyor
      - Anomali temizliği yapıyor
      - Heartbeat log atıyor
    Daha sonra buraya model + strategy + trade executor katmanlarını ekleyeceğiz.
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    # Ayarlardan sembol listesi al, boşsa fallback BTCUSDT
    symbols = Settings.TRADE_SYMBOLS or ["BTCUSDT"]
    symbol = symbols[0]
    interval = "1m"
    limit = 500

    while True:
        try:
            # Data pipeline'ı ayrı thread'de çalıştır (blocking fonksiyonlar için)
            await asyncio.to_thread(run_data_pipeline, symbol, interval, limit)

            system_logger.info("⏱ [BOT] Heartbeat - bot_loop running with data pipeline.")
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")
            break
        except Exception as e:
            logger.exception(f"[BOT] Unexpected error in bot_loop: {e}")

        # Binance'i çok sık dövmeyelim; 60 saniyede bir yeterli
        await asyncio.sleep(60)


async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text="OK")


async def ready_handler(request: web.Request) -> web.Response:
    return web.Response(text="READY")


async def start_background_tasks(app: web.Application):
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    app["bot_task"] = asyncio.create_task(bot_loop())


async def cleanup_background_tasks(app: web.Application):
    system_logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
    bot_task: asyncio.Task = app.get("bot_task")
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            system_logger.info("✅ [MAIN] bot_loop cancelled gracefully.")


async def create_app() -> web.Application:
    """
    Hem health endpoint'lerini hem de background bot'u yöneten aiohttp uygulaması.
    """
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


def main():
    """
    Cloud Run için entry point.
    """
    setup_logger("binance1_pro_entry")
    GlobalExceptionHandler.register()
    Credentials.validate()

    port = int(os.environ.get("PORT", "8080"))
    system_logger.info(f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{port} (ENV={Settings.ENV})")

    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
