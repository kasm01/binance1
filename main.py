# main.py
import os
import asyncio
import logging

from aiohttp import web

from config.credentials import Credentials
from config.settings import Settings
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler
from core.cache_manager import CacheManager


logger = logging.getLogger("binance1_pro_main")


async def bot_worker():
    """
    Binance1-Pro çekirdek bot döngüsü.
    Aşama 1: sadece heartbeat + basit Redis kontrolü.
    Aşama 2-3'te buraya veri, model ve trading eklenecek.
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core worker started.")

    cache = CacheManager()
    cache.set("binance1:heartbeat", "alive", ex=30)

    while True:
        try:
            system_logger.info("⏱ [BOT] Heartbeat - bot worker running...")
            await asyncio.sleep(15)
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] Worker cancelled, shutting down.")
            break
        except Exception as e:
            logger.exception(f"[BOT] Unexpected error in worker: {e}")
            await asyncio.sleep(10)


async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text="OK")


async def ready_handler(request: web.Request) -> web.Response:
    return web.Response(text="READY")


async def start_background_tasks(app: web.Application):
    system_logger.info("🔁 [MAIN] Starting background bot worker...")
    app["bot_task"] = asyncio.create_task(bot_worker())


async def cleanup_background_tasks(app: web.Application):
    system_logger.info("🧹 [MAIN] Cleaning up background bot worker...")
    bot_task: asyncio.Task = app.get("bot_task")
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            system_logger.info("✅ [MAIN] Bot worker cancelled gracefully.")


async def create_app() -> web.Application:
    setup_logger("binance1_pro")
    GlobalExceptionHandler.register()

    Credentials.validate()

    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


def main():
    port = int(os.environ.get("PORT", "8080"))
    system_logger.info(f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{port} (ENV={Settings.ENV})")
    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
