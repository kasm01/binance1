import os
import asyncio
import logging

from aiohttp import web

from config.credentials import Credentials
from config.settings import Settings
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler


logger = logging.getLogger("binance1_pro_main")


async def bot_loop():
    """
    Binance1-Pro botunun çekirdek döngüsü.
    Şimdilik sadece heartbeat log'u atıyor.
    Gerçek data/model/trade pipeline'ını daha sonra buraya parça parça ekleyeceğiz.
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    while True:
        try:
            system_logger.info("⏱ [BOT] Heartbeat - bot_loop running...")
            await asyncio.sleep(15)
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")
            break
        except Exception as e:
            logger.exception(f"[BOT] Unexpected error in bot_loop: {e}")
            await asyncio.sleep(10)


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
