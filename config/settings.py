# config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Genel runtime ayarları.
    Hem lokal geliştirme hem de Google Cloud Run için uyumlu.
    """

    PROJECT_NAME: str = "Binance1-Pro"
    VERSION: str = "3.0.0"
    ENV: str = os.getenv("ENV", "production")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Logs klasörü sadece lokal için anlamlı, Cloud Run stdout kullanacak.
    LOG_PATH = os.getenv("LOG_PATH", "logs")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    # Trading genel ayarlar (ileride kullanacağız)
    TRADE_SYMBOLS = os.getenv("TRADE_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
    MAX_PARALLEL_TRADES = int(os.getenv("MAX_PARALLEL_TRADES", 3))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.02))

    # Cloud Run / Logging
    USE_CLOUD: bool = os.getenv("USE_CLOUD", "True").lower() in ("true", "1")
    CLOUD_LOGGING: bool = os.getenv("CLOUD_LOGGING", "True").lower() in ("true", "1")

    # Telegram notifications
    TELEGRAM_ALERTS = os.getenv("TELEGRAM_ALERTS", "True").lower() in ("true", "1")

