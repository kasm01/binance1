import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """
    General configuration and runtime settings.
    Adaptable for both local and Google Cloud environments.
    """

    PROJECT_NAME: str = "Binance1-Pro"
    VERSION: str = "2.0.0"
    ENV: str = os.getenv("ENV", "production")

    # Logging paths
    LOG_PATH = os.getenv("LOG_PATH", "logs/")
    ERROR_LOG = os.path.join(LOG_PATH, "errors.log")
    TRADE_LOG = os.path.join(LOG_PATH, "trades.log")
    SYSTEM_LOG = os.path.join(LOG_PATH, "system.log")

    # Database / Cache
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    # Trading settings
    TRADE_SYMBOLS = os.getenv("TRADE_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT").split(",")
    MAX_PARALLEL_TRADES = int(os.getenv("MAX_PARALLEL_TRADES", 3))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.02))
    STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 0.015))
    TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 0.03))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", 3))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", 2))

    # Google Cloud / Docker mode
    USE_CLOUD: bool = os.getenv("USE_CLOUD", "True").lower() in ("true", "1")
    CLOUD_LOGGING: bool = os.getenv("CLOUD_LOGGING", "True").lower() in ("true", "1")

    # Telegram notifications
    TELEGRAM_ALERTS = os.getenv("TELEGRAM_ALERTS", "True").lower() in ("true", "1")

    # Model paths
    MODEL_PATH = os.getenv("MODEL_PATH", "models/saved/")
    TUNED_MODEL_PATH = os.getenv("TUNED_MODEL_PATH", "models/tuned/")

    # API sources for backup
    API_FALLBACK_ORDER = ["Binance", "CoinGlass", "TheGraph", "Infura", "Polygon", "Arbitrum"]
