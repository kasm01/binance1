import os
import re
from dotenv import load_dotenv

# .env yükle
load_dotenv(dotenv_path=".env")


class Settings:
    """
    Tüm config değerlerini .env üzerinden okuyan basit ayar sınıfı.
    Hem class attribute hem instance üzerinden erişilebilir.

    Kullanım:
        from config.settings import Config, Settings

        symbol = Config.BINANCE_SYMBOL
        sleep_s = Config.MAIN_LOOP_SLEEP
    """

    # ───────────── Binance temel ayarları ─────────────
    BINANCE_SYMBOL: str = os.getenv("BINANCE_SYMBOL", os.getenv("SYMBOL", "BTCUSDT"))
    BINANCE_INTERVAL: str = os.getenv("BINANCE_INTERVAL", os.getenv("INTERVAL", "1m"))
    KLINES_LIMIT: int = int(os.getenv("KLINES_LIMIT", "500"))

    # Eski kod geriye dönük uyumluluk için
    SYMBOL: str = BINANCE_SYMBOL
    INTERVAL: str = BINANCE_INTERVAL

    # ───────────── Risk yönetimi ─────────────
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.01"))
    DEFAULT_LEVERAGE: int = int(os.getenv("DEFAULT_LEVERAGE", "3"))

    # ───────────── Sinyal eşikleri ─────────────
    BUY_THRESHOLD: float = float(os.getenv("BUY_THRESHOLD", "0.60"))
    SELL_THRESHOLD: float = float(os.getenv("SELL_THRESHOLD", "0.40"))

    # ───────────── Bot çalışma ayarları ─────────────
    LIVE_TRADING_ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "False").lower() == "true"
    MAX_PARALLEL_TRADES: int = int(os.getenv("MAX_PARALLEL_TRADES", "3"))
    MAIN_LOOP_SLEEP: int = int(os.getenv("MAIN_LOOP_SLEEP", "60"))

    # ───────────── PAPER TRADING ayarları ─────────────
    # Gerçek Binance equity çekilemezse / paper modda çalışmak istersek kullanılacak
    PAPER_TRADING_ENABLED: bool = os.getenv("PAPER_TRADING_ENABLED", "True").lower() == "true"
    PAPER_EQUITY_USDT: float = float(os.getenv("PAPER_EQUITY_USDT", "1000"))

    # İstersen ileride komisyon simülasyonu için:
    PAPER_FEE_RATE: float = float(os.getenv("PAPER_FEE_RATE", "0.0004"))  # %0.04

    # ───────────── Redis ayarları ─────────────
    REDIS_HOST: str = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str | None = os.getenv("REDIS_PASSWORD") or None

    # ───────────── Telegram ayarları ─────────────
    TELEGRAM_BOT_TOKEN: str | None = os.getenv("TELEGRAM_BOT_TOKEN") or None
TELEGRAM_CHAT_ID: str | None = os.getenv("TELEGRAM_CHAT_ID") or None
# Fallback: .env'e chat_id koymadan çalışmak için allowed ids'den default seç
if not TELEGRAM_CHAT_ID:
    _allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS") or ""
    _parts = [x.strip() for x in re.split(r"[,\s]+", _allowed) if x.strip()]
    if _parts:
        TELEGRAM_CHAT_ID = _parts[0]

    # ───────────── Cloud / Logging ─────────────
    USE_CLOUD: bool = os.getenv("USE_CLOUD", "True").lower() == "true"
    CLOUD_LOGGING: bool = os.getenv("CLOUD_LOGGING", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Kolaylık için alias
Config = Settings

