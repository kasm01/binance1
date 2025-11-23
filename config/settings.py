# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Binance1-Pro tüm runtime ayarları.
    Hem lokal geliştirme hem de Google Cloud Run için optimize edildi.
    """

    # ───────────────────────────── Genel Proje Bilgisi ─────────────────────────────
    PROJECT_NAME: str = "Binance1-Pro"
    VERSION: str = "3.0.0"
    ENV: str = os.getenv("ENV", "production")

    # ───────────────────────────── Logging ─────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_PATH: str = os.getenv("LOG_PATH", "logs")  # Lokal için anlamlı; Cloud Run stdout kullanır

    # ───────────────────────────── Cloud Run ─────────────────────────────
    USE_CLOUD: bool = os.getenv("USE_CLOUD", "True").lower() in ("true", "1")
    CLOUD_LOGGING: bool = os.getenv("CLOUD_LOGGING", "True").lower() in ("true", "1")

    # Cloud Run için zorunlu: HTTP server portu
    PORT: int = int(os.getenv("PORT", 8080))

    # ───────────────────────────── Telegram ─────────────────────────────
    TELEGRAM_ALERTS: bool = os.getenv("TELEGRAM_ALERTS", "True").lower() in ("true", "1")

    # ───────────────────────────── Redis Cache ─────────────────────────────
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # ───────────────────────────── Binance Trading Ayarları ─────────────────────────────
    # İşlem yapılacak sembol (şimdilik tek coin)
    BINANCE_SYMBOL: str = os.getenv("BINANCE_SYMBOL", "BTCUSDT")

    # Kline veri aralığı (1m / 5m / 15m / 1h)
    BINANCE_INTERVAL: str = os.getenv("BINANCE_INTERVAL", "1m")

    # Kaç bar çekilecek?
    KLINES_LIMIT: int = int(os.getenv("KLINES_LIMIT", 500))

    # Paralel işlem açma limiti
    MAX_PARALLEL_TRADES: int = int(os.getenv("MAX_PARALLEL_TRADES", 3))

    # ───────────────────────────── Risk Yönetimi ─────────────────────────────
    # Her işlemde riske atılabilecek maksimum oran (%)
    # Örn: 0.02 → %2 risk
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", 0.02))

    # Günlük maksimum zarar limiti (equity yüzdesi)
    # Örn: 0.05 → %5 zarar sonrası sistem tüm gün durur
    MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", 0.05))

    # Stop-Loss yüzdesi
    # Örn: 0.01 → %1 stop-loss
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", 0.01))

    # Varsayılan kaldıraç
    DEFAULT_LEVERAGE: int = int(os.getenv("DEFAULT_LEVERAGE", 3))

    # ───────────────────────────── Sinyal Eşikleri ─────────────────────────────
    # p_buy >= BUY_THRESHOLD → BUY
    BUY_THRESHOLD: float = float(os.getenv("BUY_THRESHOLD", 0.60))

    # p_buy <= SELL_THRESHOLD → SELL
    SELL_THRESHOLD: float = float(os.getenv("SELL_THRESHOLD", 0.40))

    # HOLD aralığı: (SELL_THRESHOLD, BUY_THRESHOLD)

    # ───────────────────────────── Bot Loop ─────────────────────────────
    # Bot döngü bekleme süresi (saniye)
    MAIN_LOOP_SLEEP: int = int(os.getenv("MAIN_LOOP_SLEEP", 60))

    # DRY-RUN (gerçek emir yok) / LIVE TRADING
    LIVE_TRADING_ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "False").lower() in ("true", "1")



# Config sınıfını uygulamada daha kısa kullanmak için alias
Config = Settings()

