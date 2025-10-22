import os
from dotenv import load_dotenv

load_dotenv()

class Credentials:
    """
    Handles all API keys and sensitive data.
    Loaded from .env file for security.
    """

    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

    # External Data APIs
    COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")
    THE_GRAPH_API_KEY = os.getenv("THE_GRAPH_API_KEY")
    INFURA_API_KEY = os.getenv("INFURA_API_KEY")
    ETHEREUM_API_KEY = os.getenv("ETHEREUM_API_KEY")
    ARBITRUM_API_KEY = os.getenv("ARBITRUM_API_KEY")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Redis / Database
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    @classmethod
    def validate(cls):
        missing = [
            key for key, value in cls.__dict__.items()
            if not key.startswith("_") and value is None
        ]
        if missing:
            print(f"[WARNING] Missing credentials: {missing}")
