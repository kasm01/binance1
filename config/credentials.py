# config/credentials.py
import os
from dotenv import load_dotenv

load_dotenv()


class Credentials:
    """
    Tüm hassas credential'lar tek yerde.
    Hem local .env hem de Cloud Run env değişkenlerinden okur.
    """

    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    # .env.example'da BINANCE_API_SECRET kullanılmıştı, ikisini de destekleyelim
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET")

    # External Data APIs
    COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY")
    THE_GRAPH_API_KEY = os.getenv("THE_GRAPH_API_KEY") or os.getenv("GRAPH_API_KEY")
    INFURA_API_KEY = os.getenv("INFURA_API_KEY")
    ETHEREUM_API_KEY = os.getenv("ETH_API_KEY")
    ARBITRUM_API_KEY = os.getenv("ARBI_API_KEY") or os.getenv("ARBITRUM_API_KEY")
    POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Redis / Database
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    @classmethod
    def validate(cls) -> None:
        """
        Kritik credential'lar için basit doğrulama.
        Eksikler log ile uyarılır, istersek ileride Exception'a çevirebiliriz.
        """
        required_keys = [
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
        ]

        missing = []
        for key in required_keys:
            if getattr(cls, key, None) in (None, ""):
                missing.append(key)

        if missing:
            print(f"[WARNING] Missing critical credentials: {missing}")

