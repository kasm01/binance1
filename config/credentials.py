import os
import logging
from typing import Dict, Optional

logger = logging.getLogger("system")


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _env_any(*names: str) -> Optional[str]:
    for n in names:
        v = _env(n)
        if v is not None:
            return v
    return None


class Credentials:
    """
    Tek sözleşme:
    - Her şey os.environ'dan okunur.
    - refresh_from_env() çağrılınca class attribute'lar güncellenir.
    - log_missing() getattr-safe, AttributeError üretmez.
    """

    # --- Exchange keys ---
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None

    OKX_API_KEY: Optional[str] = None
    OKX_API_SECRET: Optional[str] = None
    OKX_PASSPHRASE: Optional[str] = None

    # --- Redis ---
    REDIS_PASSWORD: Optional[str] = None

    # --- Onchain / Providers ---
    ETHEREUM_API_KEY: Optional[str] = None   # alias: ETH_API_KEY
    ALCHEMY_ETH_API_KEY: Optional[str] = None
    INFURA_API_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None
    ARBITRUM_API_KEY: Optional[str] = None   # alias: ARBI_API_KEY
    THE_GRAPH_API_KEY: Optional[str] = None  # alias: GRAPH_API_KEY

    # --- Data providers ---
    COINGLASS_API_KEY: Optional[str] = None
    BSCSCAN_API_KEY: Optional[str] = None
    CRYPTOQUANT_API_KEY: Optional[str] = None
    COINMARKETCAP_API_KEY: Optional[str] = None
    ETHERSCAN_API_KEY: Optional[str] = None
    SANTIMENT_API_KEY: Optional[str] = None

    # --- Telegram (opsiyonel) ---
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_ALLOWED_CHAT_IDS: Optional[str] = None

    @staticmethod
    def refresh_from_env() -> None:
        """
        os.environ -> Credentials.* alanlarını yeniden doldurur.
        Secret Manager'dan env'e basma işi başka modülde yapılır;
        bu fonksiyon sadece env'den okur.
        """
        cls = Credentials

        # Exchange
        cls.BINANCE_API_KEY = _env_any("BINANCE_API_KEY")
        cls.BINANCE_API_SECRET = _env_any("BINANCE_API_SECRET")

        cls.OKX_API_KEY = _env_any("OKX_API_KEY")
        cls.OKX_API_SECRET = _env_any("OKX_API_SECRET")
        cls.OKX_PASSPHRASE = _env_any("OKX_PASSPHRASE")

        # Redis
        cls.REDIS_PASSWORD = _env_any("REDIS_PASSWORD")

        # Onchain/providers
        cls.ETHEREUM_API_KEY = _env_any("ETH_API_KEY", "ETHEREUM_API_KEY")
        cls.ALCHEMY_ETH_API_KEY = _env_any("ALCHEMY_ETH_API_KEY")
        cls.INFURA_API_KEY = _env_any("INFURA_API_KEY")
        cls.POLYGON_API_KEY = _env_any("POLYGON_API_KEY")
        cls.ARBITRUM_API_KEY = _env_any("ARBI_API_KEY", "ARBITRUM_API_KEY")
        cls.THE_GRAPH_API_KEY = _env_any("THE_GRAPH_API_KEY", "GRAPH_API_KEY")

        # Data providers
        cls.COINGLASS_API_KEY = _env_any("COINGLASS_API_KEY")
        cls.BSCSCAN_API_KEY = _env_any("BSCSCAN_API_KEY")
        cls.CRYPTOQUANT_API_KEY = _env_any("CRYPTOQUANT_API_KEY")
        cls.COINMARKETCAP_API_KEY = _env_any("COINMARKETCAP_API_KEY")
        cls.ETHERSCAN_API_KEY = _env_any("ETHERSCAN_API_KEY")
        cls.SANTIMENT_API_KEY = _env_any("SANTIMENT_API_KEY")

        # Telegram
        cls.TELEGRAM_BOT_TOKEN = _env_any("TELEGRAM_BOT_TOKEN")
        cls.TELEGRAM_ALLOWED_CHAT_IDS = _env_any("TELEGRAM_ALLOWED_CHAT_IDS")

    @staticmethod
    def log_missing(prefix: str = "[CREDENTIALS]") -> None:
        cls = Credentials

        tracked: Dict[str, Optional[str]] = {
            # Exchange
            "BINANCE_API_KEY": getattr(cls, "BINANCE_API_KEY", None),
            "BINANCE_API_SECRET": getattr(cls, "BINANCE_API_SECRET", None),
            "OKX_API_KEY": getattr(cls, "OKX_API_KEY", None),
            "OKX_API_SECRET": getattr(cls, "OKX_API_SECRET", None),
            "OKX_PASSPHRASE": getattr(cls, "OKX_PASSPHRASE", None),

            # Redis
            "REDIS_PASSWORD": getattr(cls, "REDIS_PASSWORD", None),

            # Onchain/providers
            "ETH_API_KEY": getattr(cls, "ETHEREUM_API_KEY", None),
            "ALCHEMY_ETH_API_KEY": getattr(cls, "ALCHEMY_ETH_API_KEY", None),
            "INFURA_API_KEY": getattr(cls, "INFURA_API_KEY", None),
            "POLYGON_API_KEY": getattr(cls, "POLYGON_API_KEY", None),
            "ARBI_API_KEY": getattr(cls, "ARBITRUM_API_KEY", None),
            "THE_GRAPH_API_KEY": getattr(cls, "THE_GRAPH_API_KEY", None),

            # Providers
            "COINGLASS_API_KEY": getattr(cls, "COINGLASS_API_KEY", None),
            "BSCSCAN_API_KEY": getattr(cls, "BSCSCAN_API_KEY", None),
            "CRYPTOQUANT_API_KEY": getattr(cls, "CRYPTOQUANT_API_KEY", None),
            "COINMARKETCAP_API_KEY": getattr(cls, "COINMARKETCAP_API_KEY", None),
            "ETHERSCAN_API_KEY": getattr(cls, "ETHERSCAN_API_KEY", None),
            "SANTIMENT_API_KEY": getattr(cls, "SANTIMENT_API_KEY", None),

            # Telegram
            "TELEGRAM_BOT_TOKEN": getattr(cls, "TELEGRAM_BOT_TOKEN", None),
            "TELEGRAM_ALLOWED_CHAT_IDS": getattr(cls, "TELEGRAM_ALLOWED_CHAT_IDS", None),
        }

        missing = [k for k, v in tracked.items() if v is None or str(v).strip() == ""]
        if missing:
            logger.warning("%s Missing environment variables: %s", prefix, missing)
        else:
            logger.info("%s All tracked environment variables are present.", prefix)


# Import sırasında env'den oku (ilk state)
Credentials.refresh_from_env()
