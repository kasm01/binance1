import os
from dotenv import load_dotenv


def load_environment_variables(env_file: str = ".env") -> dict:
    """
    .env dosyasını (lokalde) ya da Cloud Run ortam değişkenlerini okuyup
    tek bir dict olarak döner.
    """
    # Lokal ortamdaysan .env'yi yükler, Cloud Run'da zaten env'den okur.
    if os.path.exists(env_file):
        load_dotenv(env_file)

    env_vars = {
        "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
        "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
        "COINGLASS_API_KEY": os.getenv("COINGLASS_API_KEY"),
        "GRAPH_API_KEY": os.getenv("GRAPH_API_KEY"),
        "ETH_API_KEY": os.getenv("ETH_API_KEY"),
        "ARBI_API_KEY": os.getenv("ARBI_API_KEY"),
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "REDIS_HOST": os.getenv("REDIS_HOST"),
        "REDIS_PORT": os.getenv("REDIS_PORT"),
        "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD"),
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID"),
        "ENV": os.getenv("ENV", "production"),
    }

    return env_vars

