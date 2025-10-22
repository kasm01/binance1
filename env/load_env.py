import os
from dotenv import load_dotenv

def load_environment_variables(env_file=".env"):
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
    }
    return env_vars
