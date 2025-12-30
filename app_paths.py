# app_paths.py
import os

def env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if isinstance(v, str) and v.strip() else default

# ✅ Single source of truth
MODELS_DIR = env_str("MODELS_DIR", "models")
DATA_DIR   = env_str("DATA_DIR", "data/offline_cache")
SYMBOL     = env_str("SYMBOL", "BTCUSDT")
