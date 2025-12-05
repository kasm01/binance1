import os
from typing import Dict, List, Tuple


def load_environment_variables() -> Tuple[Dict[str, str], List[str]]:
    """
    Ortam değişkenlerini ve eksik olan zorunlu değişkenleri döndürür.
    """
    env_vars: Dict[str, str] = dict(os.environ)

    required_keys: List[str] = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "SYMBOL",
        "INTERVAL",
    ]

    missing: List[str] = [k for k in required_keys if not env_vars.get(k)]

    # Sadece kısa bir uyarı (istersen bunu da kaldırabiliriz)
    if missing:
        print(f"[load_env] WARNING: Missing environment variables: {missing}")

    return env_vars, missing

