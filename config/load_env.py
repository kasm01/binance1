import os
from typing import Dict


def load_environment_variables() -> Dict[str, str]:
    """
    Ortam değişkenlerini sözlük olarak döndürür.
    İleride istersen .env dosyası okuma, ekstra kontroller vs ekleyebiliriz.
    """
    env_vars = dict(os.environ)

    # Örnek: zorunlu değişken liste kontrolü (şimdilik sadece log amaçlı)
    required_keys = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "SYMBOL",
        "INTERVAL",
    ]

    missing = [k for k in required_keys if not env_vars.get(k)]
    if missing:
        # Burada direk print kullanıyoruz; logger import edip döngü yaratmayalım
        print(f"[load_env] WARNING: Missing environment variables: {missing}")

    return env_vars
