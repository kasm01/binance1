import os
from typing import Dict


def load_environment_variables() -> Dict[str, str]:
    """
    Cloud Run / lokal ortamdan environment variable'ları okuyup
    bir dict olarak döner. Varsayılan değerler de burada tanımlı.

    main.py içinde:
        ENV_VARS = load_environment_variables()
    şeklinde kullanıyoruz.
    """

    defaults: Dict[str, str] = {
        # Veri ile ilgili
        "SYMBOL": "BTCUSDT",
        "INTERVAL": "1m",
        "HISTORY_LIMIT": "1000",

        # Label ve feature ile ilgili
        "LABEL_HORIZON": "10",

        # Model ile ilgili
        "MODEL_DIR": "models",
        "BATCH_MODEL_NAME": "batch_model",
        "ONLINE_MODEL_NAME": "online_model",

        # Sinyal eşiği
        "BUY_THRESHOLD": "0.6",
        "SELL_THRESHOLD": "0.4",

        # Bot loop
        "BOT_LOOP_INTERVAL": "60",

        # Ortam
        "ENVIRONMENT": "production",
    }

    env: Dict[str, str] = {}
    for key, default_val in defaults.items():
        env[key] = os.getenv(key, default_val)

    return env
