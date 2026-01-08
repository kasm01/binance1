import os
from typing import Dict, List, Tuple, Optional


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def load_secrets_from_manager(
    keys: Optional[List[str]] = None,
    project_id: Optional[str] = None,
) -> None:
    """
    Secret Manager'dan secret'ları okuyup os.environ'a basar.
    - keys: Secret names (ENV isimleriyle birebir)
    - project_id: GCP project id (GCP_PROJECT / GOOGLE_CLOUD_PROJECT)
    """
    if not _bool("USE_CLOUD", False):
        return

    # hangi projeden okuyacağız?
    project = (
        project_id
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT")
    )
    if not project:
        # proje yoksa sessiz çık
        return

    # hangi secret isimlerini çekelim?
    if keys is None:
        keys = [
            # Exchanges
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "OKX_API_KEY",
            "OKX_API_SECRET",
            "OKX_PASSPHRASE",
            # Core
            "PG_DSN",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_ALLOWED_CHAT_IDS",
            # Providers / data
            "ETH_API_KEY",
            "ALCHEMY_ETH_API_KEY",
            "INFURA_API_KEY",
            "POLYGON_API_KEY",
            "ARBI_API_KEY",
            "THE_GRAPH_API_KEY",
            "COINGLASS_API_KEY",
            "BSCSCAN_API_KEY",
            "CRYPTOQUANT_API_KEY",
            "COINMARKETCAP_API_KEY",
            "ETHERSCAN_API_KEY",
            "SANTIMENT_API_KEY",
        ]

    try:
        from google.cloud import secretmanager  # type: ignore
    except Exception:
        # paket yoksa sessiz çık (istersen print ekleriz)
        return

    client = secretmanager.SecretManagerServiceClient()

    for k in keys:
        # env zaten doluysa ezme
        if os.getenv(k):
            continue
        try:
            name = f"projects/{project}/secrets/{k}/versions/latest"
            resp = client.access_secret_version(name=name)
            val = resp.payload.data.decode("utf-8").strip()
            if val:
                os.environ[k] = val
        except Exception:
            # Secret yoksa / yetki yoksa sessiz geç
            continue


def load_environment_variables() -> Tuple[Dict[str, str], List[str]]:
    """
    Ortam değişkenlerini ve eksik olan zorunlu değişkenleri döndürür.
    Secret Manager açıksa önce oradan env basar.
    """
    # 1) Secret Manager -> os.environ
    load_secrets_from_manager()

    # 2) Mevcut env snapshot
    env_vars: Dict[str, str] = dict(os.environ)

    required_keys: List[str] = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "SYMBOL",
        "INTERVAL",
    ]

    missing: List[str] = [k for k in required_keys if not env_vars.get(k)]

    if missing:
        print(f"[load_env] WARNING: Missing environment variables: {missing}")

    return env_vars, missing
