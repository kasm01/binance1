import logging
import os
from typing import Optional, Any

logger = logging.getLogger("system")

HAS_BINANCE = False
BINANCE_IMPORT_ERROR: Optional[Exception] = None

try:
    from binance.client import Client  # type: ignore
    HAS_BINANCE = True
except Exception as e:
    HAS_BINANCE = False
    BINANCE_IMPORT_ERROR = e


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_futures_data_url(is_testnet: bool) -> str:
    """
    Futures data endpoint root:
    - testnet: https://demo-fapi.binance.com/futures/data
    - live   : https://fapi.binance.com/futures/data
    """
    return "https://demo-fapi.binance.com/futures/data" if is_testnet else "https://fapi.binance.com/futures/data"

def _attach_close(client: Any, log: logging.Logger) -> Any:
    """
    requests session cleanup helper
    """

    def _close():
        try:
            sess = getattr(client, "session", None)
            if sess and hasattr(sess, "close"):
                sess.close()
                log.info("[BINANCE_CLIENT] session closed")
        except Exception as e:
            log.debug("[BINANCE_CLIENT] close error: %s", e)

    try:
        setattr(client, "close", _close)
    except Exception:
        pass

    return client


def create_binance_client(
    api_key: Optional[str],
    api_secret: Optional[str],
    testnet: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = True,
) -> Optional[Any]:

    log = logger or logging.getLogger("system")

    if not HAS_BINANCE:
        msg = f"python-binance import edilemedi: {repr(BINANCE_IMPORT_ERROR)}"
        log.error("[BINANCE_CLIENT] %s", msg)

        if dry_run:
            return None

        raise RuntimeError(msg)

    if not api_key or not api_secret:
        log.warning("[BINANCE_CLIENT] API key/secret missing")
        return None

    if testnet is None:
        is_testnet = _env_bool("BINANCE_TESTNET") or _env_bool("USE_TESTNET")
    else:
        is_testnet = bool(testnet)

    try:
        client = Client(api_key, api_secret)
    except Exception as e:
        log.exception("[BINANCE_CLIENT] Client init failed: %s", e)
        if dry_run:
            return None
        raise

    futures_url = _get_futures_base_url(is_testnet)

    # IMPORTANT
    # python-binance futures endpoint override
    client.FUTURES_URL = futures_url

    client = _attach_close(client, log)

    log.info(
        "[BINANCE_CLIENT] Futures Client oluşturuldu | testnet=%s | url=%s | dry_run=%s",
        is_testnet,
        futures_url,
        dry_run,
    )

    return client
