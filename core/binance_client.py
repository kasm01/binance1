import logging
import os
from typing import Optional, Any

logger = logging.getLogger("system")

HAS_BINANCE = False
BINANCE_IMPORT_ERROR: Optional[Exception] = None

try:
    from binance.client import Client  # type: ignore
    from binance.enums import *  # type: ignore
    HAS_BINANCE = True
except Exception as e:
    HAS_BINANCE = False
    BINANCE_IMPORT_ERROR = e


# ==============================
# Futures Base URL Resolver
# ==============================

def _get_futures_base_url(is_testnet: bool) -> str:
    """
    USDT-M Futures endpoint belirler.
    - testnet: https://testnet.binancefuture.com
    - live:    https://fapi.binance.com
    """
    return "https://testnet.binancefuture.com" if is_testnet else "https://fapi.binance.com"


# ==============================
# Attach Close Helper
# ==============================

def _attach_close(client: Any, log: logging.Logger) -> Any:
    """
    python-binance Client objesine best-effort close() ekler.
    Amaç: requests.Session kapanışı (HTTP connection pool cleanup).
    """

    def _close():
        try:
            sess = getattr(client, "session", None)
            if sess is not None and hasattr(sess, "close"):
                sess.close()
                log.info("[BINANCE_CLIENT] session closed.")
        except Exception as e:
            log.debug("[BINANCE_CLIENT] session close failed: %s", e)

        try:
            if hasattr(client, "_session") and getattr(client, "_session") is not None:
                s2 = getattr(client, "_session")
                if hasattr(s2, "close"):
                    s2.close()
        except Exception:
            pass

    try:
        setattr(client, "close", _close)
    except Exception:
        pass

    return client


# ==============================
# Client Factory
# ==============================

def create_binance_client(
    api_key: Optional[str],
    api_secret: Optional[str],
    testnet: Optional[bool] = None,  # ✅ main.py uyumu için eklendi
    logger: Optional[logging.Logger] = None,
    dry_run: bool = True,
) -> Optional[Any]:
    """
    Binance USDT-M Futures REST client oluşturur.

    - python-binance yoksa:
        * DRY_RUN=True ise: client=None ile devam eder.
        * DRY_RUN=False ise: RuntimeError fırlatır.
    - Futures testnet/live seçimi:
        * testnet parametresi verilmişse onu kullanır
        * verilmemişse BINANCE_TESTNET env'inden okur
    """

    log = logger or logging.getLogger("system")

    if not HAS_BINANCE:
        msg = f"python-binance import edilemedi: {repr(BINANCE_IMPORT_ERROR)}"
        log.error("[BINANCE_CLIENT] %s", msg)

        if dry_run or not api_key or not api_secret:
            log.warning("[BINANCE_CLIENT] DRY_RUN veya API key yok; client=None ile devam.")
            return None

        raise RuntimeError("python-binance package not installed or failed to import")

    if not api_key or not api_secret:
        log.warning(
            "[BINANCE_CLIENT] API key/secret boş. DRY_RUN=%s, client=None ile devam.",
            dry_run,
        )
        return None

    # ✅ testnet kararı: parametre > env
    if testnet is None:
        is_testnet = str(os.getenv("BINANCE_TESTNET", "0")).strip().lower() in ("1", "true", "yes", "on")
    else:
        is_testnet = bool(testnet)

    # ⚠️ python-binance Client(..., testnet=...) spot içindir.
    # Futures için base url override yapıyoruz.
    client = Client(api_key, api_secret, testnet=False)

    futures_url = _get_futures_base_url(is_testnet)

    # ✅ python-binance futures URL override (USDT-M)
    try:
        client.FUTURES_URL = futures_url
    except Exception:
        pass
    try:
        client.FUTURES_DATA_URL = futures_url
    except Exception:
        pass

    client = _attach_close(client, log)

    log.info(
        "[BINANCE_CLIENT] Futures Client oluşturuldu | testnet=%s | url=%s | dry_run=%s",
        is_testnet,
        futures_url,
        dry_run,
    )
    return client
