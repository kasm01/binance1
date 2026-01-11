import logging
from typing import Optional, Any

logger = logging.getLogger("system")

HAS_BINANCE = False
BINANCE_IMPORT_ERROR: Optional[Exception] = None

try:
    from binance.client import Client  # type: ignore
    from binance.enums import *  # type: ignore
    HAS_BINANCE = True
except Exception as e:  # ImportError dahil her şey
    HAS_BINANCE = False
    BINANCE_IMPORT_ERROR = e


def _attach_close(client: Any, log: logging.Logger) -> Any:
    """
    python-binance Client objesine best-effort close() ekler.
    Amaç: requests.Session kapanışı (HTTP connection pool cleanup).
    """

    def _close():
        # python-binance: client.session (requests.Session) genelde vardır
        try:
            sess = getattr(client, "session", None)
            if sess is not None and hasattr(sess, "close"):
                sess.close()
                log.info("[BINANCE_CLIENT] session closed.")
        except Exception as e:
            log.debug("[BINANCE_CLIENT] session close failed: %s", e)

        # bazen ek adapterler vs. olabilir
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
        # setattr olmayabilir ama çoğu durumda olur
        pass

    return client


def create_binance_client(
    api_key: Optional[str],
    api_secret: Optional[str],
    testnet: bool = False,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = True,
) -> Optional[Any]:
    """
    Binance REST client'ını oluşturur.
    - python-binance yoksa:
        * DRY_RUN=True ise: client=None ile devam eder (sadece log).
        * DRY_RUN=False ise: RuntimeError fırlatır.
    """
    log = logger or logging.getLogger("system")

    if not HAS_BINANCE:
        msg = f"python-binance import edilemedi: {repr(BINANCE_IMPORT_ERROR)}"
        log.error("[BINANCE_CLIENT] %s", msg)

        if dry_run or not api_key or not api_secret:
            log.warning(
                "[BINANCE_CLIENT] DRY_RUN veya API key yok; python-binance olmadan "
                "client=None ile devam edilecek."
            )
            return None

        raise RuntimeError("python-binance package not installed or failed to import")

    if not api_key or not api_secret:
        log.warning(
            "[BINANCE_CLIENT] API key/secret boş. DRY_RUN=%s, client=None ile devam.",
            dry_run,
        )
        return None

    client = Client(api_key, api_secret, testnet=testnet)
    client = _attach_close(client, log)

    log.info(
        "[BINANCE_CLIENT] Binance Client oluşturuldu (testnet=%s, dry_run=%s)",
        testnet,
        dry_run,
    )
    return client

