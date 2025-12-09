# core/binance_client.py

import logging
import os
from typing import Optional, Any

logger = logging.getLogger("system")

HAS_BINANCE = False
BINANCE_IMPORT_ERROR: Optional[Exception] = None

try:
    # python-binance paketi bu isimle import edilir
    from binance.client import Client  # type: ignore
    from binance.enums import *  # type: ignore
    HAS_BINANCE = True
except Exception as e:  # ImportError dahil her şey
    HAS_BINANCE = False
    BINANCE_IMPORT_ERROR = e


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

        # Gerçek trade modunda bu paketin kurulu olması şart
        raise RuntimeError("python-binance package not installed or failed to import")

    if not api_key or not api_secret:
        # Key yoksa gerçek client kurmayalım
        log.warning(
            "[BINANCE_CLIENT] API key/secret boş. DRY_RUN=%s, client=None ile devam.",
            dry_run,
        )
        return None

    # Buraya geldiysek python-binance var ve key/secret var
    client = Client(api_key, api_secret, testnet=testnet)
    log.info(
        "[BINANCE_CLIENT] Binance Client oluşturuldu (testnet=%s, dry_run=%s)",
        testnet,
        dry_run,
    )
    return client
