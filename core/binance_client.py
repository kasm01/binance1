import os
import logging
from typing import Optional

try:
    # python-binance v1
    from binance.um_futures import UMFutures
except ImportError:
    UMFutures = None  # python-binance yoksa graceful hata vereceğiz


def create_binance_client(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    testnet: Optional[bool] = None,
    logger: Optional[logging.Logger] = None,
    dry_run: Optional[bool] = None,
    **kwargs,
):
    """
    Projedeki main.py'den gelen çağrılarla uyumlu, esnek Binance futures client builder.

    Parametreler:
      - api_key / api_secret: verilmezse ENV'den okunur
      - testnet: verilmezse BINANCE_TESTNET env'ine bakar (true/false)
      - dry_run: verilmezse DRY_RUN env'ine bakar
    """

    log = logger or logging.getLogger("system")

    # python-binance yoksa anlamlı bir hata ver
    if UMFutures is None:
        log.error(
            "[BINANCE_CLIENT] python-binance yüklü değil. "
            "Lütfen `pip install python-binance` komutunu çalıştır."
        )
        raise RuntimeError("python-binance package not installed")

    # ENV fallback'leri
    if api_key is None:
        api_key = os.getenv("BINANCE_API_KEY", "")
    if api_secret is None:
        api_secret = os.getenv("BINANCE_API_SECRET", "")
    if testnet is None:
        testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    if dry_run is None:
        dry_run = os.getenv("DRY_RUN", "true").lower() == "true"

    if not api_key or not api_secret:
        log.warning(
            "[BINANCE_CLIENT] API key/secret boş görünüyor. DRY_RUN=%s, gerçek emir gönderimi kapalı olmalı.",
            dry_run,
        )

    # Base URL seçimi
    if testnet:
        base_url = "https://testnet.binancefuture.com"
    else:
        base_url = "https://fapi.binance.com"

    client = UMFutures(
        key=api_key,
        secret=api_secret,
        base_url=base_url,
    )

    log.info(
        "[BINANCE_CLIENT] UMFutures client oluşturuldu | testnet=%s | base_url=%s | dry_run=%s",
        testnet,
        base_url,
        dry_run,
    )

    # İleride istersen client içine dry_run flag'i ekleyebilirsin
    # ama şimdilik sadece loglarda tutuyoruz.
    client._dry_run = dry_run  # type: ignore[attr-defined]

    return client
