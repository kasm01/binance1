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


def _attach_close(client: Any, log: logging.Logger) -> Any:
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


def _patch_session_request_drop_version(client: Any, log: logging.Logger) -> Any:
    """
    Bazı kombinasyonlarda requests.Session.request içine yanlışlıkla
    `version` kwarg'ı sarkabiliyor. Bunu sessizce düşürür.
    """
    try:
        sess = getattr(client, "session", None)
        if sess is None:
            log.warning("[BINANCE_CLIENT] session patch skipped: session missing")
            return client

        original_request = getattr(sess, "request", None)
        if not callable(original_request):
            log.warning("[BINANCE_CLIENT] session patch skipped: request missing")
            return client

        if getattr(sess, "_b1_version_patch_applied", False):
            log.info("[BINANCE_CLIENT] session.request patch already active")
            return client

        def _request_patched(method, url, **kwargs):
            kwargs.pop("version", None)
            return original_request(method, url, **kwargs)

        setattr(sess, "request", _request_patched)
        setattr(sess, "_b1_version_patch_applied", True)
        log.info("[BINANCE_CLIENT] session.request patch applied (drops stray 'version' kwarg)")
    except Exception as e:
        log.warning("[BINANCE_CLIENT] session.request patch failed: %s", e)

    return client


def _patch_futures_v2_read_methods(client: Any, log: logging.Logger) -> Any:
    """
    python-binance 1.0.16 içinde futures_account / futures_position_information /
    futures_account_balance varsayılan olarak FUTURES_API_VERSION üzerinden gider.
    Bu kurulumda v1 -> 404 verdiği için sadece read endpoint'lerini v2'ye zorluyoruz.

    order/leverage/marginType vb endpoint'lere dokunmuyoruz.
    """

    try:
        if getattr(client, "_b1_futures_v2_patch_applied", False):
            log.info("[BINANCE_CLIENT] futures v2 read patch already active")
            return client

        def _futures_request_v2(method: str, path: str, signed: bool = False, **kwargs):
            base = client.FUTURES_TESTNET_URL if getattr(client, "testnet", False) else client.FUTURES_URL
            uri = f"{base}/v2/{path}"
            return client._request(method, uri, signed, True, **kwargs)

        def _futures_account_v2(**params):
            return _futures_request_v2("get", "account", True, data=params)

        def _futures_account_balance_v2(**params):
            return _futures_request_v2("get", "balance", True, data=params)

        def _futures_position_information_v2(**params):
            return _futures_request_v2("get", "positionRisk", True, data=params)

        setattr(client, "futures_account", _futures_account_v2)
        setattr(client, "futures_account_balance", _futures_account_balance_v2)
        setattr(client, "futures_position_information", _futures_position_information_v2)
        setattr(client, "_b1_futures_v2_patch_applied", True)

        log.info("[BINANCE_CLIENT] patched futures read methods -> v2 endpoints")
    except Exception as e:
        log.warning("[BINANCE_CLIENT] futures v2 read patch failed: %s", e)

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
        client = Client(api_key, api_secret, testnet=is_testnet)
    except Exception as e:
        log.exception("[BINANCE_CLIENT] Client init failed: %s", e)
        if dry_run:
            return None
        raise

    client = _attach_close(client, log)
    client = _patch_session_request_drop_version(client, log)
    client = _patch_futures_v2_read_methods(client, log)

    try:
        log.info(
            "[BINANCE_CLIENT] Futures Client oluşturuldu | testnet=%s | dry_run=%s",
            is_testnet,
            dry_run,
        )
    except Exception:
        pass

    return client
