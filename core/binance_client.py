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
    Bazı python-binance sürümlerinde requests.Session.request'e `version`
    kwarg'ı sarkabiliyor ve şu hataya yol açıyor:

        Session.request() got an unexpected keyword argument 'version'

    En güvenli çözüm: session.request seviyesinde version'ı düşürmek.
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

        def _request_patched(method, url, **kwargs):
            try:
                kwargs.pop("version", None)
            except Exception:
                pass
            return original_request(method, url, **kwargs)

        setattr(sess, "request", _request_patched)
        log.info("[BINANCE_CLIENT] session.request patch applied (drops stray 'version' kwarg)")
    except Exception as e:
        log.warning("[BINANCE_CLIENT] session.request patch failed: %s", e)

    return client
def _patch_python_binance_request(client: Any, log: logging.Logger) -> Any:
    """
    Bazı python-binance / requests kombinasyonlarında request kwargs içine
    yanlışlıkla `version` sarkabiliyor ve şu hataya yol açıyor:

        Session.request() got an unexpected keyword argument 'version'

    Bu patch, client._request çağrısında version'ı güvenle temizler.
    """
    try:
        original_request = getattr(client, "_request", None)
        if not callable(original_request):
            return client

        def _request_patched(method, uri: str, signed: bool, force_params: bool = False, **kwargs):
            try:
                if "version" in kwargs:
                    kwargs.pop("version", None)
            except Exception:
                pass

            return original_request(method, uri, signed, force_params, **kwargs)

        setattr(client, "_request", _request_patched)

        log.info("[BINANCE_CLIENT] _request patch applied (drops stray 'version' kwarg)")
    except Exception as e:
        log.warning("[BINANCE_CLIENT] _request patch failed: %s", e)

    return client
def _patch_problematic_futures_methods(client: Any, log: logging.Logger) -> Any:
    """
    python-binance futures read methods bazı sürümlerde problemli olabiliyor.
    Bu yüzden account / balance / positionRisk okumalarını v2 endpoint'e zorlarız.
    """

    def _futures_account_v2(**params):
        return client._request_futures_api("get", "account", True, version=2, data=params)

    def _futures_account_balance_v2(**params):
        return client._request_futures_api("get", "balance", True, version=2, data=params)

    def _futures_position_information_v2(**params):
        return client._request_futures_api("get", "positionRisk", True, version=2, data=params)

    try:
        client.futures_account = _futures_account_v2
        client.futures_account_balance = _futures_account_balance_v2
        client.futures_position_information = _futures_position_information_v2
    except Exception as e:
        log.warning("[BINANCE_CLIENT] futures method patch failed: %s", e)

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

    try:
        log.info(
            "[BINANCE_CLIENT] Futures Client oluşturuldu | testnet=%s | dry_run=%s",
            is_testnet,
            dry_run,
        )
    except Exception:
        pass

    return client
