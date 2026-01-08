import logging

logger = logging.getLogger('system')


# --- CREDENTIALS_NORMALIZED_BLOCK ---
# NOTE: Bu blok sadece helper + alias standardizasyonu sağlar.
# 'class Credentials' tanımı altta kalır.

import os
import logging

logger = logging.getLogger("system")


def _env(name: str):
    v = os.getenv(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _env_any(*names: str):
    for n in names:
        v = _env(n)
        if v is not None:
            return v
    return None


def _log_missing(mapping: dict, prefix: str = "[CREDENTIALS]") -> None:
    missing = []
    for k, v in (mapping or {}).items():
        if v is None or str(v).strip() == "":
            missing.append(k)

    if missing:
        logger.warning("%s Missing environment variables: %s", prefix, missing)
    else:
        logger.info("%s All tracked environment variables are present.", prefix)


def _apply_aliases_to_credentials() -> None:
    # Bu fonksiyon, class Credentials tanımlandıktan sonra çağrılır.
    try:
        cls = Credentials  # noqa: F821
    except Exception:
        return

    def set_if_missing(attr: str, value):
        try:
            if not hasattr(cls, attr):
                setattr(cls, attr, value)
        except Exception:
            pass

    # Exchanges (Binance / OKX)
    set_if_missing("BINANCE_API_KEY", _env_any("BINANCE_API_KEY"))
    set_if_missing("BINANCE_API_SECRET", _env_any("BINANCE_API_SECRET"))

    set_if_missing("OKX_API_KEY", _env_any("OKX_API_KEY"))
    set_if_missing("OKX_API_SECRET", _env_any("OKX_API_SECRET"))
    set_if_missing("OKX_PASSPHRASE", _env_any("OKX_PASSPHRASE"))

    # Onchain / Providers
    set_if_missing("ETHEREUM_API_KEY", _env_any("ETH_API_KEY", "ETHEREUM_API_KEY"))
    set_if_missing("ALCHEMY_ETH_API_KEY", _env_any("ALCHEMY_ETH_API_KEY"))
    set_if_missing("INFURA_API_KEY", _env_any("INFURA_API_KEY"))
    set_if_missing("POLYGON_API_KEY", _env_any("POLYGON_API_KEY"))
    set_if_missing("ARBITRUM_API_KEY", _env_any("ARBI_API_KEY", "ARBITRUM_API_KEY"))
    set_if_missing("THE_GRAPH_API_KEY", _env_any("THE_GRAPH_API_KEY", "GRAPH_API_KEY"))

    set_if_missing("COINGLASS_API_KEY", _env_any("COINGLASS_API_KEY"))
    set_if_missing("BSCSCAN_API_KEY", _env_any("BSCSCAN_API_KEY"))
    set_if_missing("CRYPTOQUANT_API_KEY", _env_any("CRYPTOQUANT_API_KEY"))
    set_if_missing("COINMARKETCAP_API_KEY", _env_any("COINMARKETCAP_API_KEY"))
    set_if_missing("ETHERSCAN_API_KEY", _env_any("ETHERSCAN_API_KEY"))
    set_if_missing("SANTIMENT_API_KEY", _env_any("SANTIMENT_API_KEY"))

    # log_missing: AttributeError yemesin diye getattr-safe
    def _safe_log_missing(prefix: str = "[CREDENTIALS]") -> None:
        tracked = {
            "BINANCE_API_KEY": getattr(cls, "BINANCE_API_KEY", None),
            "BINANCE_API_SECRET": getattr(cls, "BINANCE_API_SECRET", None),
            "OKX_API_KEY": getattr(cls, "OKX_API_KEY", None),
            "OKX_API_SECRET": getattr(cls, "OKX_API_SECRET", None),
            "OKX_PASSPHRASE": getattr(cls, "OKX_PASSPHRASE", None),

            "ETH_API_KEY": getattr(cls, "ETHEREUM_API_KEY", None),
            "ALCHEMY_ETH_API_KEY": getattr(cls, "ALCHEMY_ETH_API_KEY", None),
            "INFURA_API_KEY": getattr(cls, "INFURA_API_KEY", None),
            "POLYGON_API_KEY": getattr(cls, "POLYGON_API_KEY", None),
            "ARBI_API_KEY": getattr(cls, "ARBITRUM_API_KEY", None),
            "THE_GRAPH_API_KEY": getattr(cls, "THE_GRAPH_API_KEY", None),

            "COINGLASS_API_KEY": getattr(cls, "COINGLASS_API_KEY", None),
            "BSCSCAN_API_KEY": getattr(cls, "BSCSCAN_API_KEY", None),
            "CRYPTOQUANT_API_KEY": getattr(cls, "CRYPTOQUANT_API_KEY", None),
            "COINMARKETCAP_API_KEY": getattr(cls, "COINMARKETCAP_API_KEY", None),
            "ETHERSCAN_API_KEY": getattr(cls, "ETHERSCAN_API_KEY", None),
            "SANTIMENT_API_KEY": getattr(cls, "SANTIMENT_API_KEY", None),
        }
        _log_missing(tracked, prefix=prefix)

    # Class üzerinde varsa ezmeyelim (ama yoksa ekleyelim)
    if not hasattr(cls, "log_missing"):
        try:
            cls.log_missing = staticmethod(_safe_log_missing)  # type: ignore
        except Exception:
            pass


# Class tanımı aşağıda gelecek, import sırasında otomatik uygulansın:
# (Dosya en sonuna gelince tekrar çağıracağız.)


class Credentials:
    SANTIMENT_API_KEY = _env_any('SANTIMENT_API_KEY')
    ETHERSCAN_API_KEY = _env_any('ETHERSCAN_API_KEY')
    COINMARKETCAP_API_KEY = _env_any('COINMARKETCAP_API_KEY')
    CRYPTOQUANT_API_KEY = _env_any('CRYPTOQUANT_API_KEY')
    BSCSCAN_API_KEY = _env_any('BSCSCAN_API_KEY')
    ALCHEMY_ETH_API_KEY = _env_any('ALCHEMY_ETH_API_KEY')
    OKX_PASSPHRASE = _env_any('OKX_PASSPHRASE')
    OKX_API_SECRET = _env_any('OKX_API_SECRET')
    OKX_API_KEY = _env_any('OKX_API_KEY')
    BINANCE_API_SECRET = _env_any('BINANCE_API_SECRET')
    @staticmethod
    def log_missing(prefix: str = '[CREDENTIALS]') -> None:
        tracked = {
            'BINANCE_API_KEY': getattr(Credentials, 'BINANCE_API_KEY', None),
            'BINANCE_API_SECRET': getattr(Credentials, 'BINANCE_API_SECRET', None),
            'OKX_API_KEY': getattr(Credentials, 'OKX_API_KEY', None),
            'OKX_API_SECRET': getattr(Credentials, 'OKX_API_SECRET', None),
            'OKX_PASSPHRASE': getattr(Credentials, 'OKX_PASSPHRASE', None),

            'ETH_API_KEY': getattr(Credentials, 'ETHEREUM_API_KEY', None),
            'ALCHEMY_ETH_API_KEY': getattr(Credentials, 'ALCHEMY_ETH_API_KEY', None),
            'INFURA_API_KEY': getattr(Credentials, 'INFURA_API_KEY', None),
            'POLYGON_API_KEY': getattr(Credentials, 'POLYGON_API_KEY', None),
            'ARBI_API_KEY': getattr(Credentials, 'ARBITRUM_API_KEY', None),
            'THE_GRAPH_API_KEY': getattr(Credentials, 'THE_GRAPH_API_KEY', None),

            'COINGLASS_API_KEY': getattr(Credentials, 'COINGLASS_API_KEY', None),
            'BSCSCAN_API_KEY': getattr(Credentials, 'BSCSCAN_API_KEY', None),
            'CRYPTOQUANT_API_KEY': getattr(Credentials, 'CRYPTOQUANT_API_KEY', None),
            'COINMARKETCAP_API_KEY': getattr(Credentials, 'COINMARKETCAP_API_KEY', None),
            'ETHERSCAN_API_KEY': getattr(Credentials, 'ETHERSCAN_API_KEY', None),
            'SANTIMENT_API_KEY': getattr(Credentials, 'SANTIMENT_API_KEY', None),
        }
        _log_missing(tracked, prefix=prefix)

                missing.append(key)

        if missing:
            print(f"[WARNING] Missing critical credentials: {missing}")

