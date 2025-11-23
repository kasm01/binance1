import logging
from typing import Dict, Any, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config.credentials import Credentials
from core.exceptions import TradeExecutionException
from core.utils import retry

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Binance futures üzerinde emir açma / kapama işlemlerini yöneten sınıf.

    NOT:
      - API key/secret config.credentials içindeki Credentials sınıfından geliyor.
      - Hata durumunda TradeExecutionException fırlatıyor.
      - retry decorator'u ile geçici Binance hatalarına yeniden deneme uygulanıyor.
    """

    def __init__(self, env_vars: Optional[Dict[str, Any]] = None):
        # API key/secret artık Credentials üzerinden
        api_key = Credentials.BINANCE_API_KEY
        api_secret = Credentials.BINANCE_SECRET_KEY

        if not api_key or not api_secret:
            logger.error("[TradeExecutor] Binance API anahtarları eksik!")
            raise TradeExecutionException("Binance API credentials are missing")

        # Testnet kullanıyorsan:
        # self.client = Client(api_key, api_secret, testnet=True)
        self.client = Client(api_key, api_secret)

        # İleride ek ayarlar (timeout vb.) buraya konabilir
        logger.info("[TradeExecutor] Binance client initialized successfully")

    # -----------------------------------------------------
    # MARKET ORDER OLUŞTUR
    # -----------------------------------------------------
    @retry(
        exceptions=(BinanceAPIException, TradeExecutionException),
        tries=3,
        delay=2,
        backoff=2,
    )
    def create_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        position_side: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Futures market emri oluşturur.

        :param symbol: Örn: 'BTCUSDT'
        :param side: 'BUY' veya 'SELL'
        :param quantity: Emir miktarı
        :param position_side: 'LONG', 'SHORT' (hedge mode kullanıyorsan)
        :param reduce_only: Sadece pozisyon azaltma modu
        """
        try:
            logger.info(
                f"[TradeExecutor] Creating market order | "
                f"symbol={symbol}, side={side}, qty={quantity}, "
                f"position_side={position_side}, reduce_only={reduce_only}"
            )

            params: Dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
            }

            # Hedge mode vs. için positionSide
            if position_side:
                params["positionSide"] = position_side

            if reduce_only:
                params["reduceOnly"] = "true"

            order = self.client.futures_create_order(**params)

            logger.info(f"[TradeExecutor] Order created successfully: {order}")
            return order

        except BinanceAPIException as e:
            logger.error(f"[TradeExecutor] Binance API error: {e}")
            raise TradeExecutionException(f"Binance API error: {e}") from e
        except Exception as e:
            logger.exception(
                f"[TradeExecutor] Unknown error while creating order: {e}"
            )
            raise TradeExecutionException(
                f"Unknown trade execution error: {e}"
            ) from e

    # -----------------------------------------------------
    # POZİSYON KAPAT
    # -----------------------------------------------------
    @retry(
        exceptions=(BinanceAPIException, TradeExecutionException),
        tries=3,
        delay=2,
        backoff=2,
    )
    def close_position(
        self,
        symbol: str,
        quantity: float,
        position_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pozisyonu kapatmak için ters yönde market emri atar.

        :param symbol: Örn: 'BTCUSDT'
        :param quantity: Kapatılacak miktar
        :param position_side: 'LONG' veya 'SHORT'
        """
        try:
            # Basit örnek — gerçek senaryoda mevcut pozisyona göre belirlemelisin
            side = "SELL"

            logger.info(
                f"[TradeExecutor] Closing position | "
                f"symbol={symbol}, qty={quantity}, position_side={position_side}"
            )

            params: Dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "reduceOnly": "true",
            }

            if position_side:
                params["positionSide"] = position_side

            order = self.client.futures_create_order(**params)
            logger.info(f"[TradeExecutor] Close position order created: {order}")
            return order

        except BinanceAPIException as e:
            logger.error(f"[TradeExecutor] Binance API error (close_position): {e}")
            raise TradeExecutionException(
                f"Binance API error (close_position): {e}"
            ) from e
        except Exception as e:
            logger.exception(
                f"[TradeExecutor] Unknown error while closing position: {e}"
            )
            raise TradeExecutionException(
                f"Unknown close position error: {e}"
            ) from e


# ---------------------------------------------------------
# Basit Global Emir Fonksiyonu (fallback & multi-trade için)
# ---------------------------------------------------------

_executor_instance: Optional[TradeExecutor] = None


def get_executor() -> TradeExecutor:
    """
    Tek bir TradeExecutor instance'ı üretip tekrar tekrar kullanmak için.
    (fallback_trade ve MultiTradeEngine buradan faydalanıyor.)
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = TradeExecutor()
    return _executor_instance


def execute_trade(symbol: str, side: str, qty: float) -> Optional[Dict[str, Any]]:
    """
    MultiTradeEngine ve fallback_trade tarafından kullanılan basit emir fonksiyonu.

    :param symbol: 'BTCUSDT' vb.
    :param side: 'BUY' veya 'SELL'
    :param qty: emir miktarı (float)
    """
    executor = get_executor()

    try:
        order = executor.create_market_order(
            symbol=symbol,
            side=side,
            quantity=qty,
        )
        return order
    except Exception as e:
        logger.error(f"[execute_trade] Trade failed for {symbol} {side} {qty}: {e}")
        return None
