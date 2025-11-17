import logging
import sys


class Binance1ProError(Exception):
    """
    Binance1-Pro botuna özel genel hata sınıfı.
    """
    pass


class DataProcessingException(Binance1ProError):
    """
    Veri yükleme, temizleme, feature engineering sırasında oluşan hatalar.
    """
    pass


class APIRequestException(Binance1ProError):
    """
    Harici API (Binance, CoinGlass, The Graph, Infura vb.) istek hataları.
    """
    pass


class WebSocketConnectionError(Binance1ProError):
    """
    WebSocket bağlantı sorunları için.
    """
    pass


class TradingLogicException(Binance1ProError):
    """
    Strateji / trade yürütme sırasında oluşan mantık hataları için.
    """
    pass


class ConfigValidationException(Binance1ProError):
    """
    Eksik veya hatalı konfigürasyon / credential durumları için.
    """
    pass


class RetryLimitExceeded(Binance1ProError):
    """
    Retry mekanizması belirlenen deneme sayısını aştığında fırlatılan hata.
    """
    pass


class GlobalExceptionHandler:
    """
    Uygulama seviyesinde yakalanmayan tüm exception'ları loglamak için
    global exception handler.
    """

    @staticmethod
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Ctrl+C (KeyboardInterrupt) için default davranışı bozmuyoruz
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.critical(
            "💥 Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    @classmethod
    def register(cls):
        """
        Global exception handler'ı aktif eder.
        main.py içinde GlobalExceptionHandler.register() çağrılıyor.
        """
        sys.excepthook = cls.handle_exception
        logging.getLogger(__name__).info("✅ Global exception handler registered.")
