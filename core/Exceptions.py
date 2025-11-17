import logging
import sys
import traceback


class RetryLimitExceeded(Exception):
    """
    Retry mekanizması belirlenen deneme sayısını aştığında fırlatılan hata.
    """
    pass


class Binance1ProError(Exception):
    """
    Binance1-Pro botuna özel genel hata sınıfı.
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
