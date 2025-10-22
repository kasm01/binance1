class BinanceAPIException(Exception):
    """Binance API ile ilgili özel hata."""
    pass

class DataProcessingException(Exception):
    """Veri işleme sırasında oluşan hata."""
    pass

class TradeExecutionException(Exception):
    """Trade işlemlerinde oluşan hata."""
    pass

class RetryLimitExceeded(Exception):
    """Retry denemesi maksimum sınırı aştığında fırlatılır."""
    pass
