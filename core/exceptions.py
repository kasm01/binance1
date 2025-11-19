"""
core.exceptions

Bot içindeki tüm custom exception sınıfları burada toplanır.
"""

class BinanceBotException(Exception):
    """Tüm özel bot hataları için base exception."""
    pass


class ConfigException(BinanceBotException):
    """Konfigürasyon / environment değişkenleri ile ilgili hatalar."""
    pass


class DataLoadingException(BinanceBotException):
    """Veri yükleme (API, dosya, DB vb.) kaynaklı hatalar."""
    pass


class DataProcessingException(BinanceBotException):
    """Veri işleme, feature engineering, label creation vb. hatalar."""
    pass


class ModelTrainingException(BinanceBotException):
    """Batch veya offline model eğitimi sırasında oluşan hatalar."""
    pass


class OnlineLearningException(BinanceBotException):
    """Online / incremental öğrenme (SGDClassifier vb.) ile ilgili hatalar."""
    pass


class PredictionException(BinanceBotException):
    """Model tahmini (predict / predict_proba) aşamasındaki hatalar."""
    pass


class SignalGenerationException(BinanceBotException):
    """Trading sinyali üretimi (BUY/SELL/HOLD) sırasında oluşan hatalar."""
    pass


class PipelineException(BinanceBotException):
    """Genel pipeline akışı ile ilgili hatalar."""
    pass


class BinanceAPIException(BinanceBotException):
    """Binance API çağrıları sırasında oluşan hatalar."""
    pass


class CacheException(BinanceBotException):
    """Redis / cache yönetimi ile ilgili hatalar."""
    pass
