"""
core.exceptions

Tüm bot için kullanılan custom exception sınıfları.
Bu modüle yeni exception eklemek serbest; main.py ve diğer modüller buradan import eder.
"""


class BotBaseException(Exception):
    """Bot içerisindeki tüm özel exception'ların taban sınıfı."""
    pass


class ConfigException(BotBaseException):
    """Config / .env / ortam değişkenleri ile ilgili hatalar."""
    pass


class EnvironmentException(BotBaseException):
    """Çalışma ortamı ile ilgili hatalar (Cloud Run, Python versiyonu vs.)."""
    pass


class DataLoadingException(BotBaseException):
    """Veri yüklenirken (Binance, dış API vs.) oluşan hatalar."""
    pass


class DataProcessingException(BotBaseException):
    """Feature engineering, label üretimi vb. data pipeline hataları."""
    pass


class ModelTrainingException(BotBaseException):
    """Batch model (RandomForest vs.) eğitimi sırasında oluşan hatalar."""
    pass


class OnlineLearningException(BotBaseException):
    """Online/incremental öğrenme (SGDClassifier vb.) ile ilgili hatalar."""
    pass


class SignalGenerationException(BotBaseException):
    """Al/Sat sinyali üretimi sırasında oluşan hatalar."""
    pass


class CacheException(BotBaseException):
    """Redis / cache katmanı ile ilgili hatalar."""
    pass


class APIException(BotBaseException):
    """Dış API hataları (Binance, Telegram vb.)."""
    pass


class TradingException(BotBaseException):
    """Order açma, kapama, position yönetimi vb. trading hataları."""
    pass


class RiskManagementException(BotBaseException):
    """Risk yönetimi (limitler, max loss vs.) ile ilgili hatalar."""
    pass


class TelegramNotificationException(BotBaseException):
    """Telegram bildirimleri ile ilgili hatalar."""
    pass


class BacktestException(BotBaseException):
    """Backtest / performans ölçümü sırasında oluşan hatalar."""
    pass
