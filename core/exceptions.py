"""Custom exception classes for Binance1-Pro bot."""

class BinanceBotException(Exception):
    """Base exception for all Binance1-Pro bot related errors."""


# ---- Config & Environment -------------------------------------------------


class ConfigException(BinanceBotException):
    """Raised when configuration or environment variables are invalid or missing."""


# ---- Data Layer -----------------------------------------------------------


class DataFetchException(BinanceBotException):
    """Raised when fetching data from external sources (e.g., Binance API) fails."""


class DataLoadingException(BinanceBotException):
    """Raised when loading raw or cached data into DataFrames fails."""


class DataProcessingException(BinanceBotException):
    """Raised when cleaning, merging or transforming raw data fails."""


class FeatureEngineeringException(BinanceBotException):
    """Raised when feature engineering / indicator calculation fails."""


class LabelGenerationException(BinanceBotException):
    """Raised when label creation for supervised learning fails."""


# ---- Models & Training ----------------------------------------------------


class ModelTrainingException(BinanceBotException):
    """Raised when batch model (e.g. RandomForest) training or saving fails."""


class OnlineLearningException(BinanceBotException):
    """Raised when online learning model (SGD, etc.) training, updating or saving fails."""


class PredictionException(BinanceBotException):
    """Raised when generating or using model predictions fails."""


# ---- Signal & Trading -----------------------------------------------------


class SignalGenerationException(BinanceBotException):
    """Raised when generating trading signals from model predictions fails."""


class RedisCacheException(BinanceBotException):
    """Raised when operations with Redis cache fail."""


class BinanceAPIException(BinanceBotException):
    """Raised when interacting with Binance REST/WebSocket API fails."""


class TelegramNotificationException(BinanceBotException):
    """Raised when sending Telegram notifications fails."""


class TradeExecutionException(BinanceBotException):
    """Raised when trade execution (creating/closing orders) fails."""


# ---- Bot Loop & Retry -----------------------------------------------------


class BotLoopException(BinanceBotException):
    """Top-level exception for unexpected errors in the main bot loop."""


class RetryLimitExceeded(BinanceBotException):
    """Raised when the retry decorator exceeds the maximum retry attempts."""


# ---- Pipeline -------------------------------------------------------------


class PipelineException(BinanceBotException):
    """Raised for generic end-to-end pipeline failures."""

