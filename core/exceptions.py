"""Custom exception classes for Binance1-Pro bot."""

from typing import Optional


class ConfigException(Exception):
    """Raised when configuration or environment variables are invalid or missing."""


class DataFetchException(Exception):
    """Raised when fetching data from external sources (e.g., Binance API) fails."""


class DataProcessingException(Exception):
    """Raised when cleaning, merging or transforming raw data fails."""


class FeatureEngineeringException(Exception):
    """Raised when feature engineering / indicator calculation fails."""


class LabelGenerationException(Exception):
    """Raised when label creation for supervised learning fails."""


class ModelTrainingException(Exception):
    """Raised when batch model (e.g. RandomForest) training or saving fails."""


class OnlineLearningException(Exception):
    """Raised when online learning model (SGD, etc.) training, updating or saving fails."""


class SignalGenerationException(Exception):
    """Raised when generating trading signals from model predictions fails."""


class RedisCacheException(Exception):
    """Raised when operations with Redis cache fail."""


class BinanceAPIException(Exception):
    """Raised when interacting with Binance REST/WebSocket API fails."""


class TelegramNotificationException(Exception):
    """Raised when sending Telegram notifications fails."""


class BotLoopException(Exception):
    """Top-level exception for unexpected errors in the main bot loop."""
