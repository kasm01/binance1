"""
Custom exception classes for the Binance1-Pro bot.
Tüm modüllerde buradaki isimler import ediliyor.
"""

class EnvironmentConfigurationException(Exception):
    """Raised when required environment variables or configuration are missing or invalid."""
    pass


class DataValidationException(Exception):
    """Raised when raw or intermediate data fails validation checks."""
    pass


class FeatureEngineeringException(Exception):
    """Raised when feature generation / transformation fails."""
    pass


class LabelGenerationException(Exception):
    """Raised when target / label computation fails."""
    pass


class ModelTrainingException(Exception):
    """Raised when batch or online model training fails."""
    pass


class SignalGenerationException(Exception):
    """Raised when signal generation / decision logic fails."""
    pass


class OnlineLearningException(Exception):
    """Raised when online learning / partial fit fails."""
    pass


class BatchLearningException(Exception):
    """Raised when batch learning pipeline fails."""
    pass


class CacheException(Exception):
    """Raised for cache/Redis related errors."""
    pass


class APIFetchException(Exception):
    """Raised when fetching data from external APIs fails."""
    pass


class RedisConnectionException(Exception):
    """Raised when Redis cannot be reached or misconfigured."""
    pass


class RetryLimitExceeded(Exception):
    """
    Raised by retry helpers when a callable keeps failing after max retries.
    core.utils.retry bu class'a bağlı.
    """
    pass
