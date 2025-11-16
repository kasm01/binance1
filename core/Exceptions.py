import logging
import sys
from typing import Type

logger = logging.getLogger(__name__)


class BotError(Exception):
    """Base class for all custom bot exceptions."""
    pass


# -----------------------------
# Config / Environment Errors
# -----------------------------
class ConfigError(BotError):
    """Raised when critical configuration or environment variables are missing."""
    pass


# -----------------------------
# Retry / Utility Errors
# -----------------------------
class RetryLimitExceeded(BotError):
    """Raised when a retry-decorated function exceeds maximum retries."""
    pass


# -----------------------------
# Data / ETL Errors
# -----------------------------
class DataProcessingException(BotError):
    """Raised when there is an error in data loading or feature engineering."""
    pass


# -----------------------------
# Model / ML Errors
# -----------------------------
class ModelTrainingException(BotError):
    """Raised when model training fails."""
    pass


class OnlineLearningException(BotError):
    """Raised when online learning update fails."""
    pass


class BatchLearningException(BotError):
    """Raised when batch learning / offline training fails."""
    pass


# -----------------------------
# Trading / Execution Errors
# -----------------------------
class TradeExecutionException(BotError):
    """Raised when trade execution on Binance fails."""
    pass


class StrategyException(BotError):
    """Raised when strategy engine encounters a critical error."""
    pass


class RiskException(BotError):
    """Raised when risk manager detects an invalid state."""
    pass


# -----------------------------
# External Services
# -----------------------------
class ExternalAPIException(BotError):
    """Raised when external APIs (Coinglass, The Graph, Infura, etc.) fail."""
    pass


class WebSocketException(BotError):
    """Raised when Binance WebSocket connection fails."""
    pass


class TelegramNotificationException(BotError):
    """Raised when Telegram notifications fail."""
    pass


class RedisCacheException(BotError):
    """Raised when Redis cache operations fail."""
    pass


# -----------------------------
# Global Exception Handler
# -----------------------------
class GlobalExceptionHandler:
    """
    Global exception handler for uncaught exceptions.
    main.py içinde GlobalExceptionHandler.register() ile aktif ediliyor.
    """

    @staticmethod
    def handle(exc_type: Type[BaseException], exc_value: BaseException, exc_traceback):
        # Ctrl+C vb. durumlarda sessiz geç
        if issubclass(exc_type, KeyboardInterrupt):
            logger.warning("[GlobalExceptionHandler] KeyboardInterrupt received.")
            return

        logger.error(
            "[GlobalExceptionHandler] Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    @classmethod
    def register(cls):
        """Set sys.excepthook to this handler."""
        sys.excepthook = cls.handle
        logger.info("[GlobalExceptionHandler] Registered global exception hook.")

