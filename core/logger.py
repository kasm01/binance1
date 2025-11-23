import logging
import sys

from config.settings import Settings


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Tek bir yerde logging formatını tanımla.
    Hem lokal hem Cloud Run için stdout'a log basar.
    """
    log_level = getattr(logging, Settings.LOG_LEVEL, logging.INFO)

    logger = logging.getLogger(name if name else Settings.PROJECT_NAME)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    # Root logger'a propagate etmesin (double log'ları engeller)
    logger.propagate = False
    return logger


# Sık kullanılan loggerlar
system_logger = setup_logger("system")
error_logger = setup_logger("error")
trade_logger = setup_logger("trade")
