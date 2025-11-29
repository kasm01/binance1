import logging
import os
from logging.handlers import RotatingFileHandler

# Log klasörü
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Global logger referansları
system_logger = logging.getLogger("system")
error_logger = logging.getLogger("error")
trade_logger = logging.getLogger("trades")


def _create_file_handler(filename: str, level: int) -> RotatingFileHandler:
    """
    Rotating file handler (1 MB, 3 backup).
    """
    file_path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(
        file_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    return handler


def setup_logger(log_level: str = "INFO") -> None:
    """
    Tüm loggerları (system, error, trades) ayarlayan fonksiyon.
    main.py ve diğer modüller burayı çağırıyor.
    Idempotent: Birden fazla çağrı duplicate handler eklemez.
    """
    # Eğer daha önce kurulmuşsa tekrar handler ekleme
    if getattr(system_logger, "_initialized", False):
        return

    # Root log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)

    # Ortak console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    # system logger
    system_logger.setLevel(level)
    system_logger.propagate = False
    system_logger.addHandler(console_handler)
    system_logger.addHandler(_create_file_handler("system.log", level))

    # error logger (sadece ERROR ve üstü)
    error_logger.setLevel(logging.ERROR)
    error_logger.propagate = False
    error_logger.addHandler(console_handler)
    error_logger.addHandler(_create_file_handler("errors.log", logging.ERROR))

    # trade logger
    trade_logger.setLevel(level)
    trade_logger.propagate = False
    trade_logger.addHandler(console_handler)
    trade_logger.addHandler(_create_file_handler("trades.log", level))

    # Bir daha kurulmasın diye flag
    system_logger._initialized = True
    system_logger.info("[LOGGER] Loggers initialized (system, error, trades).")

