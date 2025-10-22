import logging
from core.utils import retry

def setup_logger(name, log_file, level=logging.INFO):
    """Logger oluşturur."""
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

# Örnek logger kullanımı
system_logger = setup_logger('system', 'logs/system.log')
error_logger = setup_logger('error', 'logs/errors.log')
trade_logger = setup_logger('trade', 'logs/trades.log')
