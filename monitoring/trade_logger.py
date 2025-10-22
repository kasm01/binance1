import logging

trade_logger = logging.getLogger('trade_logger')
trade_logger.setLevel(logging.INFO)
fh = logging.FileHandler('logs/trades.log')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
trade_logger.addHandler(fh)

def log_trade(trade_data):
    trade_logger.info(trade_data)
