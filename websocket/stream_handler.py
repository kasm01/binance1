import json
from core.logger import system_logger
from trading.trade_executor import execute_trade

def handle_message(msg):
    """
    Gelen websocket mesajını ayrıştırır ve trade modülüne yönlendirir.
    """
    try:
        data = json.loads(msg)
        # Örnek: fiyat ve işlem sinyali al
        price = float(data['p'])
        symbol = data['s']
        signal = data.get('signal', None)
        
        system_logger.info(f"Message received: {symbol} - {price}")
        
        if signal:
            execute_trade(symbol, signal, price)
            
    except Exception as e:
        system_logger.error(f"Error handling message: {e}")
