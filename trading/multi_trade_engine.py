from trading.trade_executor import execute_trade

class MultiTradeEngine:
    def __init__(self):
        self.max_trades = 3

    def execute_trades(self, trades):
        """
        trades: [{'symbol':'BTCUSDT', 'side':'BUY', 'qty':0.01}, ...]
        """
        for i, trade in enumerate(trades):
            if i >= self.max_trades:
                break
            execute_trade(trade['symbol'], trade['side'], trade['qty'])
