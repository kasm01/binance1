import unittest
from trading.trade_executor import TradeExecutor
from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager

class TestTrading(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(max_risk_per_trade=0.01)
        self.position = PositionManager()
        self.executor = TradeExecutor(risk_manager=self.risk, position_manager=self.position)

    def test_trade_execution(self):
        trade_result = self.executor.execute_trade(symbol="BTCUSDT", side="BUY", qty=0.001)
        self.assertIn(trade_result['status'], ['success', 'failed'])

if __name__ == "__main__":
    unittest.main()
