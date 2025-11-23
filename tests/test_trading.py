# tests/test_trading.py
import unittest

from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager


class TestTrading(unittest.TestCase):
    def setUp(self) -> None:
        self.risk = RiskManager(max_risk_per_trade=0.01)
        self.positions = PositionManager()

    def test_risk_manager_allows_small_position(self):
        capital = 1000.0
        position_qty = 5.0  # %0.5 risk -> 0.005
        allowed = self.risk.check_risk(capital=capital, position_qty=position_qty)
        self.assertTrue(allowed)

    def test_risk_manager_blocks_big_position(self):
        capital = 1000.0
        position_qty = 100.0  # %10 risk -> 0.1
        allowed = self.risk.check_risk(capital=capital, position_qty=position_qty)
        self.assertFalse(allowed)

    def test_position_open_and_close_long(self):
        pos_id = self.positions.open_position(
            symbol="BTCUSDT",
            qty=0.01,
            side="LONG",
            price=50000.0,
        )
        self.assertIn(pos_id, self.positions.get_open_positions())

        pnl = self.positions.close_position(pos_id, exit_price=50500.0)
        self.assertIsNotNone(pnl)
        # LONG pozisyon, fiyat yükseldi -> PnL pozitif olmalı
        self.assertGreater(pnl, 0.0)

    def test_position_open_and_close_short(self):
        pos_id = self.positions.open_position(
            symbol="BTCUSDT",
            qty=0.01,
            side="SHORT",
            price=50000.0,
        )
        self.assertIn(pos_id, self.positions.get_open_positions())

        pnl = self.positions.close_position(pos_id, exit_price=49500.0)
        self.assertIsNotNone(pnl)
        # SHORT pozisyon, fiyat düştü -> PnL pozitif olmalı
        self.assertGreater(pnl, 0.0)


if __name__ == "__main__":
    unittest.main()

