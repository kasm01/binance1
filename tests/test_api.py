import unittest
from core.utils import fetch_binance_data

class TestAPI(unittest.TestCase):
    def test_binance_connection(self):
        result = fetch_binance_data("BTCUSDT")
        self.assertTrue(result is not None)
        self.assertIn("price", result)

if __name__ == "__main__":
    unittest.main()
