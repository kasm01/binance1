# tests/test_api.py
import unittest

from data.data_loader import DataLoader


class TestAPI(unittest.TestCase):
    """
    Basit API / data pipeline testi.

    Not:
      - Gerçek Binance'e istek atar.
      - Cloud Build sırasında testler çalıştırılmıyorsa sorun yok.
      - Lokal geliştirmede "data pipeline en azından çalışıyor mu"
        diye hızlı smoke test amacıyla kullanılabilir.
    """

    def setUp(self) -> None:
        self.loader = DataLoader()

    def test_load_klines_small(self):
        """
        BTCUSDT için 1m interval, 10 bar çekip
        DataFrame döndüğünü kontrol eder.
        """
        df = self.loader.load_and_cache_klines(
            symbol="BTCUSDT",
            interval="1m",
            limit=10,
        )
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        # Feature engineering öncesi temel kolonlardan bazıları
        for col in ["open_time", "open", "high", "low", "close"]:
            self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()

