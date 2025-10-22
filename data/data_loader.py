import pandas as pd
import requests
from core.utils import retry
from core.exceptions import DataProcessingException

class DataLoader:
    def __init__(self, api_keys):
        self.api_keys = api_keys

    @retry(max_attempts=5, delay=2, exceptions=(requests.RequestException,))
    def fetch_binance_data(self, symbol, interval='1m', limit=1000):
        """
        Binance API üzerinden veri çeker.
        """
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            response = requests.get(url)
            response.raise_for_status()
            data = pd.DataFrame(response.json(), columns=[
                "open_time","open","high","low","close","volume","close_time",
                "quote_asset_volume","num_trades","taker_buy_base","taker_buy_quote","ignore"
            ])
            return data
        except Exception as e:
            raise DataProcessingException(f"Binance data fetch failed: {e}")

    @retry(max_attempts=5, delay=2, exceptions=(requests.RequestException,))
    def fetch_external_data(self, url):
        """
        Diğer API'lerden veri çekme.
        """
        try:
            response = requests.get(url, headers={"Authorization": f"Bearer {self.api_keys.get('external', '')}"})
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except Exception as e:
            raise DataProcessingException(f"External data fetch failed: {e}")
