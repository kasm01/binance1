import logging
import json
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from core.utils import retry

system_logger = logging.getLogger("system")


class DataLoader:
    """
    Binance ve opsiyonel harici kaynaklardan veri çeken sınıf.
    main.py içinde tipik kullanım:

        data_loader = DataLoader(env_vars)
        raw_df = data_loader.load_klines(symbol=..., interval=..., limit=...)
        ext_df = data_loader.fetch_external_data()  # opsiyonel
    """

    def __init__(self, env_vars: Dict[str, Any] | None = None):
    self.env_vars = env_vars or {}
    self.api_keys = {
        "binance": self.env_vars.get("BINANCE_API_KEY"),
    }

        # İleride gerekirse kullanılabilecek base URL’ler
        self.binance_base_url = self.env_vars.get(
            "BINANCE_BASE_URL", "https://api.binance.com"
        )

    # -------------------------------------------------
    # Binance verisi (esas low-level fonksiyon)
    # -------------------------------------------------
    @retry
    def fetch_binance_data(
        self, symbol: str, interval: str = "1m", limit: int = 1000
    ) -> pd.DataFrame:
        """
        Binance spot klines endpoint'inden OHLCV verisi çeker.
        Dönen DataFrame yaklaşık olarak (limit, 12) shape'inde olur.

        Kolonlar Binance kline çıktısına göre:
        [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
            "ignore"
        ]
        """
        endpoint = "/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        url = f"{self.binance_base_url}{endpoint}?{urlencode(params)}"
        system_logger.info(
            "[DATA] Fetching %s klines from Binance for %s (%s)",
            limit,
            symbol,
            interval,
        )

        try:
            with urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            system_logger.error("[DATA] Error while fetching Binance data: %s", e)
            raise

        if not isinstance(data, list) or len(data) == 0:
            system_logger.error(
                "[DATA] Binance returned empty/invalid data for %s",
                symbol,
            )
            return pd.DataFrame()

        # Binance kline formatı: 12+ alanlık liste
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]

        df = pd.DataFrame(data, columns=columns[: len(data[0])])

        system_logger.info("[DATA] Raw DF shape: %s", df.shape)
        return df

    # -------------------------------------------------
    # main.py ile uyumlu wrapper metodlar
    # -------------------------------------------------
    def load_klines(
        self, symbol: str, interval: str = "1m", limit: int = 1000
    ) -> pd.DataFrame:
        """
        main.py içindeki run_data_pipeline fonksiyonunun çağırdığı
        standart yükleme fonksiyonu.

        Aslında sadece fetch_binance_data için ince bir wrapper.
        """
        return self.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)

    # İstersen başka isimler de desteklensin diye ek wrapper'lar:
    def load_and_cache_klines(
        self, symbol: str, interval: str = "1m", limit: int = 1000
    ) -> pd.DataFrame:
        """
        Geriye dönük uyumluluk için ek isim.

        Şu an cache mekanizması yok; direkt Binance'ten çeker.
        İleride Redis / dosya cache eklenecekse burası güncellenir.
        """
        return self.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)

    def load(
        self, symbol: str, interval: str = "1m", limit: int = 1000
    ) -> pd.DataFrame:
        """
        Genel ama basit alias: load() da klines çeker.
        """
        return self.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)

    # -------------------------------------------------
    # Harici veri
    # -------------------------------------------------
    @retry
    def fetch_external_data(self, url: Optional[str] = None) -> pd.DataFrame:
        """
        Harici bir kaynaktan (örneğin CSV URL) ek veri çeker.
        main.py içinde parametresiz olarak çağrılıyor:

            ext_df = data_loader.fetch_external_data()

        Bu yüzden url parametresini OPSİYONEL yapıyoruz.
        Eğer url None ise, ENV'den EXTERNAL_DATA_URL alınır.
        Eğer o da yoksa, sadece uyarı loglayıp boş DataFrame döner.
        """
        if url is None:
            url = self.env_vars.get("EXTERNAL_DATA_URL")

        if not url:
            system_logger.warning(
                "[DATA] No EXTERNAL_DATA_URL provided; skipping external data merge."
            )
            return pd.DataFrame()

        system_logger.info("[DATA] Fetching external data from %s", url)

        try:
            df = pd.read_csv(url)
            system_logger.info("[DATA] External DF shape: %s", df.shape)
            return df
        except Exception as e:
            system_logger.error(
                "[DATA] Error while fetching external data from %s: %s", url, e
            )
            # İstersen burada raise yerine boş DF döndürebilirsin
            raise
