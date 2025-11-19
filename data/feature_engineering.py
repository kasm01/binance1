import logging
from typing import List

import numpy as np
import pandas as pd

from core.exceptions import DataProcessingException

system_logger = logging.getLogger("system")


class FeatureEngineer:
    """
    Ham OHLCV DataFrame'inden model için kullanılacak feature'ları üretir.

    main.py içinde:

        feature_engineer = FeatureEngineer(raw_df)
        features_df = feature_engineer.transform()

    şeklinde kullanılıyor.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        close, open, high, low, volume gibi kolonları numerik tipe çevirir.
        String gelmesi durumunda pct_change vs. hatalarını engeller.
        """
        numeric_cols: List[str] = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # close NaN olan satırlar bizim için işe yaramaz; düşelim
        if "close" in df.columns:
            df = df[df["close"].notnull()].copy()

        return df

    def transform(self) -> pd.DataFrame:
        """
        Ana feature engineering fonksiyonu.
        Burada:

        - Numerik kolonları float'a çevir
        - Basit getiriler (return_1, return_5, return_15)
        - Volatilite (volatility_10, volatility_30)
        - Hacim oranları (buy_ratio)
        - Rolling istatistikler (örnekler…)

        Hata olursa DataProcessingException fırlatılır.
        """
        df = self.df.copy()

        try:
            # 1) Numerik kolonları düzelt
            df = self._ensure_numeric(df)

            # 2) Getiriler
            df["return_1"] = df["close"].pct_change()
            df["return_5"] = df["close"].pct_change(5)
            df["return_15"] = df["close"].pct_change(15)

            # 3) Volatilite (return_1 üzerinden)
            df["volatility_10"] = df["return_1"].rolling(window=10).std()
            df["volatility_30"] = df["return_1"].rolling(window=30).std()

            # 4) Hacim oranları
            if "volume" in df.columns and "taker_buy_base_asset_volume" in df.columns:
                df["buy_ratio"] = (
                    df["taker_buy_base_asset_volume"] / df["volume"]
                )

            # 5) Basit hareketli ortalamalar (örnek)
            df["ma_close_10"] = df["close"].rolling(window=10).mean()
            df["ma_close_20"] = df["close"].rolling(window=20).mean()
            df["ma_close_50"] = df["close"].rolling(window=50).mean()

            # 6) Momentum benzeri feature'lar
            df["price_diff_1"] = df["close"].diff(1)
            df["price_diff_5"] = df["close"].diff(5)

            # 7) Hacim hareketleri
            if "volume" in df.columns:
                df["volume_change_1"] = df["volume"].pct_change()
                df["volume_ma_20"] = df["volume"].rolling(window=20).mean()

            # NaN'leri kısmen temizle
            df = df.dropna().copy()

            system_logger.info("[FE] Features DF shape: %s", df.shape)
            return df

        except Exception as e:
            system_logger.error("[FeatureEngineer] Feature engineering hatası: %s", e)
            raise DataProcessingException(f"Feature engineering failed: {e}") from e

