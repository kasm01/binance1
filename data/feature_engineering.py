import pandas as pd
import numpy as np
import logging

from core.exceptions import DataProcessingException

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Ham OHLCV verisinden teknik indikatörler ve model/features DataFrame'i üretir.
    """

    def __init__(self, raw_data: pd.DataFrame):
        """
        raw_data: index'i timestamp olan, en azından
        ['open', 'high', 'low', 'close', 'volume'] kolonlarını içeren DataFrame.
        """
        self.raw_data = raw_data.copy() if raw_data is not None else pd.DataFrame()

    def transform(self) -> pd.DataFrame:
        """
        Tüm feature engineering pipeline'ını çalıştırır.
        Hata olursa DataProcessingException fırlatır.
        """
        if self.raw_data.empty:
            logger.warning("[FeatureEngineer] Boş veri alındı, features üretilemedi.")
            return self.raw_data

        try:
            df = self.raw_data.copy()

            # Basit geri getiri
            df["return_1"] = df["close"].pct_change()
            df["return_5"] = df["close"].pct_change(5)
            df["return_15"] = df["close"].pct_change(15)

            # Volatilite
            df["volatility_15"] = df["return_1"].rolling(window=15).std()
            df["volatility_50"] = df["return_1"].rolling(window=50).std()

            # Hareketli ortalamalar
            df["ma_7"] = df["close"].rolling(window=7).mean()
            df["ma_25"] = df["close"].rolling(window=25).mean()
            df["ma_99"] = df["close"].rolling(window=99).mean()

            # MA oranları
            df["ma_7_25_ratio"] = df["ma_7"] / df["ma_25"]
            df["ma_25_99_ratio"] = df["ma_25"] / df["ma_99"]

            # RSI
            df["rsi_14"] = self._rsi(df["close"], window=14)

            # MACD
            macd, macd_signal, macd_hist = self._macd(df["close"])
            df["macd"] = macd
            df["macd_signal"] = macd_signal
            df["macd_hist"] = macd_hist

            # Bollinger Bands
            bb_mid = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = bb_mid + 2 * bb_std
            df["bb_lower"] = bb_mid - 2 * bb_std

            # Hacim özellikleri
            df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
            df["volume_spike"] = df["volume"] / (df["volume_ma_20"] + 1e-9)

            # NaN'leri doldur
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method="bfill").fillna(method="ffill")

            logger.info("[FeatureEngineer] Feature engineering başarıyla tamamlandı.")
            return df

        except Exception as e:
            logger.exception(f"[FeatureEngineer] Feature engineering hatası: {e}")
            raise DataProcessingException(f"Feature engineering failed: {e}") from e

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        Basit RSI hesaplama.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Basit MACD hesaplama.
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
