import pandas as pd
import numpy as np

class FeatureEngineering:
    @staticmethod
    def add_technical_indicators(df):
        """
        Basit teknik göstergeler ekler: SMA, EMA, RSI vb.
        """
        df['SMA_20'] = df['close'].astype(float).rolling(20).mean()
        df['EMA_20'] = df['close'].astype(float).ewm(span=20, adjust=False).mean()
        df['returns'] = df['close'].astype(float).pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['RSI_14'] = FeatureEngineering.compute_rsi(df['close'].astype(float), 14)
        df.fillna(0, inplace=True)
        return df

    @staticmethod
    def compute_rsi(series, period):
        delta = series.diff()
        up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
        ma_up = up.rolling(period).mean()
        ma_down = down.rolling(period).mean()
        rsi = 100 - (100/(1 + ma_up/ma_down))
        return rsi.fillna(0)
