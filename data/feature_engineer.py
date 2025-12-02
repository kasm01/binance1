# data/feature_engineer.py

from typing import Optional, Callable

import importlib
import pandas as pd

from core.logger import system_logger


class FeatureEngineer:
    """
    Stateless feature engineering wrapper.

    Kullanım:
        fe = FeatureEngineer()
        features_df = fe.transform(raw_df)

    Under the hood:
      - data.feature_engineering modülünü import eder
      - build_features / build_features_v2 / run_feature_engineering
        gibi fonksiyonlardan birini bulup çağırır.
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger or system_logger
        self._build_fn: Optional[Callable] = None

    def _resolve_build_fn(self) -> Callable:
        """
        data.feature_engineering içinden uygun feature build fonksiyonunu bulur.
        Öncelik sırası:
          1) build_features
          2) build_features_v2
          3) run_feature_engineering
        Hiçbiri yoksa RuntimeError fırlatır.
        """
        if self._build_fn is not None:
            return self._build_fn

        fe_module = importlib.import_module("data.feature_engineering")

        candidate_names = [
            "build_features",
            "build_features_v2",
            "run_feature_engineering",
        ]

        for name in candidate_names:
            fn = getattr(fe_module, name, None)
            if callable(fn):
                self._build_fn = fn
                if self.logger:
                    self.logger.info(
                        "[FE] Using feature engineering function: data.feature_engineering.%s",
                        name,
                    )
                return fn

        # Hiç uygun fonksiyon bulunamadı
        msg = (
            "[FE] No suitable feature engineering function found in "
            "data.feature_engineering (tried: build_features, build_features_v2, "
            "run_feature_engineering)."
        )
        if self.logger:
            self.logger.error(msg)
        raise RuntimeError(msg)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dışarıdan gelen raw OHLCV df için tüm feature engineering pipeline'ını çalıştırır.
        """
        if self.logger:
            self.logger.info(
                "[FE] Starting feature engineering for df with shape=%s", df.shape
            )

        build_fn = self._resolve_build_fn()
        features_df = build_fn(df)

        if self.logger:
            self.logger.info(
                "[FE] Features DF shape: %s, columns=%s",
                features_df.shape,
                list(features_df.columns),
            )

        return features_df

    # Geriye dönük uyumluluk için alias
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

