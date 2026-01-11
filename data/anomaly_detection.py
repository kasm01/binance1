# data/anomaly_detection.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.logger import system_logger
from features.schema import normalize_to_schema


Context = Literal["heavy", "scan"]


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int_env(name: str, default: int) -> int:
    try:
        v = os.getenv(name, None)
        if v is None or str(v).strip() == "":
            return int(default)
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


def _get_float_env(name: str, default: float) -> float:
    try:
        v = os.getenv(name, None)
        if v is None or str(v).strip() == "":
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


@dataclass
class _IsoCacheEntry:
    model: IsolationForest
    fitted_at_ts: float
    n: int
    d: int


class AnomalyDetector:
    """
    Cached anomaly detection wrapper.

    IsolationForest ile numeric feature'lar üzerinden aykırı gözlemleri filtreler.

    - schema verilirse: önce normalize_to_schema(schema) uygulanır (tek sözleşme)
    - çıktı index resetlenir
    - cache: aynı (schema + numeric_cols + params + context) için modeli reuse eder
      ve refit_every_sec ile periyodik refit yapar.
    """

    def __init__(
        self,
        logger=None,
        contamination: float = 0.02,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.logger = logger or system_logger
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)

        self._cache: Dict[Tuple, _IsoCacheEntry] = {}

        # HEAVY defaults
        self.enabled = _get_bool_env("ANOMALY_ENABLE", True)
        self.refit_every_sec = float(_get_int_env("ANOM_REFIT_EVERY_SEC", 300))
        self.min_samples = int(_get_int_env("ANOM_MIN_SAMPLES", 50))

        # SCAN profile defaults (aggressive speed)
        self.scan_enabled = _get_bool_env("ANOM_SCAN_ENABLE", True)
        self.scan_refit_every_sec = float(_get_int_env("ANOM_SCAN_REFIT_EVERY_SEC", 1800))
        self.scan_min_samples = int(_get_int_env("ANOM_SCAN_MIN_SAMPLES", 200))

    def _select_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if self.logger:
            self.logger.debug("[ANOM] Numeric columns for anomaly detection: %s", numeric_cols)
        return numeric_cols

    def _cache_key(self, context: Context, schema: Optional[List[str]], numeric_cols: List[str]) -> Tuple:
        sch_key = tuple(schema) if isinstance(schema, list) and schema else None
        cols_key = tuple(numeric_cols)
        return (
            "iso_v2",
            str(context),
            sch_key,
            cols_key,
            float(self.contamination),
            int(self.n_estimators),
            int(self.random_state),
        )

    def _profile(self, context: Context) -> Tuple[bool, float, int]:
        if context == "scan":
            return bool(self.scan_enabled), float(self.scan_refit_every_sec), int(self.scan_min_samples)
        return bool(self.enabled), float(self.refit_every_sec), int(self.min_samples)

    def _should_refit(self, entry: Optional[_IsoCacheEntry], n: int, d: int, refit_every_sec: float) -> bool:
        if entry is None:
            return True
        if entry.n != int(n) or entry.d != int(d):
            return True
        if refit_every_sec <= 0:
            return False
        now = time.time()
        return (now - float(entry.fitted_at_ts)) >= float(refit_every_sec)

    def filter_anomalies(
        self,
        features_df: pd.DataFrame,
        schema: Optional[List[str]] = None,
        *,
        context: Context = "heavy",
    ) -> pd.DataFrame:
        enabled, refit_every_sec, min_samples = self._profile(context)

        if not enabled:
            return features_df.reset_index(drop=True) if isinstance(features_df, pd.DataFrame) else features_df

        if features_df is None or len(features_df) == 0:
            if self.logger:
                self.logger.warning("[ANOM] Empty features_df, skipping anomaly filter.")
            return features_df

        if len(features_df) < min_samples:
            if self.logger:
                self.logger.debug("[ANOM] context=%s n=%d < min_samples=%d -> skip", context, len(features_df), min_samples)
            return features_df.reset_index(drop=True)

        df = features_df

        # schema normalize
        if isinstance(schema, list) and schema:
            try:
                df = normalize_to_schema(df, schema)
            except Exception as e:
                if self.logger:
                    self.logger.warning("[ANOM] normalize_to_schema failed, continuing raw. err=%s", e)
                df = features_df

        numeric_cols = self._select_numeric_columns(df)
        if not numeric_cols:
            if self.logger:
                self.logger.warning("[ANOM] No numeric columns found. Skipping anomaly filter.")
            return df.reset_index(drop=True)

        try:
            X = df[numeric_cols].to_numpy(dtype=float, copy=False)
        except Exception:
            X = df[numeric_cols].astype(float).to_numpy(copy=False)

        n, d = int(X.shape[0]), int(X.shape[1])
        key = self._cache_key(context=context, schema=schema, numeric_cols=numeric_cols)
        entry = self._cache.get(key)

        refit = self._should_refit(entry, n=n, d=d, refit_every_sec=refit_every_sec)

        if refit:
            if self.logger:
                self.logger.info(
                    "[ANOM] IsolationForest FIT[%s]: n=%d, d=%d, contamination=%.4f (refit_every_sec=%.0f)",
                    context,
                    n,
                    d,
                    float(self.contamination),
                    float(refit_every_sec),
                )

            iso = IsolationForest(
                contamination=float(self.contamination),
                n_estimators=int(self.n_estimators),
                random_state=int(self.random_state),
                n_jobs=-1,
            )
            iso.fit(X)
            self._cache[key] = _IsoCacheEntry(model=iso, fitted_at_ts=time.time(), n=n, d=d)
        else:
            iso = entry.model  # type: ignore[union-attr]
            if self.logger:
                self.logger.debug(
                    "[ANOM] IsolationForest CACHE HIT[%s]: n=%d d=%d age=%.1fs",
                    context,
                    n,
                    d,
                    (time.time() - float(entry.fitted_at_ts)),
                )

        preds = iso.predict(X)  # 1=normal, -1=anomaly
        mask = preds == 1

        removed = int((~mask).sum())
        clean_df = df.loc[mask].copy().reset_index(drop=True)

        if self.logger:
            self.logger.info("[ANOM] Removed %d anomalous rows. Remaining=%d", removed, clean_df.shape[0])

        return clean_df

    # Backwards compat
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return self.filter_anomalies(features_df)
