import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class LightScanResult:
    symbol: str
    score: float
    reasons: List[str]


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > 1e-12 else 1e-12


def _atr_like(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    # True range (basit)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    if len(tr) < period + 1:
        return float(np.mean(tr)) if len(tr) else 0.0
    return float(pd.Series(tr).rolling(period).mean().iloc[-1])


def score_symbol_from_klines(df: pd.DataFrame, lookback: int = 120) -> Tuple[float, List[str]]:
    """
    df: kline dataframe (en azından open/high/low/close/volume kolonları olmalı)
    Cheap metrikler:
      - volume z-score (son bar)
      - momentum (return_1, return_3)
      - volatility proxy (ATR-like)
    """
    reasons: List[str] = []
    if df is None or df.empty:
        return 0.0, ["empty"]

    # Kolon isimleri farklıysa minimal uyarlama yap
    # Beklenen: open, high, low, close, volume
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            return 0.0, [f"missing:{c}"]

    d = df.tail(lookback).copy()
    close = d["close"].astype(float).to_numpy()
    high = d["high"].astype(float).to_numpy()
    low = d["low"].astype(float).to_numpy()
    vol = d["volume"].astype(float).to_numpy()

    if len(close) < 30:
        return 0.0, ["too_few_bars"]

    # Volume z
    vol_mu = float(np.mean(vol[:-1])) if len(vol) > 1 else float(np.mean(vol))
    vol_sd = _safe_std(vol[:-1]) if len(vol) > 1 else _safe_std(vol)
    vol_z = float((vol[-1] - vol_mu) / vol_sd)

    # Momentum
    ret1 = float((close[-1] / close[-2] - 1.0)) if len(close) >= 2 else 0.0
    ret3 = float((close[-1] / close[-4] - 1.0)) if len(close) >= 4 else ret1

    # Volatility
    atr = _atr_like(high, low, close, period=14)

    # Skorlama (0-100)
    # Basit normalize:
    # - vol_z: 0..~6 arası kabul edip clamp
    # - |ret3|: 0..1% arası gibi
    # - atr/close: volatilite oranı
    vol_z_c = max(0.0, min(6.0, vol_z))
    mom_c = min(1.0, abs(ret3) / 0.01)         # 1% -> 1.0
    volat_c = min(1.0, (atr / close[-1]) / 0.01) if close[-1] > 0 else 0.0  # %1 ATR -> 1.0

    score = 100.0 * (0.50 * (vol_z_c / 6.0) + 0.30 * mom_c + 0.20 * volat_c)

    if vol_z > 2.0:
        reasons.append(f"vol_spike z={vol_z:.2f}")
    if abs(ret3) > 0.003:
        reasons.append(f"momentum r3={ret3:.4f}")
    if (atr / close[-1]) > 0.007:
        reasons.append(f"atr_ratio={(atr/close[-1]):.4f}")

    if not reasons:
        reasons.append("neutral")

    return float(score), reasons


class LightScanner:
    """
    Kline sağlayıcıyı dışarıdan alır:
      get_klines(symbol, interval, limit) -> pd.DataFrame
    Böylece mevcut data layer'ına dokunmadan entegre oluruz.
    """

    def __init__(self, get_klines_func, cooldown_sec: int = 20):
        self.get_klines = get_klines_func
        self.cooldown_sec = cooldown_sec
        self._last_pick_ts: Dict[str, float] = {}

    def rank(
        self,
        symbols: List[str],
        interval: str = "1m",
        limit: int = 200,
        lookback: int = 120,
        topk: int = 3,
    ) -> List[LightScanResult]:
        results: List[LightScanResult] = []

        now = time.time()
        for sym in symbols:
            # cooldown: aynı sembolü sürekli tepeye taşımasın
            last = self._last_pick_ts.get(sym, 0.0)
            if self.cooldown_sec > 0 and (now - last) < self.cooldown_sec:
                continue

            df = self.get_klines(sym, interval, limit)
            score, reasons = score_symbol_from_klines(df, lookback=lookback)
            results.append(LightScanResult(sym, score, reasons))

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[: max(1, topk)]

        # seçilenleri cooldown'a yaz
        for r in top:
            self._last_pick_ts[r.symbol] = now

        return top
