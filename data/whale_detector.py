from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any
import numpy as np
import pandas as pd

Direction = Literal["long", "short", "none"]

@dataclass
class WhaleSignal:
    direction: Direction
    score: float
    reason: str
    meta: Dict[str, Any]


class WhaleDetector:
    """
    Basit whale dedektörü (botun asıl kullandığı versiyon).
    - from_klines(df) → WhaleSignal döner
    """

    def __init__(
        self,
        buy_dom_threshold: float = 0.65,
        sell_dom_threshold: float = 0.65,
        volume_zscore_thr: float = 1.5,
        window: int = 50,
    ):
        self.buy_dom_threshold = buy_dom_threshold
        self.sell_dom_threshold = sell_dom_threshold
        self.volume_zscore_thr = volume_zscore_thr
        self.window = window

    def from_klines(self, df: pd.DataFrame) -> WhaleSignal:
        """Botun runtime'da kullandığı ana fonksiyon."""
        if df is None or len(df) < self.window + 1:
            return WhaleSignal("none", 0.0, "not_enough_data", {})

        tail = df.tail(self.window).copy()
        tail["body"] = tail["close"] - tail["open"]
        tail["dir"] = np.where(tail["body"] > 0, 1, np.where(tail["body"] < 0, -1, 0))

        vol = tail["volume"].astype(float)
        vol_mean = vol.mean()
        vol_std = vol.std() or 1.0
        tail["vol_z"] = (vol - vol_mean) / vol_std

        last = tail.iloc[-1]
        direction_raw = int(last["dir"])
        vol_z = float(last["vol_z"])

        meta = {
            "last_body": float(last["body"]),
            "last_dir": direction_raw,
            "last_vol_z": vol_z,
            "vol_mean": float(vol_mean),
            "vol_std": float(vol_std),
        }

        # Hacim spike yoksa whale yok
        if vol_z < self.volume_zscore_thr:
            return WhaleSignal("none", 0.0, "no_volume_spike", meta)

        # Score normalize
        score = float(max(0.0, min(1.0, vol_z / 3)))

        if direction_raw > 0:
            return WhaleSignal("long", score, "big_green_volume_spike", meta)
        elif direction_raw < 0:
            return WhaleSignal("short", score, "big_red_volume_spike", meta)
        else:
            return WhaleSignal("none", 0.0, "doji_or_flat", meta)
# ---------------------------------------------------------
# FEATURE ENGINE: Whale analizi için ortak hesaplayıcı
# ---------------------------------------------------------

class WhaleFeatureEngine:
    """
    MTF whale dedektörünün temel aldığı ek istatistikleri hesaplar.
    (RSI, volume ratio, candle body strength vs.)
    """
    def __init__(self):
        pass

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 5:
            return df
        
        out = df.copy()

        # Candle body
        out["body"] = out["close"] - out["open"]
        out["dir"] = np.where(out["body"] > 0, 1, np.where(out["body"] < 0, -1, 0))

        # Volume ratio (buy/sell dominance yoksa hacim oranı)
        vol = out["volume"].astype(float)
        out["vol_ema"] = vol.ewm(span=20).mean()
        out["vol_ratio"] = vol / (out["vol_ema"] + 1e-9)

        # Normalized body strength
        body_abs = np.abs(out["body"])
        out["body_strength"] = body_abs / (body_abs.rolling(20).mean() + 1e-9)

        # Z-score volume
        vol_mean = vol.rolling(30).mean()
        vol_std = vol.rolling(30).std()
        out["vol_z"] = (vol - vol_mean) / (vol_std + 1e-9)

        # Rolling direction momentum
        out["dir_mom"] = out["dir"].rolling(10).sum()

        return out
# ---------------------------------------------------------
# MTF Whale Detector
# ---------------------------------------------------------

class MultiTimeframeWhaleDetector:
    """
    Çoklu timeframe whale dedektörü.

    🔹 Geriye dönük uyumluluk:
        - main.py içindeki eski çağrı:
            whale_detector.from_klines(df)
          hâlâ çalışsın diye, içeride tek TF WhaleDetector kullanıyoruz.

    🔹 Yeni MTF API:
        - analyze_multiple_timeframes(dfs) → {tf: WhaleSignal}
        - get_mtf_meta(dfs) → {
              "direction": "long"/"short"/"none",
              "score": float,
              "per_tf": { "5m": {...}, ... }
          }
    """

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        base_window: int = 50,
        volume_zscore_thr: float = 1.5,
    ) -> None:
        # Zaman dilimleri: default 1m,5m,15m,1h
        self.timeframes: List[str] = timeframes or ["1m", "5m", "15m", "1h"]

        # Her TF için bir WhaleDetector oluştur
        self.detectors: Dict[str, WhaleDetector] = {}
        for tf in self.timeframes:
            # İstersen tf'ye göre farklı window/threshold verebilirsin
            if tf.endswith("m"):
                w = base_window
            else:
                w = base_window + 20

            self.detectors[tf] = WhaleDetector(
                window=w,
                volume_zscore_thr=volume_zscore_thr,
            )

        # Tek-TF uyumluluk için bir "base" detector
        self.base_detector = WhaleDetector(
            window=base_window,
            volume_zscore_thr=volume_zscore_thr,
        )

        # Feature engine (opsiyonel, MTF analizi için)
        self.feature_engine = WhaleFeatureEngine()

    # --------------------------------------------------
    # Eski API ile uyumlu: tek dataframe → WhaleSignal
    # --------------------------------------------------
    def from_klines(self, df: pd.DataFrame, interval: Optional[str] = None) -> WhaleSignal:
        """
        Eski bot koduyla uyumluluk için:
            whale_detector.from_klines(raw_df)

        interval verilmezse base_detector ile çalışır.
        """
        if interval and interval in self.detectors:
            det = self.detectors[interval]
            return det.from_klines(df)
        # fallback: tek TF
        return self.base_detector.from_klines(df)

    # --------------------------------------------------
    # Çoklu timeframe analizi
    # --------------------------------------------------
    def analyze_multiple_timeframes(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, WhaleSignal]:
        """
        dfs: {"1m": df_1m, "5m": df_5m, ...}
        Her TF için WhaleSignal döner.
        """
        signals: Dict[str, WhaleSignal] = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 10:
                signals[tf] = WhaleSignal(
                    direction="none",
                    score=0.0,
                    reason=f"not_enough_data_{tf}",
                    meta={},
                )
                continue

            det = self.detectors.get(tf, self.base_detector)

            # Gerekirse ekstra feature'lar hesapla (şu an zorunlu değil)
            _ = self.feature_engine.compute(df)

            sig = det.from_klines(df)
            signals[tf] = sig

        return signals

    # --------------------------------------------------
    # MTF meta builder: ortak whale_meta sözlüğü
    # --------------------------------------------------
    def get_mtf_meta(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Çoklu TF sinyallerinden tek bir aggregated whale_meta üretir.

        Dönüş:
          {
            "direction": "long"/"short"/"none",
            "score": 0.0~1.0,
            "per_tf": {
                "5m": { "direction": "...", "score": ..., "reason": "..."},
                ...
            }
          }
        """
        signals = self.analyze_multiple_timeframes(dfs)

        # TF ağırlıkları (daha büyük TF daha ağır basıyor)
        tf_weights = {
            "1m": 1.0,
            "5m": 1.2,
            "15m": 1.5,
            "1h": 2.0,
        }

        long_score = 0.0
        short_score = 0.0
        per_tf_meta: Dict[str, Any] = {}

        for tf, sig in signals.items():
            w = tf_weights.get(tf, 1.0)
            s = max(0.0, min(1.0, float(sig.score)))

            if sig.direction == "long":
                long_score += s * w
            elif sig.direction == "short":
                short_score += s * w

            per_tf_meta[tf] = {
                "direction": sig.direction,
                "score": s,
                "reason": sig.reason,
                "meta": sig.meta,
            }

        if long_score == 0.0 and short_score == 0.0:
            agg_dir = "none"
            agg_score = 0.0
        else:
            if long_score >= short_score:
                agg_dir = "long"
                agg_score_raw = long_score / (long_score + short_score)
            else:
                agg_dir = "short"
                agg_score_raw = short_score / (long_score + short_score)
            # Hafif sıkıştırma: 0.5 → zayıf, 1.0 → güçlü
            agg_score = float(max(0.0, min(1.0, (agg_score_raw - 0.5) * 2.0)))

        return {
            "direction": agg_dir,
            "score": agg_score,
            "per_tf": per_tf_meta,
        }
