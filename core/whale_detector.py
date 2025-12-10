from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List, Deque
from collections import deque
from datetime import datetime
import asyncio

import numpy as np
import pandas as pd

Direction = Literal["long", "short", "none"]


@dataclass
class WhaleSignal:
    direction: Direction
    score: float           # 0.0 – 1.0 arası güç
    reason: str
    meta: Dict[str, Any]


class WhaleDetector:
    """
    Ana basit + genişletilebilir whale dedektörü.

    🔹 Bot şu an sadece:
        whale_detector.from_klines(df)
    çağrısını kullanıyor.

    Bu metod tam olarak eski versiyonla uyumlu bırakıldı.
    Ek olarak calculate_features ve diğer yardımcı metodlar
    ile daha gelişmiş analiz yapılabiliyor.
    """

    def __init__(
        self,
        buy_dom_threshold: float = 0.65,
        sell_dom_threshold: float = 0.65,
        volume_zscore_thr: float = 1.5,
        window: int = 50,
        rsi_window: int = 14,
        obv_window: int = 20,
    ) -> None:
        self.buy_dom_threshold = buy_dom_threshold
        self.sell_dom_threshold = sell_dom_threshold
        self.volume_zscore_thr = volume_zscore_thr
        self.window = window
        self.rsi_window = rsi_window
        self.obv_window = obv_window

    # ------------------------------------------------------------------
    # === BOTUN KULLANDIĞI ESKİ API: from_klines ======================
    # ------------------------------------------------------------------
    def from_klines(self, df: pd.DataFrame) -> WhaleSignal:
        """
        Eski versiyonla birebir uyumlu basit whale sinyali.
        df: en az şu kolonları içermeli:
            ['open', 'high', 'low', 'close', 'volume']
        """
        if df is None or len(df) < self.window + 1:
            return WhaleSignal("none", 0.0, "not_enough_data", {})

        tail = df.tail(self.window).copy()

        # Mum body ve yön
        tail["body"] = tail["close"] - tail["open"]
        tail["dir"] = np.where(tail["body"] > 0, 1, np.where(tail["body"] < 0, -1, 0))

        # Volume z-score
        vol = tail["volume"].astype(float)
        vol_mean = vol.mean()
        vol_std = vol.std() or 1.0
        tail["vol_z"] = (vol - vol_mean) / vol_std

        last = tail.iloc[-1]
        last_dir = int(last["dir"])
        last_vol_z = float(last["vol_z"])

        meta = {
            "last_body": float(last["body"]),
            "last_dir": last_dir,
            "last_vol_z": last_vol_z,
            "vol_mean": float(vol_mean),
            "vol_std": float(vol_std),
        }

        # Volatility düşük / veri anlamsızsa
        if np.isnan(last_vol_z):
            return WhaleSignal("none", 0.0, "nan_zscore", meta)

        # Yüksek volume yoksa whale saymayalım
        if last_vol_z < self.volume_zscore_thr:
            return WhaleSignal("none", 0.0, "no_volume_spike", meta)

        # Yön analizi (son mum büyük yeşil/kırmızı)
        score = float(max(0.0, min(1.0, last_vol_z / 3.0)))
        meta["score_raw"] = score

        if last_dir > 0:
            return WhaleSignal("long", score, "big_green_volume_spike", meta)
        elif last_dir < 0:
            return WhaleSignal("short", score, "big_red_volume_spike", meta)
        else:
            return WhaleSignal("none", 0.0, "doji_or_flat", meta)

    # ------------------------------------------------------------------
    # === Gelişmiş feature hesaplamaları ===============================
    # ------------------------------------------------------------------
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş teknik göstergeler ekler.
        Beklenen kolonlar: ['open','high','low','close','volume']
        """
        df = df.copy()
        if len(df) == 0:
            return df

        # 1. Volume analizleri
        df["volume_ma"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
        df["volume_zscore"] = (
            df["volume"] - df["volume"].rolling(50, min_periods=10).mean()
        ) / df["volume"].rolling(50, min_periods=10).std()

        # 2. Price-Volume ilişkisi
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["pvt"] = (
            (df["typical_price"] - df["typical_price"].shift(1))
            / df["typical_price"].shift(1)
        ) * df["volume"]
        df["pvt_cum"] = df["pvt"].cumsum()

        # 3. Mum gövdesi / wick analizi
        df["body_size"] = (df["close"] - df["open"]).abs()
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / (
            df["body_size"] + 1e-10
        )

        # 4. Basit volume pattern'leri
        df["volume_spike"] = df["volume_ratio"] > 2.0
        df["climax_volume"] = (df["volume"] > df["volume"].rolling(20).quantile(0.9)) & (
            df["body_size"] > df["body_size"].rolling(20).mean() * 1.5
        )

        return df

    # ------------------------------------------------------------------
    # === Multi-factor skor hesaplayan yardımcı metodlar ==============
    # ------------------------------------------------------------------
    def _calculate_volume_score(self, recent: pd.DataFrame, multiplier: float) -> float:
        vol_z = recent["volume_zscore"].iloc[-1]
        if np.isnan(vol_z):
            return 0.0
        score = float(max(0.0, min(1.0, (vol_z / 3.0) * multiplier)))
        return score

    def _calculate_price_action_score(self, recent: pd.DataFrame) -> float:
        body = (recent["close"] - recent["open"]).iloc[-1]
        body_abs = abs(body)
        body_mean = recent["body_size"].rolling(10, min_periods=1).mean().iloc[-1] or 1e-8
        wick_ratio = recent["wick_ratio"].iloc[-1]

        norm_body = body_abs / body_mean
        score = float(max(0.0, min(1.0, norm_body / (1.0 + wick_ratio))))
        return score

    def _calculate_momentum_score(self, recent: pd.DataFrame) -> float:
        returns = recent["close"].pct_change().tail(5)
        cum_ret = returns.sum()
        score = float(max(0.0, min(1.0, abs(cum_ret) * 100)))  # % olarak sıkıştır
        return score

    def _calculate_absorption_score(self, recent: pd.DataFrame) -> float:
        wick_ratio = recent["wick_ratio"].iloc[-1]
        if wick_ratio > 5:
            score = float(max(0.0, min(1.0, (wick_ratio - 5) / 5)))
        else:
            score = 0.0
        return score

    def _determine_direction(
        self, recent: pd.DataFrame, scores: Dict[str, float]
    ) -> Direction:
        last_close = recent["close"].iloc[-1]
        last_open = recent["open"].iloc[-1]
        last_body = last_close - last_open

        if last_body > 0 and scores["momentum_score"] > 0.1:
            return "long"
        elif last_body < 0 and scores["momentum_score"] > 0.1:
            return "short"
        else:
            return "none"

    def _generate_reason(self, scores: Dict[str, float]) -> str:
        parts = []
        for k, v in scores.items():
            if v > 0.3:
                parts.append(f"{k}:{v:.2f}")
        return " | ".join(parts) if parts else "weak_signal"


class MultiTimeframeWhaleDetector(WhaleDetector):
    """
    Çoklu timeframe whale dedektörü.
    Şu an bot sadece tek TF'de `from_klines` kullanıyor ama
    istersen MTF kline setlerini de buraya bağlayabilirsin.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeframes = ["5m", "15m", "1h", "4h"]

    def analyze_multiple_timeframes(
        self, dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, WhaleSignal]:
        signals: Dict[str, WhaleSignal] = {}

        for tf, df in dfs.items():
            if df is None or len(df) < max(100, self.window + 10):
                signals[tf] = WhaleSignal("none", 0.0, f"insufficient_data_{tf}", {})
                continue

            df_features = self.calculate_features(df)
            tf_multiplier = {
                "5m": 1.0,
                "15m": 1.2,
                "1h": 1.5,
                "4h": 2.0,
            }.get(tf, 1.0)

            signal = self._generate_signal(df_features, tf_multiplier)
            signals[tf] = signal

        return signals

    def _generate_signal(self, df: pd.DataFrame, multiplier: float = 1.0) -> WhaleSignal:
        tail = df.tail(self.window)
        if len(tail) < 5:
            return WhaleSignal("none", 0.0, "not_enough_tail", {})

        recent = tail.tail(5)

        scores = {
            "volume_score": self._calculate_volume_score(recent, multiplier),
            "price_action_score": self._calculate_price_action_score(recent),
            "momentum_score": self._calculate_momentum_score(recent),
            "absorption_score": self._calculate_absorption_score(recent),
        }

        weights = {
            "volume_score": 0.35,
            "price_action_score": 0.25,
            "momentum_score": 0.20,
            "absorption_score": 0.20,
        }

        total_score = sum(scores[k] * weights[k] for k in scores)
        direction = self._determine_direction(recent, scores)

        return WhaleSignal(
            direction=direction,
            score=float(min(1.0, max(0.0, total_score))),
            reason=self._generate_reason(scores),
            meta={"scores": scores, "timeframe_multiplier": multiplier},
        )


# ----------------------------------------------------------------------
# === ML tabanlı, order-flow tabanlı ve backtest / optimizasyon =======
# === Bunlar OPSİYONEL, ana runtime'a bağımlı değil.                  ==
# ----------------------------------------------------------------------

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except Exception:
    IsolationForest = None        # type: ignore
    StandardScaler = None         # type: ignore


class MLWhaleDetector(WhaleDetector):
    """
    IsolationForest ile anomaly-based whale dedektörü.
    sklearn yoksa çalışmaz ama ana botu da bozmaz.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[IsolationForest] = None
        if model_path is not None:
            self._load_model(model_path)

    def _extract_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = self.calculate_features(df)
        cols = [
            "volume",
            "volume_ratio",
            "volume_zscore",
            "body_size",
            "wick_ratio",
            "pvt",
        ]
        existing = [c for c in cols if c in df_feat.columns]
        return df_feat[existing].fillna(0.0)

    def _load_model(self, model_path: str) -> None:
        import joblib

        try:
            obj = joblib.load(model_path)
            self.model = obj.get("model")
            self.scaler = obj.get("scaler")
        except Exception:
            self.model = None
            self.scaler = None

    def train_anomaly_model(self, df: pd.DataFrame) -> None:
        if IsolationForest is None or StandardScaler is None:
            raise RuntimeError("sklearn yüklü değil, MLWhaleDetector kullanılamaz.")

        features = self._extract_features_for_ml(df)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features)

        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        self.model.fit(X)

    def detect_with_ml(self, df: pd.DataFrame) -> WhaleSignal:
        if self.model is None or self.scaler is None:
            return WhaleSignal("none", 0.0, "model_not_trained", {})

        features = self._extract_features_for_ml(df.tail(self.window))
        X = self.scaler.transform(features)
        scores = self.model.decision_function(X)
        preds = self.model.predict(X)

        anomaly_score = float(scores[-1])
        is_anomaly = preds[-1] == -1

        if is_anomaly and abs(anomaly_score) > 0.5:
            last_feat = features.iloc[-1]
            direction = "long" if last_feat.get("pvt", 0.0) > 0 else "short"
            score = float(min(1.0, max(0.0, abs(anomaly_score))))
            return WhaleSignal(
                direction=direction,
                score=score,
                reason=f"ml_anomaly_detected_score_{anomaly_score:.2f}",
                meta={
                    "anomaly_score": anomaly_score,
                    "features": last_feat.to_dict(),
                },
            )

        return WhaleSignal("none", 0.0, "no_anomaly_detected", {})


class OrderFlowWhaleDetector(WhaleDetector):
    """
    Order-flow proxy analizleri (VWAP, volume cluster vs.).
    Şimdilik yardımcı bilgiler için.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _find_volume_clusters(
        self, df: pd.DataFrame, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        if len(df) == 0:
            return clusters

        price_bins = np.linspace(df["low"].min(), df["high"].max(), 20)
        vol_mean = df["volume"].mean() or 1e-8

        for i in range(len(price_bins) - 1):
            mask = (df["low"] >= price_bins[i]) & (df["high"] <= price_bins[i + 1])
            if not mask.any():
                continue
            cluster_volume = df.loc[mask, "volume"].sum()
            if cluster_volume > vol_mean * threshold:
                clusters.append(
                    {
                        "price_range": (float(price_bins[i]), float(price_bins[i + 1])),
                        "volume": float(cluster_volume),
                    }
                )
        return clusters

    def detect_large_orders(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) == 0:
            return {
                "buy_pressure": 0.0,
                "sell_pressure": 0.0,
                "volume_clusters": [],
            }

        df = df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (df["volume"] * df["typical_price"]).cumsum() / df["volume"].cumsum()
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"] * 100.0
        df["price_change"] = df["close"].pct_change()
        df["volume_delta"] = df["volume"] * np.sign(df["price_change"].fillna(0.0))

        recent = df.tail(20)
        buy_volume = recent.loc[recent["close"] > recent["open"], "volume"].sum()
        sell_volume = recent.loc[recent["close"] < recent["open"], "volume"].sum()
        total_volume = buy_volume + sell_volume

        buy_pressure = float(buy_volume / total_volume) if total_volume > 0 else 0.0
        sell_pressure = float(sell_volume / total_volume) if total_volume > 0 else 0.0

        clusters = self._find_volume_clusters(df)

        return {
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "volume_clusters": clusters,
        }


class RealTimeWhaleMonitor:
    """
    Akış bazlı whale takibi için helper.
    Şu an botta kullanılmıyor ama websocket ile ileride kullanabiliriz.
    """

    def __init__(self, detector: WhaleDetector, alert_threshold: float = 0.7):
        self.detector = detector
        self.alert_threshold = alert_threshold
        self.signal_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.alert_callbacks: List = []

    async def monitor_stream(self, data_stream):
        async for df in data_stream:
            signal = self.detector.from_klines(df)
            self.signal_history.append(
                {"timestamp": datetime.utcnow(), "signal": signal}
            )

            if signal.score >= self.alert_threshold:
                await self._trigger_alerts(signal)

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    async def _trigger_alerts(self, signal: WhaleSignal) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "direction": signal.direction,
            "score": signal.score,
            "reason": signal.reason,
            "meta": signal.meta,
        }
        for cb in self.alert_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(payload)
            else:
                cb(payload)


class WhaleDetectorBacktester:
    """
    Whale sinyallerinin performansını ölçmek için basit backtester.
    Ana runtime tarafından çağrılmıyor, offline analiz için.
    """

    def __init__(self, detector: WhaleDetector):
        self.detector = detector

    def backtest(self, df: pd.DataFrame, lookahead_bars: int = 10) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []

        for i in range(len(df) - self.detector.window - lookahead_bars):
            window_data = df.iloc[i : i + self.detector.window]
            signal = self.detector.from_klines(window_data)

            if signal.direction == "none":
                continue

            future_data = df.iloc[
                i + self.detector.window : i + self.detector.window + lookahead_bars
            ]
            performance = self._calculate_performance(signal, window_data, future_data)

            results.append(
                {
                    "signal": signal,
                    "performance": performance,
                    "index": int(i + self.detector.window),
                }
            )

        return self._analyze_results(results)

    def _calculate_performance(
        self, signal: WhaleSignal, window_data: pd.DataFrame, future_data: pd.DataFrame
    ) -> float:
        entry_price = float(window_data["close"].iloc[-1])
        exit_price = float(future_data["close"].iloc[-1])

        if signal.direction == "long":
            return (exit_price - entry_price) / entry_price
        elif signal.direction == "short":
            return (entry_price - exit_price) / entry_price
        return 0.0

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {
                "n_signals": 0,
                "avg_return": 0.0,
                "win_rate": 0.0,
                "returns": [],
            }

        rets = np.array([r["performance"] for r in results], dtype=float)
        avg = float(rets.mean())
        win_rate = float((rets > 0).mean())

        return {
            "n_signals": len(results),
            "avg_return": avg,
            "win_rate": win_rate,
            "returns": rets.tolist(),
        }


# Optuna tabanlı optimizasyon opsiyonel
try:
    import optuna  # type: ignore
except Exception:
    optuna = None


class OptimizedWhaleDetector:
    """
    Parametre optimizasyonu için helper.
    Bot çalışmasını etkilemiyor; offline tuning için kullanılabilir.
    """

    def __init__(self):
        self.best_params: Optional[Dict[str, Any]] = None

    def optimize_parameters(
        self, df: pd.DataFrame, n_trials: int = 50
    ) -> Optional[Dict[str, Any]]:
        if optuna is None:
            raise RuntimeError("optuna yüklü değil, optimizasyon kullanılamaz.")

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "volume_zscore_thr": trial.suggest_float(
                    "volume_zscore_thr", 1.0, 3.0
                ),
                "window": trial.suggest_int("window", 20, 100),
            }

            detector = WhaleDetector(
                volume_zscore_thr=params["volume_zscore_thr"],
                window=params["window"],
            )
            backtester = WhaleDetectorBacktester(detector)
            res = backtester.backtest(df)

            returns = np.array(res["returns"], dtype=float)
            if len(returns) < 5:
                return 0.0
            mean = returns.mean()
            std = returns.std() or 1e-8
            sharpe = mean / std
            return float(sharpe)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        return self.best_params
