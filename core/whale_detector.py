from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List, Deque
from collections import deque
from datetime import datetime
import asyncio
import os

import numpy as np
import pandas as pd

Direction = Literal["long", "short", "none"]


@dataclass
class WhaleSignal:
    direction: Direction
    score: float
    reason: str
    meta: Dict[str, Any]


@dataclass
class WhaleDecision:
    direction: Direction
    score: float
    regime: str
    action: str
    confidence: float
    reason: str
    meta: Dict[str, Any]


class WhaleDetector:
    """
    WhaleDetector
    - Eski API uyumu korunur: from_klines(df) -> WhaleSignal
    - Ek olarak gelişmiş feature, regime ve final decision üretir
    """

    def __init__(
        self,
        buy_dom_threshold: float = 0.65,
        sell_dom_threshold: float = 0.65,
        volume_zscore_thr: float = 1.5,
        window: int = 50,
        rsi_window: int = 14,
        obv_window: int = 20,
        boost_thr: float = 0.58,
        veto_thr: float = 0.67,
        force_exit_thr: float = 0.80,
    ) -> None:
        self.buy_dom_threshold = float(buy_dom_threshold)
        self.sell_dom_threshold = float(sell_dom_threshold)
        self.volume_zscore_thr = float(volume_zscore_thr)
        self.window = int(window)
        self.rsi_window = int(rsi_window)
        self.obv_window = int(obv_window)

        self.boost_thr = float(boost_thr)
        self.veto_thr = float(veto_thr)
        self.force_exit_thr = float(force_exit_thr)

        self.whale_confirm_thr = self._env_float(
            "WHALE_CONFIRM_THR",
            0.58,
        )
        self.whale_veto_thr = self._env_float(
            "WHALE_VETO_THR",
            0.68,
        )
        self.whale_short_veto_thr = self._env_float(
            "WHALE_SHORT_VETO_THR",
            self.whale_veto_thr,
        )

        self.whale_trend_thr = self._env_float("WHALE_TREND_THR", 0.80)
        self.whale_range_thr = self._env_float("WHALE_RANGE_THR", 0.30)
        self.whale_range_penalty = self._env_float("WHALE_RANGE_PENALTY", 0.75)
        self.whale_trend_bonus = self._env_float("WHALE_TREND_BONUS", 1.18)
        self.whale_xconf_bonus = self._env_float("WHALE_XCONF_BONUS", 1.12)

        self.whale_force_exit_enable = self._env_bool(
            "WHALE_FORCE_EXIT_ENABLE",
            True,
        )
        self.whale_force_exit_thr = self._env_float(
            "WHALE_FORCE_EXIT_THR",
            0.70,
        )
        self.whale_force_exit_min_pnl_pct = self._env_float(
            "WHALE_FORCE_EXIT_MIN_PNL_PCT",
            -0.0015,
        )
        self.whale_force_exit_on_profit_only = self._env_bool(
            "WHALE_FORCE_EXIT_ON_PROFIT_ONLY",
            False,
        )
        self.whale_force_exit_confirm_bars = self._env_int(
            "WHALE_FORCE_EXIT_CONFIRM_BARS",
            1,
        )

        self.whale_block_actions = {
            x.strip()
            for x in str(
                os.getenv("WHALE_BLOCK_ACTIONS", "hard_block,force_exit")
            ).split(",")
            if x.strip()
        }
        self.whale_reduce_actions = {
            x.strip()
            for x in str(
                os.getenv(
                    "WHALE_REDUCE_ACTIONS",
                    "reduce_size,tighten_risk,avoid_open",
                )
            ).split(",")
            if x.strip()
        }
        self.whale_boost_actions = {
            x.strip()
            for x in str(
                os.getenv(
                    "WHALE_BOOST_ACTIONS",
                    "confirm,strong_confirm,hold_winner,soft_confirm",
                )
            ).split(",")
            if x.strip()
        }

        self.ema_whale_only = self._env_bool("EMA_WHALE_ONLY", False)
        self.ema_whale_thr = self._env_float("EMA_WHALE_THR", 0.45)
        self.min_mtf_agreement_score = self._env_float(
            "WHALE_MTF_MIN_AGREEMENT_SCORE",
            0.55,
        )
    # ------------------------------------------------------------------
    # yardımcılar
    # ------------------------------------------------------------------
    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(str(os.getenv(name, str(default))).strip())
        except Exception:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(str(os.getenv(name, str(default))).strip())
        except Exception:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off", ""):
            return False
        return default

    @staticmethod
    def _clip01(x: Any) -> float:
        try:
            return float(max(0.0, min(1.0, float(x))))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return float(default)
            return v
        except Exception:
            return float(default)
    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        close_diff = close.diff().fillna(0.0)
        direction = np.sign(close_diff)
        obv = (volume.fillna(0.0) * direction).cumsum()
        return obv

    # ------------------------------------------------------------------
    # eski API
    # ------------------------------------------------------------------
    def from_klines(self, df: pd.DataFrame) -> WhaleSignal:
        """
        Geriye dönük uyumlu ana giriş noktası.
        Beklenen kolonlar:
            open, high, low, close, volume
        """
        if df is None or len(df) < self.window + 1:
            return WhaleSignal("none", 0.0, "not_enough_data", {})

        df_feat = self.calculate_features(df)
        return self._generate_signal(df_feat, multiplier=1.0)

    # ------------------------------------------------------------------
    # feature hesaplama
    # ------------------------------------------------------------------
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) == 0:
            return df

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                raise ValueError(f"missing required column: {col}")

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["volume_ma"] = df["volume"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0.0, np.nan)

        vol_mean = df["volume"].rolling(50, min_periods=10).mean()
        vol_std = df["volume"].rolling(50, min_periods=10).std().replace(0.0, np.nan)
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        prev_typ = df["typical_price"].shift(1).replace(0.0, np.nan)
        df["pvt"] = ((df["typical_price"] - prev_typ) / prev_typ) * df["volume"]
        df["pvt"] = df["pvt"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["pvt_cum"] = df["pvt"].cumsum()

        df["body"] = df["close"] - df["open"]
        df["body_size"] = df["body"].abs()
        df["range_size"] = (df["high"] - df["low"]).abs()

        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["wick_ratio"] = (df["upper_wick"] + df["lower_wick"]) / (
            df["body_size"] + 1e-10
        )
        df["body_to_range"] = df["body_size"] / (df["range_size"] + 1e-10)

        df["returns"] = df["close"].pct_change().fillna(0.0)
        df["cumret_3"] = df["returns"].rolling(3, min_periods=1).sum()
        df["cumret_5"] = df["returns"].rolling(5, min_periods=1).sum()

        df["rsi"] = self._rsi(df["close"], window=self.rsi_window)
        df["obv"] = self._calc_obv(df["close"], df["volume"])
        df["obv_delta"] = df["obv"].diff(self.obv_window).fillna(0.0)

        df["volume_spike"] = df["volume_ratio"] > 2.0
        df["climax_volume"] = (
            df["volume"] > df["volume"].rolling(20, min_periods=5).quantile(0.9)
        ) & (
            df["body_size"]
            > df["body_size"].rolling(20, min_periods=5).mean().fillna(0.0) * 1.5
        )

        df["buy_pressure_proxy"] = np.where(
            df["close"] > df["open"],
            df["volume_ratio"].fillna(0.0) * df["body_to_range"].fillna(0.0),
            0.0,
        )
        df["sell_pressure_proxy"] = np.where(
            df["close"] < df["open"],
            df["volume_ratio"].fillna(0.0) * df["body_to_range"].fillna(0.0),
            0.0,
        )

        df["absorption_long"] = (
            (df["lower_wick"] > df["body_size"] * 1.5)
            & (df["volume_ratio"] > 1.5)
            & (df["close"] >= df["open"])
        )
        df["absorption_short"] = (
            (df["upper_wick"] > df["body_size"] * 1.5)
            & (df["volume_ratio"] > 1.5)
            & (df["close"] <= df["open"])
        )

        return df
    # ------------------------------------------------------------------
    # skor parçaları
    # ------------------------------------------------------------------
    def _calculate_volume_score(self, recent: pd.DataFrame, multiplier: float) -> float:
        vol_z = self._safe_float(recent["volume_zscore"].iloc[-1], 0.0)
        vol_ratio = self._safe_float(recent["volume_ratio"].iloc[-1], 1.0)
        raw = ((vol_z / 3.0) * 0.65) + (min(3.0, vol_ratio) / 3.0) * 0.35
        return self._clip01(raw * multiplier)

    def _calculate_price_action_score(self, recent: pd.DataFrame) -> float:
        body_abs = self._safe_float(recent["body_size"].iloc[-1], 0.0)
        body_mean = self._safe_float(
            recent["body_size"].rolling(10, min_periods=1).mean().iloc[-1],
            1e-8,
        )
        wick_ratio = self._safe_float(recent["wick_ratio"].iloc[-1], 0.0)
        body_to_range = self._safe_float(recent["body_to_range"].iloc[-1], 0.0)

        norm_body = body_abs / max(body_mean, 1e-8)
        raw = (norm_body / (1.0 + wick_ratio)) * 0.7 + body_to_range * 0.3
        return self._clip01(raw)

    def _calculate_momentum_score(self, recent: pd.DataFrame) -> float:
        cumret_5 = abs(self._safe_float(recent["cumret_5"].iloc[-1], 0.0))
        obv_delta = abs(self._safe_float(recent["obv_delta"].iloc[-1], 0.0))
        vol_base = self._safe_float(
            recent["volume"].rolling(10, min_periods=1).mean().iloc[-1],
            1.0,
        )

        obv_norm = obv_delta / max(vol_base * 5.0, 1e-8)
        raw = (min(0.03, cumret_5) / 0.03) * 0.6 + min(1.0, obv_norm) * 0.4
        return self._clip01(raw)

    def _calculate_absorption_score(self, recent: pd.DataFrame) -> float:
        wick_ratio = self._safe_float(recent["wick_ratio"].iloc[-1], 0.0)
        vol_ratio = self._safe_float(recent["volume_ratio"].iloc[-1], 0.0)
        body_to_range = self._safe_float(recent["body_to_range"].iloc[-1], 0.0)

        raw = 0.0
        if wick_ratio > 2.5 and vol_ratio > 1.4 and body_to_range < 0.45:
            raw = ((min(8.0, wick_ratio) / 8.0) * 0.6) + (
                (min(3.0, vol_ratio) / 3.0) * 0.4
            )
        return self._clip01(raw)

    def _determine_direction(
        self,
        recent: pd.DataFrame,
        scores: Dict[str, float],
    ) -> Direction:
        last_body = self._safe_float(recent["body"].iloc[-1], 0.0)
        cumret_3 = self._safe_float(recent["cumret_3"].iloc[-1], 0.0)
        buy_p = self._safe_float(recent["buy_pressure_proxy"].tail(3).mean(), 0.0)
        sell_p = self._safe_float(recent["sell_pressure_proxy"].tail(3).mean(), 0.0)

        long_votes = 0
        short_votes = 0

        if last_body > 0:
            long_votes += 1
        elif last_body < 0:
            short_votes += 1

        if cumret_3 > 0:
            long_votes += 1
        elif cumret_3 < 0:
            short_votes += 1

        if buy_p > sell_p:
            long_votes += 1
        elif sell_p > buy_p:
            short_votes += 1

        if (
            scores.get("momentum_score", 0.0) < 0.08
            and scores.get("price_action_score", 0.0) < 0.12
        ):
            return "none"

        if long_votes >= 2 and long_votes > short_votes:
            return "long"
        if short_votes >= 2 and short_votes > long_votes:
            return "short"
        return "none"

    def _generate_reason(self, scores: Dict[str, float], direction: Direction) -> str:
        parts: List[str] = [f"dir={direction}"]
        for k, v in scores.items():
            if float(v) > 0.20:
                parts.append(f"{k}:{float(v):.2f}")
        if len(parts) == 1:
            parts.append("weak_signal")
        return " | ".join(parts)

    def _detect_market_regime_from_recent(self, recent: pd.DataFrame) -> str:
        try:
            vol_ratio = self._safe_float(recent["volume_ratio"].tail(5).mean(), 1.0)
            body_to_range = self._safe_float(
                recent["body_to_range"].tail(5).mean(),
                0.0,
            )
            cumret = abs(self._safe_float(recent["cumret_5"].iloc[-1], 0.0))

            trend_score = (min(1.0, cumret / 0.02) * 0.55) + (
                min(1.0, body_to_range / 0.65) * 0.45
            )
            range_score = (min(1.0, 1.0 - min(1.0, cumret / 0.02)) * 0.60) + (
                min(1.0, vol_ratio / 1.5) * 0.40
            )

            if trend_score >= self.whale_trend_thr:
                return "trend"
            if range_score >= self.whale_range_thr:
                return "range"
            return "neutral"
        except Exception:
            return "neutral"
    def _apply_regime_adjustment(
        self,
        score: float,
        regime: str,
        direction: Direction,
        recent: pd.DataFrame,
    ) -> float:
        s = float(score)

        try:
            buy_p = self._safe_float(recent["buy_pressure_proxy"].tail(3).mean(), 0.0)
            sell_p = self._safe_float(
                recent["sell_pressure_proxy"].tail(3).mean(),
                0.0,
            )

            aligned = (
                (direction == "long" and buy_p >= sell_p)
                or (direction == "short" and sell_p >= buy_p)
            )

            if regime == "trend" and aligned:
                s *= float(self.whale_trend_bonus)

            if regime == "range":
                s *= float(self.whale_range_penalty)

            if aligned and direction in ("long", "short"):
                s *= float(self.whale_xconf_bonus)
        except Exception:
            pass

        return float(self._clip01(s))

    def _action_from_decision_context(
        self,
        regime: str,
        score: float,
        aligned_with_trade: bool,
        opposed_to_trade: bool,
        aligned_with_pos: bool,
        opposed_to_pos: bool,
    ) -> str:
        if regime == "ignore":
            return "ignore"

        if regime == "watch":
            if opposed_to_trade:
                return "avoid_open"
            if opposed_to_pos:
                return "tighten_risk"
            if aligned_with_trade or aligned_with_pos:
                return "soft_confirm"
            return "watch"

        if regime == "boost":
            if opposed_to_trade:
                return "block"
            if aligned_with_trade:
                return "confirm"
            if aligned_with_pos:
                return "hold_winner"
            if opposed_to_pos:
                return "tighten_risk"
            return "boost"

        if regime == "veto":
            if opposed_to_trade:
                return "hard_block"
            if aligned_with_trade:
                return "strong_confirm"
            if aligned_with_pos:
                return "hold_winner"
            if opposed_to_pos:
                if self.whale_force_exit_enable and score >= self.whale_force_exit_thr:
                    return "force_exit"
                return "tighten_risk"
            return "veto"

        return "ignore"
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
            "volume_score": 0.40,
            "price_action_score": 0.24,
            "momentum_score": 0.22,
            "absorption_score": 0.14,
        }

        total_score = sum(float(scores[k]) * float(weights[k]) for k in scores)
        direction = self._determine_direction(recent, scores)
        regime = self._detect_market_regime_from_recent(recent)

        last = recent.iloc[-1]
        meta = {
            "scores": scores,
            "timeframe_multiplier": float(multiplier),
            "last_volume_ratio": self._safe_float(last.get("volume_ratio"), 0.0),
            "last_volume_zscore": self._safe_float(last.get("volume_zscore"), 0.0),
            "last_body": self._safe_float(last.get("body"), 0.0),
            "last_body_to_range": self._safe_float(last.get("body_to_range"), 0.0),
            "last_rsi": self._safe_float(last.get("rsi"), 50.0),
            "buy_pressure_proxy": self._safe_float(
                recent["buy_pressure_proxy"].tail(3).mean(),
                0.0,
            ),
            "sell_pressure_proxy": self._safe_float(
                recent["sell_pressure_proxy"].tail(3).mean(),
                0.0,
            ),
            "regime": regime,
        }

        if direction == "none":
            total_score *= 0.45
        else:
            total_score = self._apply_regime_adjustment(
                score=float(total_score),
                regime=str(regime),
                direction=direction,
                recent=recent,
            )

        return WhaleSignal(
            direction=direction,
            score=float(self._clip01(total_score)),
            reason=self._generate_reason(scores, direction),
            meta=meta,
        )

    # ------------------------------------------------------------------
    # final karar desteği
    # ------------------------------------------------------------------
    def classify_regime(self, signal: WhaleSignal) -> str:
        score = self._safe_float(signal.score, 0.0)
        direction = str(signal.direction or "none").lower()

        if direction not in ("long", "short"):
            return "ignore"

        if score < 0.30:
            return "ignore"

        if score < self.whale_confirm_thr:
            return "watch"

        if score < self.whale_veto_thr:
            return "boost"

        return "veto"
    def final_whale_vote(
        self,
        signal: WhaleSignal,
        trade_side: Optional[str] = None,
        current_position_side: Optional[str] = None,
    ) -> WhaleDecision:
        """
        trade_side:
            açılacak yön adayı ("long"/"short")
        current_position_side:
            açık pozisyon yönü ("long"/"short")
        """
        sig_dir = str(signal.direction or "none").lower()
        score = self._safe_float(signal.score, 0.0)
        regime = self.classify_regime(signal)

        trade_side_n = str(trade_side or "").strip().lower()
        pos_side_n = str(current_position_side or "").strip().lower()

        aligned_with_trade = bool(
            trade_side_n in ("long", "short")
            and sig_dir in ("long", "short")
            and trade_side_n == sig_dir
        )
        opposed_to_trade = bool(
            trade_side_n in ("long", "short")
            and sig_dir in ("long", "short")
            and trade_side_n != sig_dir
        )

        aligned_with_pos = bool(
            pos_side_n in ("long", "short")
            and sig_dir in ("long", "short")
            and pos_side_n == sig_dir
        )
        opposed_to_pos = bool(
            pos_side_n in ("long", "short")
            and sig_dir in ("long", "short")
            and pos_side_n != sig_dir
        )

        action = self._action_from_decision_context(
            regime=regime,
            score=score,
            aligned_with_trade=aligned_with_trade,
            opposed_to_trade=opposed_to_trade,
            aligned_with_pos=aligned_with_pos,
            opposed_to_pos=opposed_to_pos,
        )

        confidence = 0.0
        reason = signal.reason

        if action == "ignore":
            confidence = 0.0
        elif action in ("watch",):
            confidence = score * 0.40
        elif action in ("avoid_open", "soft_block"):
            confidence = score * 0.60
        elif action in ("soft_confirm",):
            confidence = score * 0.55
        elif action in ("boost",):
            confidence = score * 0.75
        elif action in ("confirm", "block", "tighten_risk"):
            confidence = score * 0.90
        elif action in ("strong_confirm", "hard_block", "hold_winner", "force_exit"):
            confidence = min(1.0, score)
        else:
            confidence = score * 0.85

        meta = dict(signal.meta or {})
        meta.update(
            {
                "trade_side": trade_side_n or None,
                "current_position_side": pos_side_n or None,
                "aligned_with_trade": aligned_with_trade,
                "opposed_to_trade": opposed_to_trade,
                "aligned_with_position": aligned_with_pos,
                "opposed_to_position": opposed_to_pos,
                "boost_thr": float(self.whale_confirm_thr),
                "veto_thr": float(self.whale_veto_thr),
                "force_exit_thr": float(self.whale_force_exit_thr),
                "force_exit_min_pnl_pct": float(self.whale_force_exit_min_pnl_pct),
                "force_exit_on_profit_only": bool(self.whale_force_exit_on_profit_only),
                "force_exit_confirm_bars": int(self.whale_force_exit_confirm_bars),
                "block_actions": sorted(self.whale_block_actions),
                "reduce_actions": sorted(self.whale_reduce_actions),
                "boost_actions": sorted(self.whale_boost_actions),
            }
        )
        return WhaleDecision(
            direction=sig_dir if sig_dir in ("long", "short") else "none",
            score=float(score),
            regime=str(regime),
            action=str(action),
            confidence=float(self._clip01(confidence)),
            reason=str(reason),
            meta=meta,
        )

    def whale_policy_hint(
        self,
        signal: WhaleSignal,
        trade_side: Optional[str] = None,
        current_position_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        dec = self.final_whale_vote(
            signal=signal,
            trade_side=trade_side,
            current_position_side=current_position_side,
        )

        out = {
            "whale_dir": dec.direction,
            "whale_score": float(dec.score),
            "whale_regime": dec.regime,
            "whale_action": dec.action,
            "whale_confidence": float(dec.confidence),
            "whale_reason": dec.reason,
            "whale_meta": dec.meta,
        }

        out["whale_should_boost_open"] = (
            dec.action in self.whale_boost_actions
            or dec.action in ("boost", "confirm", "strong_confirm", "soft_confirm")
        )
        out["whale_should_block_open"] = (
            dec.action in self.whale_block_actions
            or dec.action in ("block", "hard_block", "soft_block", "avoid_open")
        )
        out["whale_should_force_exit"] = dec.action == "force_exit"
        out["whale_should_tighten_risk"] = (
            dec.action in self.whale_reduce_actions
            or dec.action == "tighten_risk"
        )
        out["whale_should_reduce"] = dec.action == "reduce_size"
        out["whale_should_hold"] = dec.action == "hold_winner"

        return out

class MultiTimeframeWhaleDetector(WhaleDetector):
    """
    Çoklu timeframe whale dedektörü.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeframes = ["1m", "3m", "5m", "15m", "30m", "1h"]
        self.min_mtf_agreement_score = self._env_float(
            "WHALE_MTF_MIN_AGREEMENT_SCORE",
            0.55,
        )

    def analyze_multiple_timeframes(
        self,
        dfs: Dict[str, pd.DataFrame],
        okx_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Dict[str, WhaleSignal]:
        signals: Dict[str, WhaleSignal] = {}

        if not isinstance(dfs, dict):
            return signals

        for tf, df in dfs.items():
            try:
                if df is None or not isinstance(df, pd.DataFrame):
                    signals[tf] = WhaleSignal(
                        "none",
                        0.0,
                        f"invalid_df_{tf}",
                        {},
                    )
                    continue

                if len(df) < max(60, self.window + 10):
                    signals[tf] = WhaleSignal(
                        "none",
                        0.0,
                        f"insufficient_data_{tf}",
                        {},
                    )
                    continue

                df_features = self.calculate_features(df)

                tf_multiplier = {
                    "1m": 0.90,
                    "3m": 0.95,
                    "5m": 1.00,
                    "15m": 1.10,
                    "30m": 1.18,
                    "1h": 1.28,
                    "4h": 1.45,
                }.get(str(tf), 1.0)

                signals[tf] = self._generate_signal(
                    df_features,
                    multiplier=tf_multiplier,
                )
            except Exception as e:
                signals[tf] = WhaleSignal(
                    "none",
                    0.0,
                    f"mtf_error_{tf}",
                    {"error": str(e)},
                )

        return signals

    def aggregate_mtf_decision(
        self,
        dfs: Dict[str, pd.DataFrame],
        trade_side: Optional[str] = None,
        current_position_side: Optional[str] = None,
    ) -> WhaleDecision:
        signals = self.analyze_multiple_timeframes(dfs)
        if not signals:
            return WhaleDecision(
                direction="none",
                score=0.0,
                regime="ignore",
                action="ignore",
                confidence=0.0,
                reason="no_signals",
                meta={},
            )

        weights = {
            "1m": 0.60,
            "3m": 0.80,
            "5m": 1.00,
            "15m": 1.15,
            "30m": 1.25,
            "1h": 1.35,
            "4h": 1.50,
        }

        long_score = 0.0
        short_score = 0.0
        total_w = 0.0
        per_tf: Dict[str, Any] = {}

        for tf, sig in signals.items():
            w = float(weights.get(tf, 1.0))
            total_w += w

            if sig.direction == "long":
                long_score += float(sig.score) * w
            elif sig.direction == "short":
                short_score += float(sig.score) * w

            per_tf[tf] = {
                "direction": sig.direction,
                "score": float(sig.score),
                "reason": sig.reason,
                "meta": sig.meta,
                "weight": w,
            }

        if total_w <= 0:
            total_w = 1.0

        long_score_n = float(long_score / total_w)
        short_score_n = float(short_score / total_w)
        consensus_thr = float(self.min_mtf_agreement_score)

        if long_score_n > short_score_n and long_score_n >= consensus_thr:
            base_signal = WhaleSignal(
                direction="long",
                score=float(self._clip01(long_score_n)),
                reason="mtf_long_alignment",
                meta={
                    "per_tf": per_tf,
                    "long_score": float(long_score_n),
                    "short_score": float(short_score_n),
                    "consensus_thr": float(consensus_thr),
                },
            )
        elif short_score_n > long_score_n and short_score_n >= consensus_thr:
            base_signal = WhaleSignal(
                direction="short",
                score=float(self._clip01(short_score_n)),
                reason="mtf_short_alignment",
                meta={
                    "per_tf": per_tf,
                    "long_score": float(long_score_n),
                    "short_score": float(short_score_n),
                    "consensus_thr": float(consensus_thr),
                },
            )
        else:
            base_signal = WhaleSignal(
                direction="none",
                score=float(self._clip01(max(long_score_n, short_score_n))),
                reason="mtf_no_consensus",
                meta={
                    "per_tf": per_tf,
                    "long_score": float(long_score_n),
                    "short_score": float(short_score_n),
                    "consensus_thr": float(consensus_thr),
                },
            )
        decision = self.final_whale_vote(
            signal=base_signal,
            trade_side=trade_side,
            current_position_side=current_position_side,
        )
        decision.meta["signals"] = signals
        return decision


try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except Exception:
    IsolationForest = None  # type: ignore
    StandardScaler = None  # type: ignore


class MLWhaleDetector(WhaleDetector):
    """
    IsolationForest ile anomaly-based whale dedektörü.
    sklearn yoksa çalışmaz ama ana botu bozmaz.
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
            "body_to_range",
            "buy_pressure_proxy",
            "sell_pressure_proxy",
        ]
        existing = [c for c in cols if c in df_feat.columns]
        return df_feat[existing].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
        x = self.scaler.fit_transform(features)

        self.model = IsolationForest(
            contamination=0.08,
            random_state=42,
            n_estimators=120,
        )
        self.model.fit(x)

    def detect_with_ml(self, df: pd.DataFrame) -> WhaleSignal:
        if self.model is None or self.scaler is None:
            return WhaleSignal("none", 0.0, "model_not_trained", {})

        features = self._extract_features_for_ml(df.tail(self.window))
        x = self.scaler.transform(features)
        scores = self.model.decision_function(x)
        preds = self.model.predict(x)

        anomaly_score = float(scores[-1])
        is_anomaly = bool(preds[-1] == -1)

        if is_anomaly and abs(anomaly_score) > 0.20:
            last_feat = features.iloc[-1]
            pvt_last = float(last_feat.get("pvt", 0.0))
            direction: Direction = "long" if pvt_last > 0 else "short"
            score = float(self._clip01(abs(anomaly_score) * 2.0))
            return WhaleSignal(
                direction=direction,
                score=score,
                reason=f"ml_anomaly_detected_{anomaly_score:.3f}",
                meta={
                    "anomaly_score": anomaly_score,
                    "features": last_feat.to_dict(),
                },
            )

        return WhaleSignal("none", 0.0, "no_anomaly_detected", {})


class OrderFlowWhaleDetector(WhaleDetector):
    """
    Order-flow proxy analizleri (VWAP, volume cluster vs.)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _find_volume_clusters(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        clusters: List[Dict[str, Any]] = []
        if len(df) == 0:
            return clusters
        low_min = float(df["low"].min())
        high_max = float(df["high"].max())
        if high_max <= low_min:
            return clusters

        price_bins = np.linspace(low_min, high_max, 20)
        vol_mean = float(df["volume"].mean() or 1e-8)

        for i in range(len(price_bins) - 1):
            lo = float(price_bins[i])
            hi = float(price_bins[i + 1])
            mask = (df["low"] >= lo) & (df["high"] <= hi)
            if not bool(mask.any()):
                continue
            cluster_volume = float(df.loc[mask, "volume"].sum())
            if cluster_volume > vol_mean * threshold:
                clusters.append(
                    {
                        "price_range": (lo, hi),
                        "volume": cluster_volume,
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
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (
            (df["volume"] * df["typical_price"]).cumsum()
            / df["volume"].cumsum()
        )
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"] * 100.0
        df["price_change"] = df["close"].pct_change().fillna(0.0)
        df["volume_delta"] = df["volume"] * np.sign(df["price_change"])

        recent = df.tail(20)
        buy_volume = float(recent.loc[recent["close"] > recent["open"], "volume"].sum())
        sell_volume = float(recent.loc[recent["close"] < recent["open"], "volume"].sum())
        total_volume = buy_volume + sell_volume

        buy_pressure = float(buy_volume / total_volume) if total_volume > 0 else 0.0
        sell_pressure = float(sell_volume / total_volume) if total_volume > 0 else 0.0
        clusters = self._find_volume_clusters(df)

        return {
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "vwap_deviation": float(recent["vwap_deviation"].iloc[-1]),
            "volume_clusters": clusters,
        }


class RealTimeWhaleMonitor:
    """
    Akış bazlı whale takibi için helper.
    """

    def __init__(self, detector: WhaleDetector, alert_threshold: float = 0.7):
        self.detector = detector
        self.alert_threshold = alert_threshold
        self.signal_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.alert_callbacks: List[Any] = []
    async def monitor_stream(self, data_stream):
        async for df in data_stream:
            signal = self.detector.from_klines(df)
            self.signal_history.append(
                {"timestamp": datetime.utcnow(), "signal": signal}
            )

            if float(signal.score) >= float(self.alert_threshold):
                await self._trigger_alerts(signal)

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    async def _trigger_alerts(self, signal: WhaleSignal) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "direction": signal.direction,
            "score": float(signal.score),
            "reason": signal.reason,
            "meta": signal.meta,
        }
        for cb in self.alert_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(payload)
            else:
                cb(payload)


class WhaleDetectorBacktester:
    def __init__(self, detector: WhaleDetector):
        self.detector = detector

    def backtest(self, df: pd.DataFrame, lookahead_bars: int = 10) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        max_i = len(df) - self.detector.window - lookahead_bars
        for i in range(max(0, max_i)):
            window_data = df.iloc[i : i + self.detector.window]
            signal = self.detector.from_klines(window_data)
            if signal.direction == "none":
                continue
            future_data = df.iloc[i + self.detector.window : i + self.detector.window + lookahead_bars]
            performance = self._calculate_performance(signal, window_data, future_data)
            results.append({"signal": signal, "performance": performance, "index": int(i + self.detector.window)})
        return self._analyze_results(results)

    def _calculate_performance(self, signal: WhaleSignal, window_data: pd.DataFrame, future_data: pd.DataFrame) -> float:
        entry_price = float(window_data["close"].iloc[-1])
        exit_price = float(future_data["close"].iloc[-1])
        if signal.direction == "long":
            return float((exit_price - entry_price) / entry_price)
        if signal.direction == "short":
            return float((entry_price - exit_price) / entry_price)
        return 0.0

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"n_signals": 0, "avg_return": 0.0, "win_rate": 0.0, "returns": []}
        rets = np.array([r["performance"] for r in results], dtype=float)
        return {
            "n_signals": int(len(results)),
            "avg_return": float(rets.mean()),
            "win_rate": float((rets > 0).mean()),
            "returns": rets.tolist(),
        }
