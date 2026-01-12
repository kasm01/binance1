from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

Direction = Literal["long", "short", "none"]


@dataclass
class WhaleSignal:
    direction: Direction
    score: float
    reason: str
    meta: Dict[str, Any]


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _linear_slope(y: np.ndarray) -> float:
    """
    Basit slope: y ~ a*x + b (least squares).
    Return a (slope). y length < 3 => 0
    """
    try:
        n = int(len(y))
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=float)
        x = x - x.mean()
        y = y.astype(float) - float(y.mean())
        denom = float((x * x).sum()) + 1e-12
        return float((x * y).sum() / denom)
    except Exception:
        return 0.0


def _ensure_cols_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out


def _hour_utc_from_market_meta(market_meta: Optional[Dict[str, Any]]) -> Optional[int]:
    """
    Time-of-day filter için UTC hour üretir.
    Beklenen alanlar:
      - market_meta["hour_utc"] (0..23) veya
      - market_meta["ts_ms"] (epoch ms) veya market_meta["ts"] (epoch seconds)
    """
    if not market_meta:
        return None
    h = market_meta.get("hour_utc", None)
    if h is not None:
        try:
            hh = int(h)
            if 0 <= hh <= 23:
                return hh
        except Exception:
            pass

    ts_ms = market_meta.get("ts_ms", None)
    if ts_ms is not None:
        try:
            ts_ms = float(ts_ms)
            dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
            return int(dt.hour)
        except Exception:
            pass

    ts = market_meta.get("ts", None)
    if ts is not None:
        try:
            ts = float(ts)
            dt = pd.to_datetime(ts, unit="s", utc=True)
            return int(dt.hour)
        except Exception:
            pass

    return None


# ---------------------------------------------------------
# FEATURE ENGINE: Whale analizi için ortak hesaplayıcı
# ---------------------------------------------------------
class WhaleFeatureEngine:
    """
    Whale analizi için gerekli proxy metrikleri hesaplar.

    Üretilen kolonlar:
      - body, dir, impact
      - vol_ema, vol_ratio, vol_z
      - body_strength, dir_mom
      - taker_buy, taker_sell (proxy)
      - flow, flow_ratio, flow_z
      - cvd, cvd_slope
      - cvd_delta_k, cvd_break_z
      - atr_proxy, vol_regime
      - ma_fast, ma_slow, ma_slope
      - vwap, vwap_dev
    """

    def __init__(
        self,
        vol_ema_span: int = 20,
        vol_z_window: int = 30,
        body_strength_window: int = 20,
        dir_mom_window: int = 10,
        cvd_slope_window: int = 30,
        cvd_break_k: int = 5,
        cvd_break_z_window: int = 60,
        ma_fast: int = 20,
        ma_slow: int = 50,
        atr_window: int = 14,
        vwap_window: int = 50,
    ) -> None:
        self.vol_ema_span = int(vol_ema_span)
        self.vol_z_window = int(vol_z_window)
        self.body_strength_window = int(body_strength_window)
        self.dir_mom_window = int(dir_mom_window)
        self.cvd_slope_window = int(cvd_slope_window)
        self.cvd_break_k = int(cvd_break_k)
        self.cvd_break_z_window = int(cvd_break_z_window)
        self.ma_fast = int(ma_fast)
        self.ma_slow = int(ma_slow)
        self.atr_window = int(atr_window)
        self.vwap_window = int(vwap_window)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 5:
            return df

        out = df.copy()

        out = _ensure_cols_numeric(out, ["open", "high", "low", "close", "volume", "taker_buy_base_volume"])

        out["body"] = out["close"] - out["open"]
        out["dir"] = np.where(out["body"] > 0, 1, np.where(out["body"] < 0, -1, 0))
        out["impact"] = out["body"].abs() / (out["close"].abs() + 1e-9)

        vol = out["volume"].astype(float)
        out["vol_ema"] = vol.ewm(span=self.vol_ema_span, adjust=False).mean()
        out["vol_ratio"] = vol / (out["vol_ema"] + 1e-9)

        v_mean = vol.rolling(self.vol_z_window).mean()
        v_std = vol.rolling(self.vol_z_window).std()
        out["vol_z"] = (vol - v_mean) / (v_std + 1e-9)

        body_abs = out["body"].abs().astype(float)
        out["body_strength"] = body_abs / (body_abs.rolling(self.body_strength_window).mean() + 1e-9)
        out["dir_mom"] = out["dir"].rolling(self.dir_mom_window).sum()

        # Aggressive flow proxy
        if "taker_buy_base_volume" in out.columns:
            taker_buy = out["taker_buy_base_volume"].astype(float)
            taker_sell = (out["volume"].astype(float) - taker_buy).clip(lower=0.0)
        else:
            taker_buy = (out["volume"].astype(float) * (out["dir"].clip(lower=0))).astype(float)
            taker_sell = (out["volume"].astype(float) * ((-out["dir"]).clip(lower=0))).astype(float)

        out["taker_buy"] = taker_buy
        out["taker_sell"] = taker_sell
        out["flow"] = (taker_buy - taker_sell).astype(float)
        out["flow_ratio"] = taker_buy / (out["volume"].astype(float) + 1e-9)

        flow = out["flow"].astype(float)
        f_mean = flow.rolling(self.cvd_break_z_window).mean()
        f_std = flow.rolling(self.cvd_break_z_window).std()
        out["flow_z"] = (flow - f_mean) / (f_std + 1e-9)

        out["cvd"] = out["flow"].cumsum()

        cvd_vals = out["cvd"].astype(float).values
        n = self.cvd_slope_window
        if len(out) >= n:
            slopes = np.zeros(len(out), dtype=float)
            for i in range(len(out)):
                j0 = max(0, i - n + 1)
                slopes[i] = _linear_slope(cvd_vals[j0 : i + 1])
            out["cvd_slope"] = slopes
        else:
            out["cvd_slope"] = 0.0

        k = max(1, self.cvd_break_k)
        out["cvd_delta_k"] = out["cvd"] - out["cvd"].shift(k)
        d = out["cvd_delta_k"].astype(float)
        d_mean = d.rolling(self.cvd_break_z_window).mean()
        d_std = d.rolling(self.cvd_break_z_window).std()
        out["cvd_break_z"] = (d - d_mean) / (d_std + 1e-9)

        # ATR proxy + vol regime
        if {"high", "low", "close"}.issubset(out.columns):
            high = out["high"].astype(float)
            low = out["low"].astype(float)
            close = out["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            out["atr_proxy"] = tr.rolling(self.atr_window).mean().fillna(0.0)
            out["vol_regime"] = out["atr_proxy"] / (close.abs() + 1e-9)
        else:
            out["atr_proxy"] = 0.0
            out["vol_regime"] = 0.0

        # Trend regime proxy
        close = out["close"].astype(float)
        out["ma_fast"] = close.rolling(self.ma_fast).mean()
        out["ma_slow"] = close.rolling(self.ma_slow).mean()
        kk = max(3, int(self.ma_fast / 4))
        out["ma_slope"] = (out["ma_fast"] - out["ma_fast"].shift(kk)) / (kk + 1e-9)

        # VWAP + deviation
        # typical price * vol rolling / vol rolling
        tp = (out["high"].astype(float) + out["low"].astype(float) + out["close"].astype(float)) / 3.0
        pv = tp * vol
        vwap = pv.rolling(self.vwap_window).sum() / (vol.rolling(self.vwap_window).sum() + 1e-9)
        out["vwap"] = vwap.fillna(method="ffill").fillna(0.0)
        out["vwap_dev"] = (out["close"].astype(float) - out["vwap"].astype(float)) / (out["vwap"].abs() + 1e-9)

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

# ---------------------------------------------------------
# Single TF Whale Detector (asıl model)
# ---------------------------------------------------------
class WhaleDetector:
    """
    Asıl whale dedektörü (tek timeframe).

    Eklenenler:
      - OBI: market_meta["obi"] varsa flow ile birlikte score’a ek
      - Spread shock: spread z-score (rolling) ile ani açılma veto/penalty
      - VWAP deviation: continuation/fakeout ayrımı
      - Time-of-day: düşük likidite saatlerinde eşikleri yükselt
    """

    def __init__(
        self,
        window: int = 80,
        # spike thresholds (base)
        volume_zscore_thr: float = 1.5,
        flow_z_thr: float = 1.8,
        cvd_break_z_thr: float = 1.8,
        body_strength_thr: float = 1.2,
        impact_thr: float = 0.0012,  # ~0.12%
        # cvd
        cvd_slope_min_abs: float = 0.0,
        # regime
        trend_ma_slope_thr: float = 0.0,
        vol_regime_thr: float = 0.003,
        range_penalty: float = 0.70,
        # tradeability
        max_spread_pct: float = 0.0015,
        min_liq_score: float = 0.0,
        hard_veto_untradeable: bool = False,
        # spread shock
        spread_z_window: int = 40,
        spread_z_thr: float = 2.2,
        hard_veto_spread_shock: bool = True,
        # OBI usage
        obi_weight: float = 0.12,          # score bonus/penalty scale
        obi_align_thr: float = 0.15,       # |obi| < thr => ignore
        # VWAP deviation
        vwap_dev_thr: float = 0.0012,      # 0.12% deviation
        vwap_weight: float = 0.10,
        # time-of-day
        tod_low_liq_hours_utc: Optional[List[int]] = None,
        tod_thr_multiplier: float = 1.15,  # low-liq hours => eşikler * multiplier
        # cooldown/decay
        cooldown_bars: int = 10,
        decay_halflife_bars: int = 25,
    ) -> None:
        self.window = int(window)

        self.volume_zscore_thr = float(volume_zscore_thr)
        self.flow_z_thr = float(flow_z_thr)
        self.cvd_break_z_thr = float(cvd_break_z_thr)
        self.body_strength_thr = float(body_strength_thr)
        self.impact_thr = float(impact_thr)

        self.cvd_slope_min_abs = float(cvd_slope_min_abs)

        self.trend_ma_slope_thr = float(trend_ma_slope_thr)
        self.vol_regime_thr = float(vol_regime_thr)
        self.range_penalty = float(range_penalty)

        self.max_spread_pct = float(max_spread_pct)
        self.min_liq_score = float(min_liq_score)
        self.hard_veto_untradeable = bool(hard_veto_untradeable)

        self.spread_z_window = int(spread_z_window)
        self.spread_z_thr = float(spread_z_thr)
        self.hard_veto_spread_shock = bool(hard_veto_spread_shock)

        self.obi_weight = float(obi_weight)
        self.obi_align_thr = float(obi_align_thr)

        self.vwap_dev_thr = float(vwap_dev_thr)
        self.vwap_weight = float(vwap_weight)

        self.tod_low_liq_hours_utc = tod_low_liq_hours_utc or [0, 1, 2, 3, 4, 5]
        self.tod_thr_multiplier = float(tod_thr_multiplier)

        self.cooldown_bars = int(cooldown_bars)
        self.decay_halflife_bars = int(decay_halflife_bars)

        self.fe = WhaleFeatureEngine()

        # state keyed by (symbol, interval)
        self._state: Dict[str, Dict[str, Any]] = {}

    def _key(self, symbol: Optional[str], interval: Optional[str]) -> str:
        s = (symbol or "default").upper()
        itv = (interval or "na").lower()
        return f"{s}:{itv}"

    def _apply_cooldown_decay(self, key: str, base_score: float, bar_index: int) -> Tuple[float, Dict[str, Any]]:
        st = self._state.get(key) or {}
        last_bar = int(st.get("last_event_bar", -10**9))
        bars_since = int(bar_index - last_bar)

        cooldown_active = bars_since < self.cooldown_bars
        if cooldown_active:
            score = base_score * 0.25
        else:
            score = base_score

        if bars_since > 0 and self.decay_halflife_bars > 0:
            decay = float(0.5 ** (bars_since / float(self.decay_halflife_bars)))
            persist_penalty = max(0.35, min(1.0, 1.0 - (1.0 - decay) * 0.6))
            score = score * persist_penalty
        else:
            persist_penalty = 1.0

        meta = {
            "cooldown_active": bool(cooldown_active),
            "bars_since_event": int(bars_since),
            "persist_penalty": float(persist_penalty),
        }
        return float(_clip01(score)), meta

    def _mark_event(self, key: str, bar_index: int) -> None:
        st = self._state.get(key) or {}
        st["last_event_bar"] = int(bar_index)
        self._state[key] = st

    # ---- spread shock state ----
    def _update_spread_stats(self, key: str, spread_pct: float) -> float:
        """
        Rolling spread z-score hesaplar (state içinde).
        Return: spread_z (float)
        """
        st = self._state.get(key) or {}
        arr = st.get("spread_hist", None)
        if not isinstance(arr, list):
            arr = []
        arr.append(float(spread_pct))
        # cap
        cap = max(10, self.spread_z_window)
        if len(arr) > cap:
            arr = arr[-cap:]
        st["spread_hist"] = arr
        self._state[key] = st

        if len(arr) < 10:
            return 0.0
        a = np.array(arr, dtype=float)
        mu = float(a.mean())
        sd = float(a.std()) + 1e-12
        z = (float(spread_pct) - mu) / sd
        return float(z)

    def from_klines(
        self,
        df: pd.DataFrame,
        *,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        market_meta: Optional[Dict[str, Any]] = None,
    ) -> WhaleSignal:
        if df is None or len(df) < max(10, self.window):
            return WhaleSignal("none", 0.0, "not_enough_data", {})

        key = self._key(symbol, interval)

        # time-of-day multiplier
        hour_utc = _hour_utc_from_market_meta(market_meta)
        low_liq = (hour_utc in set(self.tod_low_liq_hours_utc)) if hour_utc is not None else False
        thr_mul = self.tod_thr_multiplier if low_liq else 1.0

        tail = df.tail(self.window).copy()
        feat = self.fe.compute(tail)
        if feat is None or feat.empty:
            return WhaleSignal("none", 0.0, "feature_engine_failed", {})

        last = feat.iloc[-1]
        bar_index = len(df) - 1

        vol_z = _safe_float(last.get("vol_z", 0.0))
        flow_z = _safe_float(last.get("flow_z", 0.0))
        cvd_break_z = _safe_float(last.get("cvd_break_z", 0.0))
        body_strength = _safe_float(last.get("body_strength", 0.0))
        impact = _safe_float(last.get("impact", 0.0))
        cvd_slope = _safe_float(last.get("cvd_slope", 0.0))
        ma_slope = _safe_float(last.get("ma_slope", 0.0))
        vol_regime = _safe_float(last.get("vol_regime", 0.0))
        dir_raw = int(_safe_float(last.get("dir", 0.0), 0.0))

        flow = _safe_float(last.get("flow", 0.0))
        flow_sign = 1 if flow > 0 else (-1 if flow < 0 else 0)

        vwap_dev = _safe_float(last.get("vwap_dev", 0.0))

        # Tradeability + spread shock
        mm = market_meta or {}
        spread_pct = _safe_float(mm.get("spread_pct", 0.0), 0.0)
        liq_score = _safe_float(mm.get("liq_score", 0.0), 0.0)
        spread_ok = (spread_pct <= self.max_spread_pct) if spread_pct > 0 else True
        liq_ok = (liq_score >= self.min_liq_score) if self.min_liq_score > 0 else True
        tradeable_ok = bool(spread_ok and liq_ok)

        spread_z = 0.0
        spread_shock = False
        if spread_pct and spread_pct > 0:
            spread_z = self._update_spread_stats(key, spread_pct)
            spread_shock = (spread_z >= self.spread_z_thr)

        if self.hard_veto_untradeable and not tradeable_ok:
            return WhaleSignal("none", 0.0, "untradeable_veto", {"tradeable_ok": False, "spread_pct": spread_pct, "liq_score": liq_score})

        if self.hard_veto_spread_shock and spread_shock:
            return WhaleSignal("none", 0.0, "spread_shock_veto", {"spread_pct": spread_pct, "spread_z": spread_z, "hour_utc": hour_utc, "low_liq": low_liq})

        # thresholds (with time-of-day multiplier)
        vol_thr = self.volume_zscore_thr * thr_mul
        flow_thr = self.flow_z_thr * thr_mul
        cvd_thr = self.cvd_break_z_thr * thr_mul
        body_thr = self.body_strength_thr * thr_mul
        impact_thr = self.impact_thr * thr_mul

        meta: Dict[str, Any] = {
            "vol_z": float(vol_z),
            "flow_z": float(flow_z),
            "cvd_break_z": float(cvd_break_z),
            "body_strength": float(body_strength),
            "impact": float(impact),
            "cvd_slope": float(cvd_slope),
            "ma_slope": float(ma_slope),
            "vol_regime": float(vol_regime),
            "dir_raw": int(dir_raw),
            "flow": float(flow),
            "flow_sign": int(flow_sign),
            "flow_ratio": float(_safe_float(last.get("flow_ratio", 0.0))),
            "vwap_dev": float(vwap_dev),
            "spread_pct": float(spread_pct),
            "spread_z": float(spread_z),
            "spread_shock": bool(spread_shock),
            "liq_score": float(liq_score),
            "tradeable_ok": bool(tradeable_ok),
            "hour_utc": hour_utc,
            "low_liq_hours": bool(low_liq),
            "thr_multiplier": float(thr_mul),
        }
        if market_meta:
            meta["market_meta"] = market_meta

        # Spike gates
        has_vol_spike = vol_z >= vol_thr
        has_flow_spike = abs(flow_z) >= flow_thr
        has_cvd_break = abs(cvd_break_z) >= cvd_thr
        has_body_strength = body_strength >= body_thr
        has_impact = impact >= impact_thr

        # Regime
        trendish = abs(ma_slope) > (self.trend_ma_slope_thr if self.trend_ma_slope_thr > 0 else 0.0)
        high_vol = vol_regime >= self.vol_regime_thr
        meta["trendish"] = bool(trendish)
        meta["high_vol"] = bool(high_vol)

        # Pre-filter
        if not (has_vol_spike or has_flow_spike):
            return WhaleSignal("none", 0.0, "no_spike", meta)

        # Absorption / fakeout: flow sign ↔ candle dir ters
        absorption = False
        if flow_sign != 0 and dir_raw != 0:
            if (flow_sign > 0 and dir_raw < 0) or (flow_sign < 0 and dir_raw > 0):
                absorption = True
        meta["absorption_flag"] = bool(absorption)

        # Direction proposal
        if has_flow_spike and flow_sign != 0:
            dir_candidate = "long" if flow_sign > 0 else "short"
        elif abs(cvd_slope) > 0:
            dir_candidate = "long" if cvd_slope > 0 else "short"
        else:
            dir_candidate = "long" if dir_raw > 0 else ("short" if dir_raw < 0 else "none")

        # Signature:
        signature_a = bool(has_flow_spike and has_cvd_break and (abs(cvd_slope) >= self.cvd_slope_min_abs))
        signature_b = bool(has_vol_spike and has_body_strength and has_impact)
        if not (signature_a or signature_b):
            return WhaleSignal("none", 0.0, "weak_signature", meta)

        # Score components
        vol_comp = _clip01((vol_z - vol_thr) / 2.0) if has_vol_spike else 0.0
        flow_comp = _clip01((abs(flow_z) - flow_thr) / 2.0) if has_flow_spike else 0.0
        cvd_comp = _clip01(abs(cvd_break_z) / (cvd_thr * 2.0)) if has_cvd_break else 0.0
        impact_comp = _clip01(impact / (impact_thr * 2.0)) if impact_thr > 0 else 0.0
        body_comp = _clip01((body_strength - 1.0) / 2.0) if has_body_strength else 0.0
        slope_comp = _clip01(abs(cvd_slope) / (abs(cvd_slope) + 1.0))

        base_score = (
            0.33 * max(vol_comp, flow_comp)
            + 0.25 * cvd_comp
            + 0.14 * impact_comp
            + 0.14 * body_comp
            + 0.14 * slope_comp
        )

        # VWAP deviation logic
        # - long continuation: vwap_dev > +thr
        # - short continuation: vwap_dev < -thr
        vwap_ok = False
        if dir_candidate == "long" and vwap_dev >= self.vwap_dev_thr:
            vwap_ok = True
        elif dir_candidate == "short" and vwap_dev <= -self.vwap_dev_thr:
            vwap_ok = True

        if vwap_ok:
            base_score = min(1.0, base_score + self.vwap_weight * _clip01(abs(vwap_dev) / (self.vwap_dev_thr * 2.0)))
            meta["vwap_confirm"] = True
        else:
            # ters sapma = fakeout olasılığı
            if dir_candidate == "long" and vwap_dev <= -self.vwap_dev_thr:
                base_score *= 0.80
                meta["vwap_conflict"] = True
            elif dir_candidate == "short" and vwap_dev >= self.vwap_dev_thr:
                base_score *= 0.80
                meta["vwap_conflict"] = True
            else:
                meta["vwap_conflict"] = False
            meta["vwap_confirm"] = False

        # OBI integration (market_meta["obi"] expected in [-1..+1])
        obi = _safe_float(mm.get("obi", 0.0), 0.0)
        meta["obi"] = float(obi)
        obi_used = False
        if abs(obi) >= self.obi_align_thr and flow_sign != 0 and self.obi_weight > 0:
            obi_used = True
            # aligned if sign(obi) matches direction candidate
            if dir_candidate == "long":
                aligned = obi > 0
            elif dir_candidate == "short":
                aligned = obi < 0
            else:
                aligned = False

            mag = _clip01(abs(obi))
            if aligned:
                base_score = min(1.0, base_score + self.obi_weight * mag)
                meta["obi_aligned"] = True
            else:
                base_score *= (1.0 - min(0.25, self.obi_weight) * mag)
                meta["obi_aligned"] = False
        meta["obi_used"] = bool(obi_used)

        # Absorption penalty
        if absorption:
            base_score *= 0.55

        # Regime handling
        if trendish and high_vol:
            base_score = min(1.0, base_score + 0.08)
            meta["regime_bonus"] = True
        else:
            base_score *= self.range_penalty * (0.80 if absorption else 1.0)
            meta["regime_bonus"] = False
            meta["range_penalty"] = True

        # Spread shock soft penalty (hard veto zaten yukarıda)
        if spread_shock:
            base_score *= 0.65
            meta["spread_shock_penalty"] = True
        else:
            meta["spread_shock_penalty"] = False

        # Tradeability soft penalty
        if not tradeable_ok:
            base_score *= 0.70
            meta["untradeable_soft_penalty"] = True
        else:
            meta["untradeable_soft_penalty"] = False

        base_score = _clip01(base_score)

        # Cooldown & decay
        score, cd_meta = self._apply_cooldown_decay(key, base_score, bar_index)
        meta.update(cd_meta)

        direction: Direction = "none"
        if dir_candidate in ("long", "short") and score > 0.0:
            direction = dir_candidate  # type: ignore[assignment]

        reason = "cvd_flow_signature" if signature_a else "volume_body_signature"
        if absorption:
            reason += "_absorption"
        if spread_shock:
            reason += "_spreadshock"
        if not tradeable_ok:
            reason += "_untradeable"
        if low_liq:
            reason += "_tod_lowliq"
        if meta.get("cooldown_active"):
            reason += "_cooldown"

        # Mark event
        if direction != "none" and score >= 0.55 and tradeable_ok and not meta.get("cooldown_active"):
            self._mark_event(key, bar_index)
            meta["event_marked"] = True
        else:
            meta["event_marked"] = False

        return WhaleSignal(direction, float(score), reason, meta)

# ---------------------------------------------------------
# MTF Whale Detector (aggregator + optional cross-exchange confirm)
# ---------------------------------------------------------
class MultiTimeframeWhaleDetector:
    """
    Çoklu timeframe whale dedektörü.

    API:
      - from_klines(df) (backwards compat)
      - analyze_multiple_timeframes(dfs, okx_dfs=None, market_meta_by_tf=None, symbol=None)
      - get_mtf_meta(...)
    """

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        base_window: int = 80,
        volume_zscore_thr: float = 1.5,
        flow_z_thr: float = 1.8,
        cvd_break_z_thr: float = 1.8,
    ) -> None:
        self.timeframes: List[str] = timeframes or ["1m", "3m", "5m", "15m", "30m", "1h"]

        self.detectors: Dict[str, WhaleDetector] = {}
        for tf in self.timeframes:
            w = base_window + 40 if tf.endswith("h") else base_window
            self.detectors[tf] = WhaleDetector(
                window=w,
                volume_zscore_thr=volume_zscore_thr,
                flow_z_thr=flow_z_thr,
                cvd_break_z_thr=cvd_break_z_thr,
            )

        self.base_detector = WhaleDetector(
            window=base_window,
            volume_zscore_thr=volume_zscore_thr,
            flow_z_thr=flow_z_thr,
            cvd_break_z_thr=cvd_break_z_thr,
        )

    def from_klines(self, df: pd.DataFrame, interval: Optional[str] = None) -> WhaleSignal:
        if interval and interval in self.detectors:
            return self.detectors[interval].from_klines(df, interval=interval)
        return self.base_detector.from_klines(df)

    def analyze_multiple_timeframes(
        self,
        dfs: Dict[str, pd.DataFrame],
        *,
        okx_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        market_meta_by_tf: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, WhaleSignal]:
        signals: Dict[str, WhaleSignal] = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 10:
                signals[tf] = WhaleSignal("none", 0.0, f"not_enough_data_{tf}", {})
                continue

            det = self.detectors.get(tf, self.base_detector)
            mmeta = (market_meta_by_tf or {}).get(tf)

            sig = det.from_klines(df, symbol=symbol, interval=tf, market_meta=mmeta)

            # Cross-exchange confirm (OKX)
            if okx_dfs is not None and tf in okx_dfs and okx_dfs[tf] is not None and len(okx_dfs[tf]) >= 10:
                try:
                    okx_sig = det.from_klines(
                        okx_dfs[tf],
                        symbol=(symbol or "NA"),
                        interval=f"okx:{tf}",
                        market_meta=None,
                    )
                    aligned = (sig.direction != "none") and (sig.direction == okx_sig.direction) and (okx_sig.score >= 0.45)
                    if aligned:
                        boosted = min(1.0, float(sig.score) + 0.10 * max(0.0, okx_sig.score))
                        sig = WhaleSignal(
                            sig.direction,
                            boosted,
                            sig.reason + "_xex_confirm",
                            {**sig.meta, "xex_confirm": True, "okx_score": float(okx_sig.score), "okx_dir": okx_sig.direction},
                        )
                    else:
                        if okx_sig.direction in ("long", "short") and sig.direction in ("long", "short") and okx_sig.direction != sig.direction and okx_sig.score >= 0.45:
                            penal = float(sig.score) * 0.70
                        else:
                            penal = float(sig.score) * 0.92
                        sig = WhaleSignal(
                            sig.direction,
                            penal,
                            sig.reason,
                            {**sig.meta, "xex_confirm": False, "okx_score": float(okx_sig.score), "okx_dir": okx_sig.direction},
                        )
                except Exception:
                    sig = WhaleSignal(sig.direction, sig.score, sig.reason, {**sig.meta, "xex_confirm": None})

            signals[tf] = sig

        return signals

    def get_mtf_meta(
        self,
        dfs: Dict[str, pd.DataFrame],
        *,
        okx_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        market_meta_by_tf: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        signals = self.analyze_multiple_timeframes(
            dfs,
            okx_dfs=okx_dfs,
            market_meta_by_tf=market_meta_by_tf,
            symbol=symbol,
        )

        tf_weights = {
            "1m": 1.0,
            "3m": 1.1,
            "5m": 1.2,
            "15m": 1.5,
            "30m": 1.7,
            "1h": 2.0,
        }

        long_score = 0.0
        short_score = 0.0
        per_tf_meta: Dict[str, Any] = {}

        for tf, sig in signals.items():
            w = float(tf_weights.get(tf, 1.0))
            s = _clip01(float(sig.score))

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
            agg_dir: Direction = "none"
            agg_score = 0.0
        else:
            if long_score >= short_score:
                agg_dir = "long"
                agg_score_raw = long_score / (long_score + short_score + 1e-9)
            else:
                agg_dir = "short"
                agg_score_raw = short_score / (long_score + short_score + 1e-9)

            agg_score = float(_clip01((agg_score_raw - 0.5) * 2.0))

        return {
            "direction": agg_dir,
            "score": float(agg_score),
            "per_tf": per_tf_meta,
        }
