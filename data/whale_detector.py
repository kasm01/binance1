from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, List, Tuple

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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(lo)


def _ensure_cols_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    return out


def _extract_ts_seconds_from_df(df: pd.DataFrame) -> Optional[float]:
    """
    open_time saniye veya ms olabilir. Saniyeye normalize etmeye çalışır.
    """
    if df is None or df.empty:
        return None
    if "open_time" not in df.columns:
        return None
    t = _safe_float(df["open_time"].iloc[-1], default=np.nan)
    if not (t == t):
        return None
    # ms ise genelde 1e12 civarı
    if t > 3e10:
        return float(t / 1000.0)
    return float(t)


def _hour_utc_from_ts(ts_sec: Optional[float]) -> Optional[int]:
    if ts_sec is None:
        return None
    try:
        # pandas kullanmadan basit UTC hour
        dt = pd.to_datetime(ts_sec, unit="s", utc=True)
        return int(dt.hour)
    except Exception:
        return None


def _parse_hours_list(s: str) -> List[int]:
    """
    "0,1,2,23" gibi -> [0,1,2,23]
    """
    out: List[int] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        try:
            h = int(float(p))
            if 0 <= h <= 23:
                out.append(h)
        except Exception:
            continue
    return sorted(list(set(out)))


def _time_of_day_multiplier(
    hour_utc: Optional[int],
    *,
    low_liq_hours: Optional[List[int]] = None,
    low_liq_mult: float = 1.25,
) -> Tuple[float, Dict[str, Any]]:
    """
    Düşük likidite saatlerinde threshold'ları yükseltmek için çarpan döner.
    """
    if low_liq_hours is None:
        # Varsayılan: Asya açılışı öncesi + gün kapanışı civarı (kabaca)
        low_liq_hours = [0, 1, 2, 3, 22, 23]

    low = False
    if hour_utc is not None and hour_utc in set(low_liq_hours):
        low = True

    mult = float(low_liq_mult) if low else 1.0
    meta = {
        "hour_utc": hour_utc,
        "low_liq_hours": list(low_liq_hours),
        "tod_low_liq": bool(low),
        "tod_mult": float(mult),
    }
    return float(mult), meta


def _obi_to_unit(obi: float) -> float:
    """
    OBI normalize:
      - OBI [0..1] ise => (obi-0.5)*2 -> [-1..1]
      - OBI [-1..1] ise aynen
    return: unit in [-1..1]
    """
    v = float(obi)
    if v < -1.2 or v > 1.2:
        # çok uç değer => clamp
        return float(max(-1.0, min(1.0, v)))
    if 0.0 <= v <= 1.0:
        return float((v - 0.5) * 2.0)
    return float(max(-1.0, min(1.0, v)))


# ---------------------------------------------------------
# FEATURE ENGINE
# ---------------------------------------------------------
class WhaleFeatureEngine:
    """
    Whale analizi için gerekli proxy metrikleri hesaplar.

    Üretilen kolonlar:
      - body, dir
      - vol_ema, vol_ratio, vol_z
      - body_strength, dir_mom
      - taker_buy, taker_sell (proxy)
      - flow = taker_buy - taker_sell
      - flow_ratio = taker_buy / (volume + eps)
      - cvd = cumsum(flow)
      - cvd_slope_last (son N)
      - flow_z (son bar ani kırılma)
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
        flow_z_window: int = 60,
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
        self.flow_z_window = int(flow_z_window)
        self.ma_fast = int(ma_fast)
        self.ma_slow = int(ma_slow)
        self.atr_window = int(atr_window)
        self.vwap_window = int(vwap_window)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 5:
            return df

        out = df.copy()

        # Core numeric
        out = _ensure_cols_numeric(out, ["open", "high", "low", "close", "volume", "taker_buy_base_volume"])

        # Candle body / dir
        out["body"] = out["close"] - out["open"]
        out["dir"] = np.where(out["body"] > 0, 1, np.where(out["body"] < 0, -1, 0))

        # Volume ratio + Z
        vol = out["volume"].astype(float)
        out["vol_ema"] = vol.ewm(span=self.vol_ema_span, adjust=False).mean()
        out["vol_ratio"] = vol / (out["vol_ema"] + 1e-9)

        v_mean = vol.rolling(self.vol_z_window).mean()
        v_std = vol.rolling(self.vol_z_window).std()
        out["vol_z"] = (vol - v_mean) / (v_std + 1e-9)

        # Body strength
        body_abs = np.abs(out["body"].astype(float))
        out["body_strength"] = body_abs / (body_abs.rolling(self.body_strength_window).mean() + 1e-9)

        # Direction momentum
        out["dir_mom"] = out["dir"].rolling(self.dir_mom_window).sum()

        # Aggressive flow proxy (taker_buy - taker_sell)
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

        # CVD
        out["cvd"] = out["flow"].cumsum()

        # CVD slope (sadece last için, hızlı)
        n = int(self.cvd_slope_window)
        cvd_vals = out["cvd"].astype(float).values
        if len(cvd_vals) >= 3:
            j0 = max(0, len(cvd_vals) - n)
            y = cvd_vals[j0:]
            if len(y) >= 3:
                x = np.arange(len(y), dtype=float)
                x = x - x.mean()
                y = y - float(np.mean(y))
                denom = float((x * x).sum()) + 1e-12
                slope = float((x * y).sum() / denom)
            else:
                slope = 0.0
        else:
            slope = 0.0
        out["cvd_slope_last"] = float(slope)

        # Flow z-score (ani kırılma)
        flow = out["flow"].astype(float)
        f_mean = flow.rolling(self.flow_z_window).mean()
        f_std = flow.rolling(self.flow_z_window).std()
        out["flow_z"] = (flow - f_mean) / (f_std + 1e-9)

        # ATR proxy + volatility regime
        if {"high", "low", "close"}.issubset(out.columns):
            high = out["high"].astype(float)
            low = out["low"].astype(float)
            close = out["close"].astype(float)
            prev_close = close.shift(1)
            tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            out["atr_proxy"] = tr.rolling(self.atr_window).mean().fillna(0.0)
            out["vol_regime"] = out["atr_proxy"] / (close.abs() + 1e-9)
        else:
            out["atr_proxy"] = 0.0
            out["vol_regime"] = 0.0

        # Trend regime proxy: MA slope
        close = out["close"].astype(float)
        out["ma_fast"] = close.rolling(self.ma_fast).mean()
        out["ma_slow"] = close.rolling(self.ma_slow).mean()
        k = max(3, int(self.ma_fast / 4))
        out["ma_slope"] = (out["ma_fast"] - out["ma_fast"].shift(k)) / (k + 1e-9)

        # VWAP + deviation
        # vwap = sum(tp*vol)/sum(vol), tp=(h+l+c)/3
        if {"high", "low", "close", "volume"}.issubset(out.columns):
            tp = (out["high"].astype(float) + out["low"].astype(float) + out["close"].astype(float)) / 3.0
            pv = tp * out["volume"].astype(float)
            sum_pv = pv.rolling(self.vwap_window).sum()
            sum_v = out["volume"].astype(float).rolling(self.vwap_window).sum()
            out["vwap"] = (sum_pv / (sum_v + 1e-9)).fillna(0.0)
            out["vwap_dev"] = (out["close"].astype(float) - out["vwap"].astype(float)) / (out["vwap"].abs() + 1e-9)
        else:
            out["vwap"] = 0.0
            out["vwap_dev"] = 0.0

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out

# ---------------------------------------------------------
# Single TF Whale Detector (asıl model)
# ---------------------------------------------------------
class WhaleDetector:
    """
    Asıl whale dedektörü (tek timeframe).

    Eklenen güçlendirmeler:
      ✅ Order-book imbalance (OBI): market_meta["obi"] varsa score’a katkı
      ✅ Spread shock: market_meta["spread_z"] / ["spread_zscore"] ile veto/penalty
      ✅ VWAP deviation: flow yönü ile vwap_dev uyumu continuation/fakeout ayrımı
      ✅ Time-of-day filter: düşük likidite saatlerinde threshold yükselt

    Not:
      - spread_z hesaplamasını main/WS tarafında tutmak daha mantıklı.
        (burada sadece "gelirse uygula" mantığı var)
    """

    def __init__(
        self,
        window: int = 80,
        # spike thresholds
        volume_zscore_thr: float = 1.5,
        flow_z_thr: float = 1.8,
        body_strength_thr: float = 1.2,
        # cvd
        cvd_slope_min_abs: float = 0.0,
        # regime
        trend_ma_slope_thr: float = 0.0,
        vol_regime_thr: float = 0.003,
        # tradeability
        max_spread_pct: float = 0.0015,   # 0.15%
        min_liq_score: float = 0.0,
        # spread shock z
        spread_z_veto_thr: float = 2.2,
        spread_z_penalty_thr: float = 1.6,
        # OBI
        obi_weight: float = 0.10,
        # VWAP
        vwap_weight: float = 0.10,
        vwap_dev_thr: float = 0.0015,     # 0.15% dev eşiği
        # TOD (time-of-day)
        low_liq_mult: float = 1.25,
        # cooldown/decay
        cooldown_bars: int = 10,
        decay_halflife_bars: int = 25,
    ) -> None:
        self.window = int(window)

        self.volume_zscore_thr = float(volume_zscore_thr)
        self.flow_z_thr = float(flow_z_thr)
        self.body_strength_thr = float(body_strength_thr)

        self.cvd_slope_min_abs = float(cvd_slope_min_abs)

        self.trend_ma_slope_thr = float(trend_ma_slope_thr)
        self.vol_regime_thr = float(vol_regime_thr)

        self.max_spread_pct = float(max_spread_pct)
        self.min_liq_score = float(min_liq_score)

        self.spread_z_veto_thr = float(spread_z_veto_thr)
        self.spread_z_penalty_thr = float(spread_z_penalty_thr)

        self.obi_weight = float(obi_weight)

        self.vwap_weight = float(vwap_weight)
        self.vwap_dev_thr = float(vwap_dev_thr)

        self.low_liq_mult = float(low_liq_mult)

        self.cooldown_bars = int(cooldown_bars)
        self.decay_halflife_bars = int(decay_halflife_bars)

        self.fe = WhaleFeatureEngine()

        # state keyed by (symbol, interval) if given; else "default"
        self._state: Dict[str, Dict[str, Any]] = {}

        # env override: low liquidity hours
        self._env_low_liq_hours = None
        try:
            s = os.getenv("WHALE_LOW_LIQ_HOURS", "").strip()
            if s:
                self._env_low_liq_hours = _parse_hours_list(s)
        except Exception:
            self._env_low_liq_hours = None

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

        # decay: event sonrası tekrar sinyalde "persist" gibi davranmasın diye penalty
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
        return float(_clamp(score)), meta

    def _mark_event(self, key: str, bar_index: int) -> None:
        st = self._state.get(key) or {}
        st["last_event_bar"] = int(bar_index)
        self._state[key] = st

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

        tail = df.tail(self.window).copy()
        feat = self.fe.compute(tail)
        if feat is None or feat.empty:
            return WhaleSignal("none", 0.0, "feature_engine_failed", {})

        last = feat.iloc[-1]
        bar_index = len(df) - 1

        # -------------------------
        # Time-of-day filter (threshold multipliers)
        # -------------------------
        mmeta = market_meta or {}
        hour_utc = None
        if "hour_utc" in mmeta:
            hour_utc = int(_safe_float(mmeta.get("hour_utc"), default=-1))
            hour_utc = hour_utc if 0 <= hour_utc <= 23 else None
        else:
            ts_sec = _extract_ts_seconds_from_df(df)
            hour_utc = _hour_utc_from_ts(ts_sec)

        tod_mult, tod_meta = _time_of_day_multiplier(
            hour_utc,
            low_liq_hours=self._env_low_liq_hours,
            low_liq_mult=self.low_liq_mult,
        )

        # TOD uygulanan efektif eşikler
        vol_thr_eff = float(self.volume_zscore_thr * tod_mult)
        flow_thr_eff = float(self.flow_z_thr * tod_mult)
        body_thr_eff = float(self.body_strength_thr)  # body'i şişirmiyoruz (isteğe bağlı)

        # -------------------------
        # Core features
        # -------------------------
        vol_z = _safe_float(last.get("vol_z", 0.0))
        flow_z = _safe_float(last.get("flow_z", 0.0))
        body_strength = _safe_float(last.get("body_strength", 0.0))
        cvd_slope = _safe_float(last.get("cvd_slope_last", 0.0))
        ma_slope = _safe_float(last.get("ma_slope", 0.0))
        vol_regime = _safe_float(last.get("vol_regime", 0.0))
        flow = _safe_float(last.get("flow", 0.0))
        vwap_dev = _safe_float(last.get("vwap_dev", 0.0))
        dir_raw = int(_safe_float(last.get("dir", 0.0), 0.0))

        # -------------------------
        # Tradeability (spread/liquidity)
        # -------------------------
        spread_pct = _safe_float(mmeta.get("spread_pct", 0.0), 0.0)
        liq_score = _safe_float(mmeta.get("liq_score", 0.0), 0.0)
        spread_ok = (spread_pct <= self.max_spread_pct) if spread_pct > 0 else True
        liq_ok = (liq_score >= self.min_liq_score) if self.min_liq_score > 0 else True

        # Spread shock z-score (ani açılma)
        spread_z = None
        if "spread_z" in mmeta:
            spread_z = _safe_float(mmeta.get("spread_z"), default=np.nan)
        elif "spread_zscore" in mmeta:
            spread_z = _safe_float(mmeta.get("spread_zscore"), default=np.nan)
        elif "spread_z_score" in mmeta:
            spread_z = _safe_float(mmeta.get("spread_z_score"), default=np.nan)

        spread_shock_veto = False
        spread_shock_penalty = False
        if spread_z is not None and (spread_z == spread_z):
            if spread_z >= self.spread_z_veto_thr:
                spread_shock_veto = True
            elif spread_z >= self.spread_z_penalty_thr:
                spread_shock_penalty = True

        tradeable_ok = bool(spread_ok and liq_ok and (not spread_shock_veto))

        # -------------------------
        # Spike gates (TOD 적용)
        # -------------------------
        has_vol_spike = vol_z >= vol_thr_eff
        has_flow_break = flow_z >= flow_thr_eff
        has_body_strength = body_strength >= body_thr_eff

        # Regime
        trendish = abs(ma_slope) > self.trend_ma_slope_thr if self.trend_ma_slope_thr > 0 else (abs(ma_slope) > 0.0)
        high_vol = vol_regime >= self.vol_regime_thr

        # Direction candidate: flow sign öncelik
        flow_sign = 1 if flow > 0 else (-1 if flow < 0 else 0)
        if flow_sign != 0:
            dir_candidate: Direction = "long" if flow_sign > 0 else "short"
        elif cvd_slope != 0:
            dir_candidate = "long" if cvd_slope > 0 else "short"
        else:
            dir_candidate = "none"

        # CVD slope strength
        cvd_ok = abs(cvd_slope) >= self.cvd_slope_min_abs if self.cvd_slope_min_abs > 0 else True

        meta: Dict[str, Any] = {
            "vol_z": float(vol_z),
            "flow_z": float(flow_z),
            "body_strength": float(body_strength),
            "cvd_slope": float(cvd_slope),
            "ma_slope": float(ma_slope),
            "vol_regime": float(vol_regime),
            "trendish": bool(trendish),
            "high_vol": bool(high_vol),
            "dir_raw": int(dir_raw),
            "flow": float(flow),
            "flow_ratio": float(_safe_float(last.get("flow_ratio", 0.0))),
            "vwap_dev": float(vwap_dev),
            "spread_pct": float(spread_pct),
            "liq_score": float(liq_score),
            "spread_z": float(spread_z) if (spread_z is not None and spread_z == spread_z) else None,
            "spread_shock_veto": bool(spread_shock_veto),
            "spread_shock_penalty": bool(spread_shock_penalty),
            "tradeable_ok": bool(tradeable_ok),
            "tod": tod_meta,
            "thr_eff": {"vol_z": float(vol_thr_eff), "flow_z": float(flow_thr_eff), "body_strength": float(body_thr_eff)},
        }

        # -------------------------
        # Pre-filter
        # -------------------------
        if not (has_vol_spike or has_flow_break):
            return WhaleSignal("none", 0.0, "no_spike", meta)

        # Signature: (flow_break + cvd_ok) veya (vol_spike + body_strength)
        signature_a = bool(has_flow_break and cvd_ok)
        signature_b = bool(has_vol_spike and has_body_strength)

        if not (signature_a or signature_b):
            return WhaleSignal("none", 0.0, "weak_signature", meta)

        # -------------------------
        # Score compose (base)
        # -------------------------
        vol_comp = _clamp(vol_z / 3.0)
        flow_comp = _clamp(flow_z / 4.0)
        body_comp = _clamp((body_strength - 1.0) / 2.0)
        slope_comp = _clamp(abs(cvd_slope) / (abs(cvd_slope) + 1.0))

        base_score = 0.0
        base_score += 0.35 * max(vol_comp, flow_comp)
        base_score += 0.25 * slope_comp
        base_score += 0.20 * body_comp

        # Regime bonus: trend + high_vol => continuation ihtimali artar
        if trendish and high_vol:
            base_score += 0.10
            meta["regime_bonus"] = True
        else:
            meta["regime_bonus"] = False

        # -------------------------
        # OBI contribution (market_meta["obi"])
        # -------------------------
        obi_raw = mmeta.get("obi", None)
        if obi_raw is not None:
            obi_unit = _obi_to_unit(_safe_float(obi_raw, default=0.0))  # [-1..1]
            # yön uyumu
            if dir_candidate == "long":
                obi_align = max(0.0, obi_unit)
            elif dir_candidate == "short":
                obi_align = max(0.0, -obi_unit)
            else:
                obi_align = 0.0

            base_score += float(self.obi_weight) * _clamp(obi_align)
            meta["obi"] = float(_safe_float(obi_raw, default=0.0))
            meta["obi_unit"] = float(obi_unit)
            meta["obi_align"] = float(obi_align)
        else:
            meta["obi"] = None

        # -------------------------
        # VWAP deviation contribution (continuation/fakeout)
        # -------------------------
        # long continuation: close > vwap (vwap_dev positive)
        # short continuation: close < vwap (vwap_dev negative)
        vwap_bonus = 0.0
        if dir_candidate in ("long", "short"):
            if dir_candidate == "long":
                # vwap_dev >= thr -> continuation bonus
                if vwap_dev >= self.vwap_dev_thr:
                    vwap_bonus = _clamp((vwap_dev - self.vwap_dev_thr) / (5.0 * self.vwap_dev_thr + 1e-9))
                # vwap_dev çok negatifse fakeout penalty
                elif vwap_dev <= -self.vwap_dev_thr:
                    vwap_bonus = -0.60 * _clamp((abs(vwap_dev) - self.vwap_dev_thr) / (5.0 * self.vwap_dev_thr + 1e-9))
            else:
                if vwap_dev <= -self.vwap_dev_thr:
                    vwap_bonus = _clamp((abs(vwap_dev) - self.vwap_dev_thr) / (5.0 * self.vwap_dev_thr + 1e-9))
                elif vwap_dev >= self.vwap_dev_thr:
                    vwap_bonus = -0.60 * _clamp((vwap_dev - self.vwap_dev_thr) / (5.0 * self.vwap_dev_thr + 1e-9))

        base_score += float(self.vwap_weight) * float(vwap_bonus)
        meta["vwap_bonus_raw"] = float(vwap_bonus)

        base_score = float(_clamp(base_score))

        # -------------------------
        # Spread shock penalty (veto zaten tradeable_ok içinde)
        # -------------------------
        if spread_shock_penalty:
            base_score *= 0.65
            meta["spread_shock_penalty_applied"] = True
        else:
            meta["spread_shock_penalty_applied"] = False

        # -------------------------
        # Tradeability penalty/veto
        # -------------------------
        if not tradeable_ok:
            base_score *= 0.20
            meta["tradeability_penalty"] = True
        else:
            meta["tradeability_penalty"] = False

        # -------------------------
        # Cooldown & decay
        # -------------------------
        key = self._key(symbol, interval)
        score, cd_meta = self._apply_cooldown_decay(key, base_score, bar_index)
        meta.update(cd_meta)

        # Final direction
        direction: Direction = "none"
        if dir_candidate in ("long", "short") and score > 0.0:
            direction = dir_candidate

        reason = "cvd_flow_signature" if signature_a else "volume_body_signature"
        if spread_shock_veto:
            reason += "_spreadshock_veto"
        elif spread_shock_penalty:
            reason += "_spreadshock_penalty"
        if not tradeable_ok:
            reason += "_untradeable_penalty"
        if meta.get("tod", {}).get("tod_low_liq"):
            reason += "_tod_lowliq"
        if meta.get("cooldown_active"):
            reason += "_cooldown"

        # Event mark (güçlü sinyal)
        if direction != "none" and score >= 0.55 and tradeable_ok and not meta.get("cooldown_active"):
            self._mark_event(key, bar_index)
            meta["event_marked"] = True
        else:
            meta["event_marked"] = False

        return WhaleSignal(direction, float(score), reason, meta)


# ---------------------------------------------------------
# MTF Whale Detector
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
    ) -> None:
        self.timeframes: List[str] = timeframes or ["1m", "3m", "5m", "15m", "30m", "1h"]

        self.detectors: Dict[str, WhaleDetector] = {}
        for tf in self.timeframes:
            w = base_window
            if tf.endswith("h"):
                w = base_window + 40
            elif tf.endswith("m"):
                w = base_window

            self.detectors[tf] = WhaleDetector(
                window=w,
                volume_zscore_thr=volume_zscore_thr,
                flow_z_thr=flow_z_thr,
            )

        self.base_detector = WhaleDetector(
            window=base_window,
            volume_zscore_thr=volume_zscore_thr,
            flow_z_thr=flow_z_thr,
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
        """
        dfs: {"1m": df_1m, "5m": df_5m, ...} (Binance)
        okx_dfs (ops): aynı TF'ler için OKX df'leri (cross-confirm)
        market_meta_by_tf (ops): {"1m": {"spread_pct":..,"liq_score":..,"obi":..,"spread_z":..}, ...}
        """
        signals: Dict[str, WhaleSignal] = {}

        for tf, df in dfs.items():
            if df is None or len(df) < 10:
                signals[tf] = WhaleSignal("none", 0.0, f"not_enough_data_{tf}", {})
                continue

            det = self.detectors.get(tf, self.base_detector)
            mmeta = (market_meta_by_tf or {}).get(tf)

            sig = det.from_klines(df, symbol=symbol, interval=tf, market_meta=mmeta)

            # Cross-exchange confirmation bonus (opsiyonel)
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
                        boosted = min(1.0, float(sig.score) + 0.10)
                        sig = WhaleSignal(
                            sig.direction,
                            boosted,
                            sig.reason + "_xex_confirm",
                            {**sig.meta, "xex_confirm": True, "okx_score": okx_sig.score},
                        )
                    else:
                        sig = WhaleSignal(
                            sig.direction,
                            float(sig.score) * 0.92,
                            sig.reason,
                            {**sig.meta, "xex_confirm": False, "okx_score": okx_sig.score},
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
        """
        Çoklu TF sinyallerinden tek aggregated whale_meta üretir.
        """
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
            s = _clamp(float(sig.score))

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

            # 0.5->0, 1.0->1 ölçeğine map
            agg_score = float(_clamp((agg_score_raw - 0.5) * 2.0))

        return {
            "direction": agg_dir,
            "score": float(agg_score),
            "per_tf": per_tf_meta,
        }

