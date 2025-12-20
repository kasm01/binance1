import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from config.load_env import load_environment_variables
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from models.hybrid_inference import HybridModel
from core.hybrid_mtf import MultiTimeframeHybridEnsemble
from data.whale_detector import MultiTimeframeWhaleDetector
from data.anomaly_detection import AnomalyDetector

# main.py helper'ları
from main import build_features, compute_atr_from_klines


# ==========================================================
# Atomic CSV writer (boş/yarım dosya kalmasın)
# ==========================================================
def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # df boşsa header bile yazmıyordu -> 1 byte dosya üretiyordu.
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = None

    return df[cols]


# ==========================================================
# Backtest Stats
# ==========================================================
@dataclass
class BacktestStats:
    starting_equity: float = 1000.0
    equity: float = 1000.0
    peak_equity: float = 1000.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0

    def on_pnl_delta(self, delta: float) -> None:
        if delta == 0.0:
            return

        self.equity += delta
        self.peak_equity = max(self.peak_equity, self.equity)

        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, dd)

        self.n_trades += 1
        if delta > 0:
            self.n_wins += 1
        else:
            self.n_losses += 1

    def summary_dict(self) -> Dict[str, float]:
        pnl = self.equity - self.starting_equity
        return {
            "starting_equity": self.starting_equity,
            "ending_equity": self.equity,
            "pnl": pnl,
            "pnl_pct": (pnl / self.starting_equity) * 100 if self.starting_equity else 0.0,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "winrate": (self.n_wins / self.n_trades) * 100 if self.n_trades else 0.0,
            "max_drawdown_pct": self.max_drawdown * 100.0,
        }


# ==========================================================
# Globals
# ==========================================================
system_logger: Optional[logging.Logger] = None
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]

# main.py ile uyumlu feature set
FEATURE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "hl_range",
    "oc_change",
    "return_1",
    "return_3",
    "return_5",
    "ma_5",
    "ma_10",
    "ma_20",
    "vol_10",
    "dummy_extra",
]


def get_bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else str(v).strip().lower() in ("1", "true", "yes", "on")


# ==========================================================
# Offline kline loader
# ==========================================================
def load_offline_klines(symbol: str, interval: str, limit: Optional[int]) -> pd.DataFrame:
    path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if limit and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df


def _safe_bar_time_iso(raw_df: pd.DataFrame, i: int) -> str:
    if "open_time" not in raw_df.columns:
        return str(i)
    try:
        ot = raw_df["open_time"].iloc[i]
        if pd.isna(ot):
            return str(i)
        return datetime.utcfromtimestamp(float(ot) / 1000.0).isoformat()
    except Exception:
        return str(i)


def _extract_mtf_p_last(mtf_debug: Any, itv: str) -> Optional[float]:
    try:
        if not isinstance(mtf_debug, dict):
            return None
        per = mtf_debug.get("per_interval", {})
        if not isinstance(per, dict):
            return None
        d = per.get(itv, {})
        if not isinstance(d, dict):
            return None
        v = d.get("p_last", None)
        return float(v) if v is not None else None
    except Exception:
        return None


def _prep_X(feat_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in feat_df.columns]
    if not cols:
        return pd.DataFrame()

    X = feat_df[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return X


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ==========================================================
# Backtest Core
# ==========================================================
async def run_backtest() -> None:
    global system_logger

    symbol = os.getenv("BT_SYMBOL", "BTCUSDT")
    main_interval = os.getenv("BT_MAIN_INTERVAL", "5m")
    data_limit = int(os.getenv("BT_DATA_LIMIT", "500"))

    HYBRID_MODE = get_bool_env("HYBRID_MODE", True)
    USE_MTF_ENS = get_bool_env("USE_MTF_ENS", True)

    warmup = int(os.getenv("BT_WARMUP_BARS", "200"))
    atr_period = int(os.getenv("ATR_PERIOD", "14"))
    long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
    short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))

    # Whale Step-1 knobs
    WHALE_FILTER = get_bool_env("BT_WHALE_FILTER", False)
    WHALE_ONLY = get_bool_env("BT_WHALE_ONLY", False)
    WHALE_THR = float(os.getenv("BT_WHALE_THR", "0.50"))
    WHALE_VETO_OPPOSED = get_bool_env("BT_WHALE_VETO_OPPOSED", False)

    OPPOSED_SCALE = float(os.getenv("BT_WHALE_OPPOSED_SCALE", "0.30"))
    ALIGNED_BOOST = float(os.getenv("BT_WHALE_ALIGNED_BOOST", "1.00"))

    # Ensemble-p sizing knobs
    ENS_SIZE_MODE = os.getenv("BT_ENS_SIZE_MODE", "off").strip().lower()  # off | linear
    ENS_MIN_P = float(os.getenv("BT_MIN_ENSEMBLE_P", "0.0"))
    ENS_MAX_BOOST = float(os.getenv("BT_ENS_SIZE_MAX_BOOST", "2.0"))
    ENS_MIN_SCALE = float(os.getenv("BT_ENS_SIZE_MIN_SCALE", "0.2"))

    if main_interval not in MTF_INTERVALS:
        raise ValueError(f"BT_MAIN_INTERVAL={main_interval} MTF_INTERVALS içinde olmalı: {MTF_INTERVALS}")

    system_logger.info(
        "[BT] start | symbol=%s main_interval=%s HYBRID_MODE=%s USE_MTF_ENS=%s warmup=%d limit=%d",
        symbol, main_interval, HYBRID_MODE, USE_MTF_ENS, warmup, data_limit
    )
    system_logger.info(
        "[BT] whale | FILTER=%s ONLY=%s THR=%.2f VETO_OPPOSED=%s opposed_scale=%.2f aligned_boost=%.2f",
        WHALE_FILTER, WHALE_ONLY, WHALE_THR, WHALE_VETO_OPPOSED, OPPOSED_SCALE, ALIGNED_BOOST
    )
    system_logger.info(
        "[BT] ens_sizing | mode=%s min_p=%.3f min_scale=%.2f max_boost=%.2f",
        ENS_SIZE_MODE, ENS_MIN_P, ENS_MIN_SCALE, ENS_MAX_BOOST
    )

    # --------------------------------------------------
    # Load data + features + anomaly
    # --------------------------------------------------
    anomaly_detector = AnomalyDetector(logger=system_logger)
    raw_by_interval: Dict[str, pd.DataFrame] = {}
    feat_by_interval: Dict[str, pd.DataFrame] = {}

    alias_map = {
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }

    for itv in MTF_INTERVALS:
        raw = load_offline_klines(symbol, itv, data_limit)
        feat = build_features(raw)

        for old_col, new_col in alias_map.items():
            if old_col not in feat.columns and new_col in feat.columns:
                feat[old_col] = feat[new_col]

        feat = anomaly_detector.filter_anomalies(feat)

        raw_by_interval[itv] = raw.reset_index(drop=True)
        feat_by_interval[itv] = feat.reset_index(drop=True)

    min_len = min(len(v) for v in feat_by_interval.values())
    if min_len < 200:
        raise RuntimeError(f"Backtest için veri yetersiz (min_len={min_len})")

    if warmup >= min_len:
        warmup = max(50, min_len // 3)

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    mtf_models: Dict[str, HybridModel] = {}
    main_model: Optional[HybridModel] = None

    for itv in MTF_INTERVALS:
        m = HybridModel(model_dir="models", interval=itv, logger=system_logger)
        try:
            if hasattr(m, "use_lstm_hybrid"):
                m.use_lstm_hybrid = HYBRID_MODE
        except Exception:
            pass

        mtf_models[itv] = m
        system_logger.info("[BT] HybridModel loaded | interval=%s", itv)
        if itv == main_interval:
            main_model = m

    if main_model is None:
        raise RuntimeError("Main interval için HybridModel bulunamadı.")

    mtf_ensemble = MultiTimeframeHybridEnsemble(mtf_models)

    whale_detector: Optional[MultiTimeframeWhaleDetector] = None
    try:
        whale_detector = MultiTimeframeWhaleDetector()
        system_logger.info("[BT-WHALE] detector init OK")
    except Exception as e:
        system_logger.warning("[BT-WHALE] init hata: %s (whale kapalı)", e)
        whale_detector = None

    # --------------------------------------------------
    # Risk & Executor
    # --------------------------------------------------
    equity_start_of_day = float(os.getenv("BT_EQUITY_START_OF_DAY", "1000"))

    # Backtestte "max_consecutive_losses=5" yüzünden kilitlenmesin diye varsayılanı 999
    max_consecutive_losses = int(os.getenv("BT_MAX_CONSECUTIVE_LOSSES", "999"))

    risk_manager = RiskManager(
        daily_max_loss_usdt=float(os.getenv("BT_DAILY_MAX_LOSS_USDT", "100")),
        daily_max_loss_pct=float(os.getenv("BT_DAILY_MAX_LOSS_PCT", "0.03")),
        max_consecutive_losses=max_consecutive_losses,
        max_open_trades=int(os.getenv("BT_MAX_OPEN_TRADES", "3")),
        equity_start_of_day=equity_start_of_day,
        logger=system_logger,
    )

    base_order_notional = float(os.getenv("BT_BASE_ORDER_NOTIONAL", "50"))

    trade_executor = TradeExecutor(
        client=None,
        risk_manager=risk_manager,
        position_manager=None,
        logger=system_logger,
        dry_run=True,
        base_order_notional=base_order_notional,
        max_position_notional=float(os.getenv("BT_MAX_POSITION_NOTIONAL", "500")),
        max_leverage=float(os.getenv("BT_MAX_LEVERAGE", "3")),
        sl_pct=float(os.getenv("BT_SL_PCT", "0.01")),
        tp_pct=float(os.getenv("BT_TP_PCT", "0.02")),
        trailing_pct=float(os.getenv("BT_TRAILING_PCT", "0.01")),
        use_atr_sltp=get_bool_env("BT_USE_ATR_SLTP", True),
        atr_sl_mult=float(os.getenv("BT_ATR_SL_MULT", "1.5")),
        atr_tp_mult=float(os.getenv("BT_ATR_TP_MULT", "3.0")),
        whale_risk_boost=float(os.getenv("BT_WHALE_RISK_BOOST", "2.0")),
    )

    # --------------------------------------------------
    # CSV Buffers + bt_context
    # --------------------------------------------------
    equity_rows: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []

    bt_context: Dict[str, Any] = {
        "bar": None,
        "time": None,
        "signal": None,
        "price": None,
        "p_used": None,
        "p_single": None,
        "p_1m": None,
        "p_5m": None,
        "p_15m": None,
        "p_1h": None,
        "whale_dir": None,
        "whale_score": None,
        "whale_on": None,
        "whale_alignment": None,
        "whale_thr": None,
        "model_confidence_factor": None,
        "ens_scale": None,
        "ens_notional": None,
        "atr": None,
    }

    # Patch close_position → closed_trades’e whale_* + bt_* + ens_* ekle
    if hasattr(trade_executor, "_close_position"):
        try:
            orig = trade_executor._close_position  # type: ignore[attr-defined]

            def patched(symbol: str, price: float, reason: str, interval: str):
                closed = orig(symbol=symbol, price=price, reason=reason, interval=interval)
                if isinstance(closed, dict):
                    closed["bt_symbol"] = symbol
                    closed["bt_interval"] = interval

                    closed["bt_bar"] = bt_context.get("bar")
                    closed["bt_time"] = bt_context.get("time")
                    closed["bt_signal"] = bt_context.get("signal")
                    closed["bt_price"] = bt_context.get("price")
                    closed["bt_p_used"] = bt_context.get("p_used")
                    closed["bt_p_single"] = bt_context.get("p_single")
                    closed["bt_p_1m"] = bt_context.get("p_1m")
                    closed["bt_p_5m"] = bt_context.get("p_5m")
                    closed["bt_p_15m"] = bt_context.get("p_15m")
                    closed["bt_p_1h"] = bt_context.get("p_1h")

                    # whale + sizing meta
                    closed["whale_dir"] = closed.get("whale_dir", bt_context.get("whale_dir"))
                    closed["whale_score"] = closed.get("whale_score", bt_context.get("whale_score"))
                    closed["whale_on"] = bt_context.get("whale_on")
                    closed["whale_alignment"] = bt_context.get("whale_alignment")
                    closed["whale_thr"] = bt_context.get("whale_thr")
                    closed["model_confidence_factor"] = bt_context.get("model_confidence_factor")

                    closed["ens_scale"] = bt_context.get("ens_scale")
                    closed["ens_notional"] = bt_context.get("ens_notional")

                    closed["bt_atr"] = bt_context.get("atr")

                    closed_trades.append(closed)
                return closed

            trade_executor._close_position = patched  # type: ignore[method-assign]
            system_logger.info("[BT] TradeExecutor._close_position patch OK (trades içine whale_* + ens_* yazılacak).")
        except Exception as e:
            system_logger.warning("[BT] TradeExecutor patch başarısız (trades whale_* eksik olabilir): %s", e)
    else:
        system_logger.warning("[BT] TradeExecutor içinde _close_position yok. trades whale_* yakalama pasif.")

    bt_stats = BacktestStats(
        starting_equity=equity_start_of_day,
        equity=equity_start_of_day,
        peak_equity=equity_start_of_day,
    )

    # --------------------------------------------------
    # Loop
    # --------------------------------------------------
    system_logger.info(
        "[BT] loop | warmup=%d min_len=%d long_thr=%.3f short_thr=%.3f",
        warmup, min_len, long_thr, short_thr
    )

    for i in range(warmup, min_len - 1):
        try:
            X_by_interval: Dict[str, pd.DataFrame] = {}
            mtf_whale_raw: Dict[str, pd.DataFrame] = {}

            for itv in MTF_INTERVALS:
                feat_slice_full = feat_by_interval[itv].iloc[: i + 1]
                raw_slice_full = raw_by_interval[itv].iloc[: i + 1]

                X = _prep_X(feat_slice_full).tail(500)
                if X.empty:
                    continue

                X_by_interval[itv] = X
                mtf_whale_raw[itv] = raw_slice_full.tail(500)

            if main_interval not in X_by_interval:
                continue

            # --- model probs ---
            p_arr, _debug = main_model.predict_proba(X_by_interval[main_interval])
            p_single = float(p_arr[-1])
            p_used = p_single
            mtf_debug = None

            if USE_MTF_ENS:
                try:
                    ens_p, mtf_debug = mtf_ensemble.predict_mtf(X_by_interval)
                    p_used = float(ens_p)
                except Exception as e:
                    system_logger.warning("[BT-MTF] predict_mtf hata: %s (fallback single)", e)
                    p_used = p_single
                    mtf_debug = None

            p_1m = _extract_mtf_p_last(mtf_debug, "1m")
            p_5m = _extract_mtf_p_last(mtf_debug, "5m")
            p_15m = _extract_mtf_p_last(mtf_debug, "15m")
            p_1h = _extract_mtf_p_last(mtf_debug, "1h")

            # --- base signal ---
            signal = "hold"
            if p_used >= long_thr:
                signal = "long"
            elif p_used <= short_thr:
                signal = "short"

            # --- whale meta (ÖNCE hesapla!) ---
            whale_dir = "none"
            whale_score = 0.0
            if whale_detector is not None:
                try:
                    whale_signals = whale_detector.analyze_multiple_timeframes(mtf_whale_raw)  # type: ignore[attr-defined]
                    best_score = 0.0
                    best_dir = "none"
                    for _tf, sig in (whale_signals or {}).items():
                        s_dir = str(getattr(sig, "direction", "none") or "none")
                        s_score = float(getattr(sig, "score", 0.0) or 0.0)
                        if s_dir != "none" and s_score > best_score:
                            best_score = s_score
                            best_dir = s_dir
                    whale_dir = best_dir
                    whale_score = best_score
                except Exception as e:
                    system_logger.warning("[BT-WHALE] analyze hata: %s", e)

            # --- whale policy Step-1 (tek kaynak) ---
            whale_on = (
                whale_score is not None
                and float(whale_score) >= WHALE_THR
                and str(whale_dir).lower() not in ("none", "nan", "null", "")
            )

            whale_alignment = "no_whale"
            if whale_on:
                wd = str(whale_dir).lower().strip()
                if signal in ("long", "short") and wd in ("long", "short"):
                    whale_alignment = "aligned" if signal == wd else "opposed"
                else:
                    whale_alignment = "other"

            # Whale-only
            if WHALE_ONLY and ((not whale_on) or (whale_alignment != "aligned")):
                system_logger.info(
                    "[BT-WHALE-ONLY] HOLD | bar=%d signal=%s whale_dir=%s whale_score=%.3f thr=%.2f alignment=%s",
                    i, signal, str(whale_dir), float(whale_score or 0.0), WHALE_THR, whale_alignment
                )
                signal = "hold"

            # Opsiyonel veto
            if WHALE_FILTER and WHALE_VETO_OPPOSED and whale_alignment == "opposed":
                system_logger.info(
                    "[BT-WHALE-VETO] HOLD(opposed) | bar=%d signal=%s whale_dir=%s whale_score=%.3f thr=%.2f",
                    i, signal, str(whale_dir), float(whale_score or 0.0), WHALE_THR
                )
                signal = "hold"

            # ----------------------------------------------------------
            # Model + whale confidence factor (mevcut)
            # ----------------------------------------------------------
            base_model_conf = 1.0
            whale_scale = 1.0

            if whale_on and whale_alignment == "aligned":
                whale_scale = ALIGNED_BOOST
            elif whale_on and whale_alignment == "opposed":
                whale_scale = OPPOSED_SCALE

            model_confidence_factor = float(base_model_conf) * float(whale_scale)

            # ----------------------------------------------------------
            # Ensemble_p (p_used) based sizing (ENV controlled)
            # - hard filter: p_used < ENS_MIN_P => HOLD
            # - linear scaling: p=0.5 => 1.0, p=1 => ENS_MAX_BOOST, p=0 => ENS_MIN_SCALE
            # ----------------------------------------------------------
            try:
                p_used_f = float(p_used)
            except Exception:
                p_used_f = 0.5

            p_used_f = _clamp(p_used_f, 0.0, 1.0)

            if ENS_MIN_P > 0.0 and p_used_f < ENS_MIN_P:
                signal = "hold"

            ens_scale = 1.0

            if ENS_SIZE_MODE == "off":
                ens_scale = 1.0
            elif ENS_SIZE_MODE == "linear":
                # piecewise around 0.5
                if p_used_f >= 0.5:
                    # 0.5 -> 1.0, 1.0 -> max_boost
                    t = (p_used_f - 0.5) / 0.5
                    ens_scale = 1.0 + t * (ENS_MAX_BOOST - 1.0)
                else:
                    # 0.5 -> 1.0, 0.0 -> min_scale
                    t = (0.5 - p_used_f) / 0.5
                    ens_scale = 1.0 - t * (1.0 - ENS_MIN_SCALE)

                ens_scale = _clamp(float(ens_scale), float(ENS_MIN_SCALE), float(ENS_MAX_BOOST))
            else:
                # bilinmeyen mod => off
                ens_scale = 1.0

            # ----------------------------------------------------------
            # ATR
            # ----------------------------------------------------------
            raw_main = raw_by_interval[main_interval].iloc[: i + 1]
            atr_value = compute_atr_from_klines(raw_main.tail(atr_period + 2), period=atr_period)

            # price/time
            price = float(raw_by_interval[main_interval]["close"].iloc[i])
            bar_time = _safe_bar_time_iso(raw_by_interval[main_interval], i)

            # ----------------------------------------------------------
            # Notional (execute_decision size)
            # - base_notional * model_confidence_factor * ens_scale
            # ----------------------------------------------------------
            sizing_factor = float(model_confidence_factor) * float(ens_scale)
            notional = float(base_order_notional) * float(sizing_factor)

            # HOLD ise size gönderme (TradeExecutor kendi içinde açmayacak)
            size_arg: Optional[float] = None
            if signal in ("long", "short"):
                size_arg = float(notional)

            # extra (tek dict)
            extra: Dict[str, Any] = {
                "mtf_debug": mtf_debug,
                "whale_dir": whale_dir,
                "whale_score": float(whale_score),
                "whale_on": bool(whale_on),
                "whale_alignment": whale_alignment,
                "whale_thr": float(WHALE_THR),
                "model_confidence_factor": float(model_confidence_factor),
                "ens_size_mode": str(ENS_SIZE_MODE),
                "ens_min_p": float(ENS_MIN_P),
                "ens_scale": float(ens_scale),
                "ens_notional": float(notional),
                "atr": float(atr_value) if atr_value is not None else None,
            }

            # bt_context (close patch buradan okuyacak)
            bt_context.update(
                {
                    "bar": i,
                    "time": bar_time,
                    "signal": signal,
                    "price": price,
                    "p_used": float(p_used),
                    "p_single": float(p_single),
                    "p_1m": p_1m,
                    "p_5m": p_5m,
                    "p_15m": p_15m,
                    "p_1h": p_1h,
                    "whale_dir": whale_dir,
                    "whale_score": float(whale_score),
                    "whale_on": bool(whale_on),
                    "whale_alignment": whale_alignment,
                    "whale_thr": float(WHALE_THR),
                    "model_confidence_factor": float(model_confidence_factor),
                    "ens_scale": float(ens_scale),
                    "ens_notional": float(notional),
                    "atr": float(atr_value) if atr_value is not None else None,
                }
            )

            # pnl before
            try:
                prev_pnl = float(getattr(risk_manager, "daily_realized_pnl", 0.0) or 0.0)
            except Exception:
                prev_pnl = 0.0

            await trade_executor.execute_decision(
                signal=signal,
                symbol=symbol,
                price=price,
                size=size_arg,  # <<<<<<<<<<<<<< sizing burada
                interval=main_interval,
                training_mode=False,
                hybrid_mode=HYBRID_MODE,
                probs={"p_used": float(p_used), "p_single": float(p_single)},
                extra=extra,
            )

            # pnl after
            try:
                new_pnl = float(getattr(risk_manager, "daily_realized_pnl", prev_pnl) or prev_pnl)
            except Exception:
                new_pnl = prev_pnl

            delta = new_pnl - prev_pnl
            if delta != 0.0:
                bt_stats.on_pnl_delta(delta)

            equity_rows.append(
                {
                    "bar": i,
                    "time": bar_time,
                    "symbol": symbol,
                    "interval": main_interval,
                    "price": price,
                    "signal": signal,
                    "p_used": float(p_used),
                    "p_single": float(p_single),
                    "p_1m": p_1m,
                    "p_5m": p_5m,
                    "p_15m": p_15m,
                    "p_1h": p_1h,
                    "whale_dir": whale_dir,
                    "whale_score": float(whale_score),
                    "whale_on": bool(whale_on),
                    "whale_alignment": whale_alignment,
                    "whale_thr": float(WHALE_THR),
                    "model_confidence_factor": float(model_confidence_factor),
                    "ens_scale": float(ens_scale),
                    "ens_notional": float(notional),
                    "atr": float(atr_value) if atr_value is not None else None,
                    "equity": float(bt_stats.equity),
                    "peak_equity": float(bt_stats.peak_equity),
                    "max_drawdown_pct": float(bt_stats.max_drawdown) * 100.0,
                    "pnl_total": float(bt_stats.equity - bt_stats.starting_equity),
                }
            )

        except Exception as e:
            system_logger.exception("[BT-LOOP-ERROR] bar=%d hata=%s", i, e)

    # --------------------------------------------------
    # Export CSV
    # --------------------------------------------------
    out_dir = Path(os.getenv("BT_OUT_DIR", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    equity_path = out_dir / f"equity_curve_{symbol}_{main_interval}_{tag}.csv"
    trades_path = out_dir / f"trades_{symbol}_{main_interval}_{tag}.csv"
    summary_path = out_dir / f"summary_{symbol}_{main_interval}_{tag}.csv"

    EQ_COLS = [
        "bar", "time", "symbol", "interval", "price", "signal",
        "p_used", "p_single", "p_1m", "p_5m", "p_15m", "p_1h",
        "whale_dir", "whale_score", "whale_on", "whale_alignment", "whale_thr",
        "model_confidence_factor",
        "ens_scale", "ens_notional",
        "atr",
        "equity", "peak_equity", "max_drawdown_pct", "pnl_total"
    ]

    TR_COLS = [
        "symbol", "side", "qty", "entry_price", "notional", "interval", "opened_at",
        "sl_price", "tp_price", "trailing_pct", "atr_value", "highest_price", "lowest_price",
        "meta", "closed_at", "close_price", "realized_pnl", "close_reason",
        "bt_symbol", "bt_interval", "bt_bar", "bt_time", "bt_signal", "bt_price",
        "bt_p_used", "bt_p_single", "bt_p_1m", "bt_p_5m", "bt_p_15m", "bt_p_1h",
        "whale_dir", "whale_score", "whale_on", "whale_alignment", "whale_thr",
        "model_confidence_factor",
        "ens_scale", "ens_notional",
        "bt_atr"
    ]

    eq_df = _ensure_columns(pd.DataFrame(equity_rows), EQ_COLS)
    tr_df = _ensure_columns(pd.DataFrame(closed_trades), TR_COLS)
    sm_df = pd.DataFrame([bt_stats.summary_dict()])

    _atomic_to_csv(eq_df, equity_path)
    _atomic_to_csv(tr_df, trades_path)
    _atomic_to_csv(sm_df, summary_path)

    system_logger.info("[BT-CSV] equity_curve: %s (rows=%d)", str(equity_path), len(eq_df))
    system_logger.info("[BT-CSV] trades:      %s (trades=%d)", str(trades_path), len(tr_df))
    system_logger.info("[BT-CSV] summary:     %s", str(summary_path))


# ==========================================================
# Entry
# ==========================================================
async def async_main() -> None:
    global system_logger
    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")
    await run_backtest()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
