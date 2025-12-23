import asyncio
import logging
import os
import signal
from typing import Any, Dict, Optional

import pandas as pd

from config.load_env import load_environment_variables
from config.settings import Settings

from core.logger import setup_logger
from core.binance_client import create_binance_client
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor

from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner
from data.whale_detector import MultiTimeframeWhaleDetector

from core.hybrid_mtf import MultiTimeframeHybridEnsemble
from models.hybrid_inference import HybridModel

from tg_bot.telegram_bot import TelegramBot

# WebSocket
from websocket.binance_ws import BinanceWS  # Biz bunu aşağıda websocket/binance_ws.py içinde tanımlayacağız

# NEW: p_buy stabilization + safe proba
from core.prob_stabilizer import ProbStabilizer
from core.model_utils import safe_p_buy


# ----------------------------------------------------------------------
# Global config / flags
# ----------------------------------------------------------------------
USE_TESTNET = getattr(Settings, "USE_TESTNET", True)
SYMBOL = getattr(Settings, "SYMBOL", "BTCUSDT")

system_logger: Optional[logging.Logger] = None

LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]

current_ws = None

def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


# ---- helpers: EMA adaptive alpha ----
def _compute_adaptive_alpha(atr_value: float) -> float:
    """
    Volatilite artınca alpha yükselsin (EMA daha hızlı adapte olsun),
    volatilite düşükken alpha düşsün (EMA daha stabil olsun).
    """
    a_min = float(os.getenv("EMA_ALPHA_MIN", "0.05"))
    a_max = float(os.getenv("EMA_ALPHA_MAX", "0.45"))
    atr_ref = float(os.getenv("EMA_ATR_REF", "100.0"))

    try:
        atr = float(atr_value)
    except Exception:
        atr = 0.0

    if atr_ref <= 1e-9:
        atr_ref = 100.0

    x = atr / atr_ref
    x = max(0.0, min(2.0, x))

    alpha = a_min + (a_max - a_min) * (x / 2.0)
    alpha = max(a_min, min(a_max, alpha))
    return float(alpha)


# Global env flag’ler (async_main içinde tekrar güncellenecek)
BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

HYBRID_MODE: bool = get_bool_env("HYBRID_MODE", True)
TRAINING_MODE: bool = get_bool_env("TRAINING_MODE", False)
USE_MTF_ENS: bool = get_bool_env("USE_MTF_ENS", False)
DRY_RUN: bool = get_bool_env("DRY_RUN", True)


# ----------------------------------------------------------------------
# Basit feature engineering
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # 1) Zaman kolonlarını normalize et
    for col in ["open_time", "close_time"]:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            dt = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            dt = pd.to_datetime(df[col], utc=True)
        df[col] = dt.astype("int64") / 1e9  # saniye

    # 2) Numeric cast
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    int_cols = ["number_of_trades"]

    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    if "ignore" not in df.columns:
        df["ignore"] = 0.0

    # 3) Teknik feature'lar
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["volume"].rolling(10).std()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    # 4) NaN temizliği
    df = df.ffill().bfill().fillna(0.0)
    return df


def build_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    return (future > close).astype(int)


# ----------------------------------------------------------------------
# ATR hesaplayıcı
# ----------------------------------------------------------------------
def compute_atr_from_klines(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


# ----------------------------------------------------------------------
# Binance Kline fetch helper
# ----------------------------------------------------------------------
async def fetch_klines(client, symbol: str, interval: str, limit: int, logger: Optional[logging.Logger]) -> pd.DataFrame:
    if client is None:
        csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
        if not os.path.exists(csv_path):
            if logger:
                logger.error("[DATA] client=None ve offline CSV bulunamadı: %s", csv_path)
            raise RuntimeError(
                f"Offline kline dosyası yok: {csv_path}. "
                "Lütfen BINANCE_API_KEY/BINANCE_API_SECRET set edin veya offline cache oluşturun."
            )

        df = pd.read_csv(csv_path)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        if logger:
            logger.info("[DATA] OFFLINE mod: %s dosyasından kline yüklendi. shape=%s", csv_path, df.shape)
        return df

    try:
        klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        if logger:
            logger.error("[DATA] Binance get_klines hatası: %s", e)
        raise

    columns = [
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
    ]
    df = pd.DataFrame(klines, columns=columns)

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades"]

    for c in float_cols:
        df[c] = df[c].astype(float)
    for c in int_cols:
        df[c] = df[c].astype(int)

    if logger:
        logger.info("[DATA] ONLINE mod: Binance'ten kline çekildi. symbol=%s interval=%s shape=%s", symbol, interval, df.shape)
    return df


# ----------------------------------------------------------------------
# Trading objeleri kurulum
# ----------------------------------------------------------------------
def create_trading_objects() -> Dict[str, Any]:
    global system_logger
    global BINANCE_API_KEY, BINANCE_API_SECRET
    global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS, DRY_RUN, USE_TESTNET

# --- SYMBOL & INTERVAL (env öncelikli, cfg opsiyonel) ---
    cfg = globals().get("cfg", None)

    symbol = (
        os.getenv("SYMBOL")
        or getattr(Settings, "SYMBOL", None)
        or "BTCUSDT"
    )
    interval = (
        os.getenv("INTERVAL")
        or getattr(Settings, "INTERVAL", None)
        or "5m"
    )
# --- /SYMBOL & INTERVAL ---

    client = create_binance_client(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=bool(USE_TESTNET),
        logger=system_logger,
        dry_run=bool(DRY_RUN),
    )

    # --------------------------
    # Risk Manager
    # --------------------------
    daily_max_loss_usdt = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))
    daily_max_loss_pct = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.03"))
    max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
    max_open_trades = int(os.getenv("MAX_OPEN_TRADES", "3"))
    equity_start_of_day = float(os.getenv("EQUITY_START_OF_DAY", "1000"))

    risk_manager = RiskManager(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_loss_pct=daily_max_loss_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_open_trades=max_open_trades,
        equity_start_of_day=equity_start_of_day,
        logger=system_logger,
    )

    # --------------------------
    # Telegram (opsiyonel)
    # --------------------------
    tg_bot = None
    try:
        tg_bot = TelegramBot()

        # python-telegram-bot token yoksa TelegramBot içi bot/dispatcher None kalabiliyor
        if getattr(tg_bot, "dispatcher", None):
            if hasattr(tg_bot, "set_risk_manager"):
                tg_bot.set_risk_manager(risk_manager)
            else:
                # çok eski sürümler için fallback
                setattr(tg_bot, "risk_manager", risk_manager)

            if system_logger:
                system_logger.info(
                    "[MAIN] TelegramBot'a RiskManager enjekte edildi (/risk komutu aktif)."
                )
        else:
            if system_logger:
                system_logger.warning(
                    "[MAIN] Telegram dispatcher yok (muhtemelen TELEGRAM_BOT_TOKEN tanımsız)."
                )
    except Exception as e:
        tg_bot = None
        if system_logger:
            system_logger.warning("[MAIN] TelegramBot init/enjeksiyon hata: %s", e)

    # --------------------------
    # Position Manager (Redis/PG)
    # --------------------------
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", "bot:positions")

    enable_pg = os.getenv("ENABLE_PG_POS_LOG", "0")
    enable_pg_flag = enable_pg not in ("0", "false", "False", "FALSE", "")
    pg_dsn = os.getenv("PG_DSN") if enable_pg_flag else None

    position_manager = PositionManager(
        redis_url=redis_url,
        redis_key_prefix=redis_key_prefix,
        logger=system_logger,
        enable_pg=enable_pg_flag,
        pg_dsn=pg_dsn,
    )

    # --------------------------
    # Hybrid Model (single TF)
    # --------------------------
    hybrid_model = HybridModel(model_dir="models", interval=interval, logger=system_logger)
    try:
        # HYBRID_MODE, LSTM kullanımını kontrol ediyorsa
        if hasattr(hybrid_model, "use_lstm_hybrid"):
            hybrid_model.use_lstm_hybrid = bool(HYBRID_MODE)
    except Exception:
        pass

    # --------------------------
    # MTF Ensemble (opsiyonel)
    # --------------------------
    mtf_ensemble = None
    if USE_MTF_ENS:
        try:
            # MTF_INTERVALS main.py içinde tanımlı değilse fallback:
            try:
                mtf_intervals = list(MTF_INTERVALS)  # type: ignore[name-defined]
            except Exception:
                mtf_intervals = ["1m", "5m", "15m", "1h"]

            mtf_models: Dict[str, HybridModel] = {}
            for itv in mtf_intervals:
                try:
                    hm = HybridModel(model_dir="models", interval=itv, logger=system_logger)
                    if hasattr(hm, "use_lstm_hybrid"):
                        hm.use_lstm_hybrid = bool(HYBRID_MODE)
                    mtf_models[itv] = hm
                    if system_logger:
                        system_logger.info("[HYBRID-MTF] HybridModel yüklendi | interval=%s", itv)
                except Exception as e:
                    if system_logger:
                        system_logger.warning(
                            "[HYBRID-MTF] %s interval'i için HybridModel yüklenemedi: %s",
                            itv,
                            e,
                        )

            if mtf_models:
                mtf_ensemble = MultiTimeframeHybridEnsemble(models_by_interval=mtf_models)
                if system_logger:
                    system_logger.info(
                        "[MAIN] Multi-timeframe hybrid ensemble aktif: intervals=%s",
                        list(mtf_models.keys()),
                    )
        except Exception as e:
            mtf_ensemble = None
            if system_logger:
                system_logger.warning(
                    "[MAIN] MultiTimeframeHybridEnsemble init hata, MTF ensemble devre dışı: %s",
                    e,
                )

    # --------------------------
    # Whale Detector (opsiyonel)
    # --------------------------
    whale_detector = None
    try:
        whale_detector = MultiTimeframeWhaleDetector()
        if system_logger:
            system_logger.info("[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi.")
    except Exception as e:
        whale_detector = None
        if system_logger:
            system_logger.warning("[WHALE] MultiTimeframeWhaleDetector init hata: %s", e)

    # --------------------------
    # Trade Executor
    # --------------------------
    base_order_notional = float(os.getenv("BASE_ORDER_NOTIONAL", "50"))
    max_position_notional = float(os.getenv("MAX_POSITION_NOTIONAL", "500"))
    max_leverage = float(os.getenv("MAX_LEVERAGE", "3"))

    sl_pct = float(os.getenv("SL_PCT", "0.01"))
    tp_pct = float(os.getenv("TP_PCT", "0.02"))
    trailing_pct = float(os.getenv("TRAILING_PCT", "0.01"))

    use_atr_sltp = os.getenv("USE_ATR_SLTP", "true").lower() == "true"
    atr_sl_mult = float(os.getenv("ATR_SL_MULT", "1.5"))
    atr_tp_mult = float(os.getenv("ATR_TP_MULT", "3.0"))

    whale_risk_boost = float(os.getenv("WHALE_RISK_BOOST", "2.0"))

    trade_executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        position_manager=position_manager,
        logger=system_logger,
        dry_run=bool(DRY_RUN),
        base_order_notional=base_order_notional,
        max_position_notional=max_position_notional,
        max_leverage=max_leverage,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trailing_pct=trailing_pct,
        use_atr_sltp=use_atr_sltp,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        whale_risk_boost=whale_risk_boost,
    )

    # --------------------------
    # Online model
    # --------------------------
    online_model = OnlineLearner(model_dir="models", base_model_name="online_model", interval=interval)

    return {
        "symbol": symbol,
        "interval": interval,
        "client": client,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "hybrid_model": hybrid_model,
        "mtf_ensemble": mtf_ensemble,
        "whale_detector": whale_detector,
        "tg_bot": tg_bot,
        "online_model": online_model,
    }


def _normalize_signal(sig: Any) -> str:
    """
    ProbStabilizer.signal çıktısını TradeExecutor formatına çevir.
    Kabul edilen: BUY/SELL/HOLD veya long/short/hold
    """
    s = str(sig).strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return "hold"


# ----------------------------------------------------------------------
# Ana trading loop
# ----------------------------------------------------------------------
async def bot_loop(objs: Dict[str, Any], prob_stab: ProbStabilizer) -> None:
    global system_logger

    client = objs["client"]
    trade_executor = objs["trade_executor"]
    whale_detector = objs.get("whale_detector")
    hybrid_model = objs["hybrid_model"]
    mtf_ensemble = objs.get("mtf_ensemble")
    online_model = objs["online_model"]

    symbol = objs.get("symbol", os.getenv("SYMBOL", getattr(Settings, "SYMBOL", "BTCUSDT")))
    interval = objs.get("interval", os.getenv("INTERVAL", "5m"))
    data_limit = int(os.getenv("DATA_LIMIT", "500"))

    use_ema_signal_default = get_bool_env("USE_PBUY_STABILIZER_SIGNAL", True)

    if system_logger:
        system_logger.info(
            "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s, USE_MTF_ENS=%s, USE_PBUY_STABILIZER_SIGNAL=%s)",
            symbol,
            interval,
            TRAINING_MODE,
            HYBRID_MODE,
            USE_MTF_ENS,
            use_ema_signal_default,
        )

    anomaly_detector = AnomalyDetector(logger=system_logger)

    alias_map = {
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }

    feature_cols = [
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

    while True:
        try:
            raw_df = await fetch_klines(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=data_limit,
                logger=system_logger,
            )

            feat_df = build_features(raw_df)

            for old_col, new_col in alias_map.items():
                if old_col not in feat_df.columns and new_col in feat_df.columns:
                    feat_df[old_col] = feat_df[new_col]

            feat_df = anomaly_detector.filter_anomalies(feat_df)

            feature_cols_existing = [c for c in feature_cols if c in feat_df.columns]
            missing = [c for c in feature_cols if c not in feat_df.columns]
            if missing and system_logger:
                system_logger.warning("[FE] Eksik feature kolonları tespit edildi: %s", missing)

            X_live = feat_df[feature_cols_existing].tail(500)

            # 1) Single-TF hibrit skor
            p_arr_single, debug_single = hybrid_model.predict_proba(X_live)
            p_single = float(p_arr_single[-1])

            meta = getattr(hybrid_model, "meta", {}) or {}
            model_conf_factor = float(meta.get("confidence_factor", 1.0) or 1.0)
            best_auc = float(meta.get("best_auc", 0.5) or 0.5)
            best_side = meta.get("best_side", "long")

            # 2) MTF verileri
            mtf_feats: Dict[str, pd.DataFrame] = {interval: feat_df}
            mtf_whale_raw: Dict[str, pd.DataFrame] = {interval: raw_df}

            if USE_MTF_ENS and mtf_ensemble is not None:
                for itv in ["1m", "15m", "1h"]:
                    try:
                        raw_df_itv = await fetch_klines(
                            client=client,
                            symbol=symbol,
                            interval=itv,
                            limit=data_limit,
                            logger=system_logger,
                        )
                        feat_df_itv = build_features(raw_df_itv)

                        for old_col, new_col in alias_map.items():
                            if old_col not in feat_df_itv.columns and new_col in feat_df_itv.columns:
                                feat_df_itv[old_col] = feat_df_itv[new_col]

                        mtf_feats[itv] = feat_df_itv
                        mtf_whale_raw[itv] = raw_df_itv
                    except Exception as e:
                        if system_logger:
                            system_logger.warning("[MTF] %s interval'i hazırlanırken hata: %s", itv, e)

            # 3) MTF ensemble skoru
            p_used = p_single
            mtf_debug: Optional[Dict[str, Any]] = None
            p_1m = p_5m = p_15m = p_1h = None

            if USE_MTF_ENS and mtf_ensemble is not None:
                try:
                    X_by_interval: Dict[str, pd.DataFrame] = {}
                    for itv, df_itv in mtf_feats.items():
                        cols_itv = [c for c in feature_cols if c in df_itv.columns]
                        if not cols_itv:
                            if system_logger:
                                system_logger.warning("[MTF] Interval=%s için kullanılabilir feature yok, skip ediliyor.", itv)
                            continue
                        X_by_interval[itv] = df_itv[cols_itv].tail(500)

                    if X_by_interval:
                        p_ens, mtf_debug = mtf_ensemble.predict_mtf(X_by_interval)
                        p_used = float(p_ens)

                        per_int = mtf_debug.get("per_interval", {}) if isinstance(mtf_debug, dict) else {}
                        p_1m = per_int.get("1m", {}).get("p_last")
                        p_5m = per_int.get("5m", {}).get("p_last")
                        p_15m = per_int.get("15m", {}).get("p_last")
                        p_1h = per_int.get("1h", {}).get("p_last")

                except Exception as e:
                    if system_logger:
                        system_logger.warning("[MTF] Ensemble hesaplanırken hata: %s", e)
                    p_used = p_single
                    mtf_debug = None
                    p_1m = p_5m = p_15m = p_1h = None

            # 4) Whale meta
            whale_meta: Dict[str, Any] = {"direction": "none", "score": 0.0, "per_tf": {}}

            if whale_detector is not None:
                try:
                    if hasattr(whale_detector, "analyze_multiple_timeframes"):
                        whale_signals = whale_detector.analyze_multiple_timeframes(mtf_whale_raw)
                        best_tf = None
                        best_score = 0.0

                        for tf, sig in whale_signals.items():
                            whale_meta["per_tf"][tf] = {
                                "direction": sig.direction,
                                "score": sig.score,
                                "reason": sig.reason,
                            }
                            if sig.direction != "none" and sig.score > best_score:
                                best_score = sig.score
                                best_tf = tf

                        if best_tf is not None:
                            best_sig = whale_signals[best_tf]
                            whale_meta.update(
                                {
                                    "direction": best_sig.direction,
                                    "score": best_sig.score,
                                    "best_tf": best_tf,
                                    "best_reason": best_sig.reason,
                                }
                            )
                    elif hasattr(whale_detector, "from_klines"):
                        ws = whale_detector.from_klines(raw_df)
                        whale_meta.update(
                            {
                                "direction": ws.direction,
                                "score": ws.score,
                                "reason": ws.reason,
                                "meta": ws.meta,
                            }
                        )
                except Exception as e:
                    if system_logger:
                        system_logger.warning("[WHALE] MTF whale hesaplanırken hata: %s", e)

            # 5) ATR
            atr_period = int(os.getenv("ATR_PERIOD", "14"))
            atr_value = compute_atr_from_klines(raw_df, period=atr_period)

            # 6) probs + extra
            probs: Dict[str, Any] = {
                "p_used": p_used,
                "p_single": p_single,
                "p_sgd_mean": float(debug_single.get("p_sgd_mean", 0.0)),
                "p_lstm_mean": float(debug_single.get("p_lstm_mean", 0.5)),
            }

            extra: Dict[str, Any] = {
                "model_confidence_factor": model_conf_factor,
                "best_auc": best_auc,
                "best_side": best_side,
                "mtf_debug": mtf_debug,
                "whale_meta": whale_meta,
                "atr": atr_value,
            }

            # ------------------------------------------------------------------
            # Online model p_buy stabilizasyonu (EMA) + zscore clip + adaptive alpha
            # ------------------------------------------------------------------
            use_ema_signal = get_bool_env("USE_PBUY_STABILIZER_SIGNAL", False)
            ema_whale_only = get_bool_env("EMA_WHALE_ONLY", False)
            ema_whale_thr = float(os.getenv("EMA_WHALE_THR", "0.50"))

            prob_stab.use_zclip = get_bool_env("PBUY_ZCLIP_ON", True)
            prob_stab.zclip = float(os.getenv("PBUY_ZCLIP", "3.0"))
            prob_stab.zwin = int(os.getenv("PBUY_ZWIN", "60"))

            signal_side: Optional[str] = None

            try:
                alpha_dyn = _compute_adaptive_alpha(atr_value)
                if hasattr(prob_stab, "set_alpha"):
                    prob_stab.set_alpha(alpha_dyn)
                else:
                    prob_stab.alpha = float(alpha_dyn)

                X_last = X_live.tail(1)

                whale_score = 0.0
                whale_dir = "none"
                if isinstance(whale_meta, dict):
                    whale_score = float(whale_meta.get("score", 0.0) or 0.0)
                    whale_dir = str(whale_meta.get("direction", "none") or "none")
                whale_on = (whale_dir != "none") and (whale_score >= ema_whale_thr)

                p_buy_raw = safe_p_buy(online_model, X_last)
                p_buy_ema = prob_stab.update(p_buy_raw)
                sig_from_ema = _normalize_signal(prob_stab.signal(p_buy_ema))

                probs["p_buy_raw"] = float(p_buy_raw)
                probs["p_buy_ema"] = float(p_buy_ema)

                extra["p_buy_raw"] = float(p_buy_raw)
                extra["p_buy_ema"] = float(p_buy_ema)
                extra["ema_alpha"] = float(getattr(prob_stab, "alpha", alpha_dyn))
                extra["ema_whale_only"] = bool(ema_whale_only)
                extra["ema_whale_thr"] = float(ema_whale_thr)
                extra["ema_whale_on"] = bool(whale_on)

                if use_ema_signal and (not ema_whale_only or whale_on):
                    signal_side = sig_from_ema

            except Exception as e:
                if system_logger:
                    system_logger.warning("[ONLINE] EMA block hata: %s", e)
                signal_side = None

            # EMA kullanılmadıysa hybrid threshold sinyaline düş
            if signal_side is None:
                long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
                short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))

                if p_used >= long_thr:
                    signal_side = "long"
                elif p_used <= short_thr:
                    signal_side = "short"
                else:
                    signal_side = "hold"

                extra["signal_source"] = "HYBRID"
            else:
                extra["signal_source"] = "EMA"

            # ------------------------------
            # Trend filtresi + mikro filtre
            # ------------------------------
            micro_conf_scale = 1.0
            if signal_side == "long" and isinstance(p_1m, float) and p_1m < 0.30:
                micro_conf_scale = 0.7
            elif signal_side == "short" and isinstance(p_1m, float) and p_1m > 0.70:
                micro_conf_scale = 0.7

            effective_model_conf = float(model_conf_factor) * micro_conf_scale

            # === SATURATION SNAPSHOT (auto) ===
            prob_dbg = getattr(prob_stab, "_last_dbg", None)
            # rate-limit: 60sn (loop sık ise log şişmesin)
            try:
                from datetime import datetime
                _now = datetime.utcnow().timestamp()
                _last = globals().get('_SATDBG_LAST_TS', 0) or 0
                if (_now - float(_last)) > 60:
                    globals()['_SATDBG_LAST_TS'] = _now
                    if system_logger:
                        system_logger.info(
                            "[SATDBG] p_used=%.6f p_src=%s raw=%r ema=%r stable=%r mcf=%.6f micro=%.3f eff=%.6f signal_side=%s prob_dbg=%s",
                            float(p_used),
                            (extra.get('p_buy_source','unknown') if isinstance(extra, dict) else 'noextra'),
                            (extra.get('p_buy_raw') if isinstance(extra, dict) else None),
                            (extra.get('p_buy_ema') if isinstance(extra, dict) else None),
                            (extra.get('p_buy_stable') if isinstance(extra, dict) else None),
                            float(model_conf_factor),
                            float(micro_conf_scale),
                            float(effective_model_conf),
                            str(signal_side),
                            getattr(globals().get("prob_stab", None), "_dbg_last", None),
                        )
            except Exception:
                pass
            # === /SATURATION SNAPSHOT ===

            # ==============================
            # p_used DEBUG SNAPSHOT (rate-limited)
            # ==============================
            try:
                from datetime import datetime
                _now = datetime.utcnow().timestamp()
                _last = globals().get("_P_USED_DBG_LAST_TS", 0) or 0
                _ok = (_now - float(_last)) > 60
                if _ok:
                    globals()["_P_USED_DBG_LAST_TS"] = _now
                    if system_logger:
                        system_logger.info(
                            "[P_USED_DBG] p_used=%.6f src=%s p_buy_raw=%r p_buy_ema=%r stable=%r thr_buy=%r thr_sell=%r",
                            float(p_used),
                            (extra.get("p_buy_source", "unknown") if isinstance(extra, dict) else "unknown"),
                            (extra.get("p_buy_raw", None) if isinstance(extra, dict) else None),
                            (extra.get("p_buy_ema", None) if isinstance(extra, dict) else None),
                            (extra.get("p_buy_stable", None) if isinstance(extra, dict) else None),
                            os.getenv("LONG_THRESHOLD"),
                            os.getenv("SHORT_THRESHOLD"),
                        )
            except Exception:
                pass

            # ==============================
            # ATR bazlı dinamik MCF_GAMMA
            # ==============================
            gamma = float(os.getenv("MCF_GAMMA", "0.6"))
            try:
                atr = None
                if isinstance(extra, dict):
                    atr = extra.get("atr")
                atr = float(atr) if atr is not None else None

                atr_lo = float(os.getenv("MCF_ATR_LO", "50"))
                atr_hi = float(os.getenv("MCF_ATR_HI", "300"))
                g_min  = float(os.getenv("MCF_GAMMA_MIN", "0.45"))
                g_max  = float(os.getenv("MCF_GAMMA_MAX", "0.85"))

                if atr is not None and atr_hi > atr_lo:
                    t = (atr - atr_lo) / (atr_hi - atr_lo)
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                    gamma = g_max - t * (g_max - g_min)  # atr ↑ -> gamma ↓
            except Exception:
                pass

            # debug için extra'ya yaz
            try:
                if isinstance(extra, dict):
                    extra["mcf_gamma"] = gamma
            except Exception:
                pass

            # --- model_confidence_factor (normalized + compression + HOLD-safe) ---
            mcf = None
            try:
                mcf = float(effective_model_conf)
            except Exception:
                mcf = None

            if mcf is not None:
                # 0..100 gelme ihtimaline karşı normalize
                if mcf > 1.0 and mcf <= 100.0:
                    mcf = mcf / 100.0

                # 1) clamp 0..1
                if mcf < 0.0:
                    mcf = 0.0
                elif mcf > 1.0:
                    mcf = 1.0

                # 2) compression (aşırı confidence'ı bastır) -> ATR'den gelen gamma kullanılır
                mcf = 0.5 + (mcf - 0.5) * float(gamma)

                # 3) tekrar clamp
                if mcf < 0.0:
                    mcf = 0.0
                elif mcf > 1.0:
                    mcf = 1.0

            # HOLD guard: signal_side 'hold' ise yazma, varsa da temizle
            if str(signal_side).lower() != "hold" and mcf is not None:
                if isinstance(extra, dict):
                    extra["model_confidence_factor"] = mcf
                if system_logger:
                    try:
                        system_logger.info(
                            "[MCF_DBG] eff=%r micro=%r mcf_final=%r gamma=%r sig=%s atr=%r",
                            float(effective_model_conf),
                            float(micro_conf_scale),
                            mcf,
                            float(gamma),
                            str(signal_side),
                            (extra.get("atr") if isinstance(extra, dict) else None),
                        )
                    except Exception:
                        pass
            else:
                try:
                    if isinstance(extra, dict):
                        extra.pop("model_confidence_factor", None)
                except Exception:
                    pass

            # --- /model_confidence_factor ---

            if system_logger:
                whale_dir_dbg = whale_meta.get("direction") if isinstance(whale_meta, dict) else None
                whale_score_dbg = whale_meta.get("score") if isinstance(whale_meta, dict) else None
                system_logger.info(
                    "[SIGNAL] source=%s p_used=%.4f signal=%s model_conf=%.3f eff_conf=%.3f p_1m=%s p_5m=%s p_15m=%s p_1h=%s whale_dir=%s whale_score=%s",
                    extra.get("signal_source", "unknown"),
                    float(p_used),
                    signal_side,
                    float(model_conf_factor),
                    float(effective_model_conf),
                    f"{p_1m:.4f}" if isinstance(p_1m, float) else "None",
                    f"{p_5m:.4f}" if isinstance(p_5m, float) else "None",
                    f"{p_15m:.4f}" if isinstance(p_15m, float) else "None",
                    f"{p_1h:.4f}" if isinstance(p_1h, float) else "None",
                    whale_dir_dbg if whale_dir_dbg is not None else "None",
                    f"{whale_score_dbg:.3f}" if isinstance(whale_score_dbg, (int, float)) else "None",
                )

            last_price = float(raw_df["close"].iloc[-1])

            await trade_executor.execute_decision(
                signal=signal_side,
                symbol=symbol,
                price=last_price,
                size=None,
                interval=interval,
                training_mode=TRAINING_MODE,
                hybrid_mode=HYBRID_MODE,
                probs=probs,
                extra=extra,
            )

        except Exception as e:
            if system_logger:
                system_logger.exception("[LOOP ERROR] %s", e)
            else:
                print("[LOOP ERROR]", e)

        await asyncio.sleep(LOOP_SLEEP_SECONDS)


# ----------------------------------------------------------------------
# Async main
# ----------------------------------------------------------------------
async def async_main() -> None:
    global system_logger
    global BINANCE_API_KEY, BINANCE_API_SECRET
    global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS, DRY_RUN

    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

    HYBRID_MODE = get_bool_env("HYBRID_MODE", HYBRID_MODE)
    TRAINING_MODE = get_bool_env("TRAINING_MODE", TRAINING_MODE)
    USE_MTF_ENS = get_bool_env("USE_MTF_ENS", USE_MTF_ENS)
    DRY_RUN = get_bool_env("DRY_RUN", DRY_RUN)

    if system_logger:
        system_logger.info("[MAIN] TRAINING_MODE=%s", TRAINING_MODE)
        system_logger.info("[MAIN] HYBRID_MODE=%s", HYBRID_MODE)
        system_logger.info("[MAIN] USE_MTF_ENS=%s", USE_MTF_ENS)
        system_logger.info("[MAIN] DRY_RUN=%s", DRY_RUN)

        if not (BINANCE_API_KEY and BINANCE_API_SECRET):
            system_logger.warning(
                "[BINANCE] API key/secret env'de yok. DRY_RUN modunda olduğundan emin ol."
            )

    prob_stab = ProbStabilizer(
        alpha=float(os.getenv("PBUY_EMA_ALPHA", "0.20")),
        buy_thr=float(os.getenv("BUY_THRESHOLD", "0.60")),
        sell_thr=float(os.getenv("SELL_THRESHOLD", "0.40")),
    )
    if system_logger:
        system_logger.info(
            "[MAIN] ProbStabilizer init | alpha=%.3f buy_thr=%.2f sell_thr=%.2f",
            float(getattr(prob_stab, "alpha", 0.20)),
            float(getattr(prob_stab, "buy_thr", 0.60)),
            float(getattr(prob_stab, "sell_thr", 0.40)),
        )
    # -----------------------------
    # WebSocket enable/disable
    # -----------------------------
    enable_ws = get_bool_env("ENABLE_WS", False)
    ws = None

    if enable_ws:
        symbol = os.getenv("SYMBOL", getattr(Settings, "SYMBOL", "BTCUSDT"))
        ws = BinanceWS(symbol=symbol)
        ws.run_background()

        if system_logger:
            system_logger.info(
                "[WS] ENABLE_WS=true -> websocket started. symbol=%s",
                symbol,
            )
    else:
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=false -> websocket disabled.")

    # -----------------------------
    # Trading objects
    # -----------------------------
    trading_objects = create_trading_objects()

    # ------------------------------------------------------------
    # WebSocket enable (Binance trade stream)
    # ------------------------------------------------------------
    enable_ws = get_bool_env("ENABLE_WS", False)

    global current_ws
    if enable_ws:
        try:
            symbol = getattr(Settings, "SYMBOL", "BTCUSDT")
        except Exception:
            symbol = "BTCUSDT"

        # İlk bağlantı
        current_ws = start_ws_in_thread(symbol=symbol)

        # Reconnect loop (ayrı thread)
        def get_ws():
            global current_ws
            return current_ws

        def set_ws(new_ws):
            global current_ws
            current_ws = new_ws

        threading.Thread(
            target=reconnect_ws_loop,
            args=(get_ws, set_ws),
            kwargs={"retry_interval": int(os.getenv("WS_RETRY_INTERVAL", "5"))},
            daemon=True,
        ).start()

        if system_logger:
            system_logger.info("[WS] ENABLE_WS=true -> websocket thread + reconnect loop started.")
    else:
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=false -> websocket disabled.")

    # -----------------------------
    # WebSocket (optional) + reconnect loop
    # -----------------------------
    enable_ws = get_bool_env("ENABLE_WS", False)

    # nonlocal hatasına girmemek için holder dict kullanıyoruz
    ws_holder = {"ws": None}

    if enable_ws:
        # symbol'ü trading_objects içinden yakalamaya çalış
        symbol = None
        try:
            symbol = trading_objects.get("symbol")  # type: ignore[attr-defined]
        except Exception:
            symbol = None

        ws_holder["ws"] = start_ws_in_thread(symbol=symbol)

        def _get_ws():
            return ws_holder["ws"]

        def _set_ws(new_ws):
            ws_holder["ws"] = new_ws

        threading.Thread(
            target=reconnect_ws_loop,
            args=(_get_ws, _set_ws),
            daemon=True,
        ).start()

        if system_logger:
            system_logger.info("[WebSocket] ENABLE_WS=true → WS started + reconnect loop running.")

    # -----------------------------
    # Main bot loop
    # -----------------------------
    try:
        await bot_loop(trading_objects, prob_stab)
    finally:
        # WS graceful close (optional)
        try:
            ws = ws_holder.get("ws")
            if ws is not None:
                ws.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Sync main + signal handling
# ----------------------------------------------------------------------
def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(async_main())
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()



# -----------------------------
# WebSocket (optional)
# -----------------------------
enable_ws = get_bool_env("ENABLE_WS", False)
ws = None

if enable_ws:
    # İlk bağlantı
    ws = start_ws_in_thread(symbol=symbol)

    # Reconnect loop: ws kapanırsa yeniden bağlan
    current_ws = ws

    def _get_ws():
        return current_ws

    def _set_ws(new_ws):
        current_ws = new_ws

    threading.Thread(
        target=reconnect_ws_loop,
        args=(_get_ws, _set_ws),
        daemon=True,
    ).start()

    if system_logger:
        system_logger.info("[WebSocket] ENABLE_WS=true → WS thread + reconnect loop started.")

if __name__ == "__main__":
    main()
