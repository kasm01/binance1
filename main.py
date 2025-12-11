import os
import asyncio
import logging
import signal
from typing import Optional, Dict, Any

import pandas as pd

from config.load_env import load_environment_variables
from config import config
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.trade_executor import TradeExecutor
from data.online_learning import OnlineLearner  # şimdilik sadece placeholder
from core.binance_client import create_binance_client
from models.hybrid_inference import HybridModel
from config.settings import Settings
from tg_bot.telegram_bot import TelegramBot
from data.whale_detector import MultiTimeframeWhaleDetector
from data.anomaly_detection import AnomalyDetector
from core.hybrid_mtf import MultiTimeframeHybridEnsemble


# ----------------------------------------------------------------------
# Global config / flags
# ----------------------------------------------------------------------
USE_TESTNET = getattr(Settings, "USE_TESTNET", True)
SYMBOL = getattr(Settings, "SYMBOL", "BTCUSDT")

# Global logger
system_logger: Optional[logging.Logger] = None

# Loop bekleme süresi (sn)
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))

# MTF intervals
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]


def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


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
    """
    Hem Binance canlı verisi (ms epoch) hem de offline CSV (ISO datetime string)
    ile çalışacak şekilde feature üretir.
    """
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
    """
    Basit label: horizon bar sonra close > current close ise 1 (up), yoksa 0 (down)
    """
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    labels = (future > close).astype(int)
    return labels


# ----------------------------------------------------------------------
# ATR hesaplayıcı
# ----------------------------------------------------------------------
def compute_atr_from_klines(df: pd.DataFrame, period: int = 14) -> float:
    """
    True Range bazlı ATR hesaplar.
    df: raw kline DataFrame (open, high, low, close ...)
    """
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
async def fetch_klines(
    client,
    symbol: str,
    interval: str,
    limit: int,
    logger: Optional[logging.Logger],
) -> pd.DataFrame:
    """
    Kline fetch helper.

    - Eğer client None ise:
        -> OFFLINE / DRY_RUN mod
        -> data/offline_cache/{symbol}_{interval}_6m.csv dosyasından okur
    - Eğer client varsa:
        -> Binance'ten async get_klines ile veri çeker
    """
    # OFFLINE / DRY_RUN
    if client is None:
        csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
        if not os.path.exists(csv_path):
            if logger:
                logger.error(
                    "[DATA] client=None ve offline CSV bulunamadı: %s",
                    csv_path,
                )
            raise RuntimeError(
                f"Offline kline dosyası yok: {csv_path}. "
                "Lütfen BINANCE_API_KEY/BINANCE_API_SECRET set edin "
                "veya offline cache oluşturun."
            )

        df = pd.read_csv(csv_path)
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        if logger:
            logger.info(
                "[DATA] OFFLINE mod: %s dosyasından kline yüklendi. shape=%s",
                csv_path,
                df.shape,
            )
        return df

    # ONLINE
    try:
        klines = await client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
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
        logger.info(
            "[DATA] ONLINE mod: Binance'ten kline çekildi. symbol=%s interval=%s shape=%s",
            symbol,
            interval,
            df.shape,
        )
    return df


# ----------------------------------------------------------------------
# Trading objeleri kurulum
# ----------------------------------------------------------------------
def create_trading_objects() -> Dict[str, Any]:
    """
    Tüm trading bileşenlerini (client, risk, position, trade_executor, modeller, whale) oluşturur.
    """
    global system_logger
    symbol = getattr(config, "SYMBOL", SYMBOL)
    interval = os.getenv("INTERVAL", "5m")

    # Binance client
    client = create_binance_client(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=USE_TESTNET,
        logger=system_logger,
    )

    # Risk Manager
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

    # Telegram Bot + RiskManager entegrasyonu
    tg_bot = None
    try:
        tg_bot = TelegramBot()
        if getattr(tg_bot, "dispatcher", None):
            # Bot içinde set_risk_manager varsa kullan, yoksa attribute set et
            if hasattr(tg_bot, "set_risk_manager"):
                tg_bot.set_risk_manager(risk_manager)
            else:
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
        if system_logger:
            system_logger.warning("[MAIN] TelegramBot init/set_risk_manager hata: %s", e)
        tg_bot = None

    # Position Manager
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

    # Hybrid model (ana interval)
    hybrid_model = HybridModel(
        model_dir="models",
        interval=interval,
        logger=system_logger,
    )
    try:
        if hasattr(hybrid_model, "use_lstm_hybrid"):
            hybrid_model.use_lstm_hybrid = HYBRID_MODE
    except Exception:
        pass

    # MTF hybrid ensemble (interval başına HybridModel seti)
    mtf_ensemble = None
    if USE_MTF_ENS:
        try:
            mtf_models: Dict[str, HybridModel] = {}
            for itv in MTF_INTERVALS:
                try:
                    hm = HybridModel(
                        model_dir="models",
                        interval=itv,
                        logger=system_logger,
                    )
                    if hasattr(hm, "use_lstm_hybrid"):
                        hm.use_lstm_hybrid = HYBRID_MODE
                    mtf_models[itv] = hm
                    if system_logger:
                        system_logger.info(
                            "[HYBRID-MTF] HybridModel yüklendi | interval=%s", itv
                        )
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

    # Whale detector (MTF)
    whale_detector = None
    try:
        whale_detector = MultiTimeframeWhaleDetector()
        if system_logger:
            system_logger.info(
                "[WHALE] MultiTimeframeWhaleDetector başarıyla init edildi."
            )
    except Exception as e:
        whale_detector = None
        if system_logger:
            system_logger.warning(
                "[WHALE] MultiTimeframeWhaleDetector init hata: %s",
                e,
            )

    # Trade Executor
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
        dry_run=DRY_RUN,
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

    return {
        "symbol": symbol,
        "interval": interval,
        "client": client,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "hybrid_model": hybrid_model,
        "mtf_ensemble": mtf_ensemble,   # yeni
        "mtf_model": mtf_ensemble,      # geriye dönük uyumluluk için
        "whale_detector": whale_detector,
        "tg_bot": tg_bot,
    }
# ----------------------------------------------------------------------
# Ana trading loop
# ----------------------------------------------------------------------
async def bot_loop(objs: Dict[str, Any]) -> None:
    """
    Ana trading loop:
      - Kline fetch
      - Feature build
      - Hybrid / MTF prediction
      - ATR & MTF whale meta
      - TradeExecutor.execute_decision
    """
    global system_logger

    client = objs["client"]
    risk_manager = objs["risk_manager"]
    trade_executor = objs["trade_executor"]
    whale_detector = objs.get("whale_detector")
    hybrid_model = objs["hybrid_model"]
    mtf_ensemble = objs.get("mtf_ensemble")  # yeni

    symbol = objs.get("symbol", getattr(config, "SYMBOL", SYMBOL))
    interval = objs.get("interval", os.getenv("INTERVAL", "5m"))
    data_limit = 500

    if system_logger:
        system_logger.info(
            "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s, USE_MTF_ENS=%s)",
            symbol,
            interval,
            TRAINING_MODE,
            HYBRID_MODE,
            USE_MTF_ENS,
        )

    # --- AnomalyDetector: her loop’ta kullanılacak tek instance ---
    anomaly_detector = AnomalyDetector(logger=system_logger)

    # Backward-compat alias map (asset_volume -> volume)
    alias_map = {
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }

    # ------------------------------ LOOP ------------------------------
    while True:
        try:
            # 1) Ana TF KLINES -> FEATURES
            raw_df = await fetch_klines(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=data_limit,
                logger=system_logger,
            )

            feat_df = build_features(raw_df)
            if system_logger:
                system_logger.info(
                    "[FE] Features DF shape: %s, columns=%s",
                    feat_df.shape,
                    list(feat_df.columns),
                )

            # Backward-compat aliasları ana DF için uygula
            for old_col, new_col in alias_map.items():
                if old_col not in feat_df.columns and new_col in feat_df.columns:
                    feat_df[old_col] = feat_df[new_col]

            # Anomali filtresi (IsolationForest)
            feat_df = anomaly_detector.filter_anomalies(feat_df)

            # Feature kolonları – hem eski hem yeni isimlerle uyumlu
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

            feature_cols_existing = [c for c in feature_cols if c in feat_df.columns]
            missing = [c for c in feature_cols if c not in feat_df.columns]
            if missing and system_logger:
                system_logger.warning(
                    "[FE] Eksik feature kolonları tespit edildi: %s",
                    missing,
                )

            X_live = feat_df[feature_cols_existing].tail(500)

            # 2) Single-TF hibrit skor (ana interval)
            p_arr_single, debug_single = hybrid_model.predict_proba(X_live)
            p_single = float(p_arr_single[-1])

            # Model meta
            meta = getattr(hybrid_model, "meta", {}) or {}
            model_conf_factor = float(meta.get("confidence_factor", 1.0) or 1.0)
            best_auc = float(meta.get("best_auc", 0.5) or 0.5)
            best_side = meta.get("best_side", "long")

            # 3) MTF features (1m, 5m, 15m, 1h) & whale raw
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

                        # Backward-compat alias map
                        for old_col, new_col in alias_map.items():
                            if old_col not in feat_df_itv.columns and new_col in feat_df_itv.columns:
                                feat_df_itv[old_col] = feat_df_itv[new_col]

                        mtf_feats[itv] = feat_df_itv
                        mtf_whale_raw[itv] = raw_df_itv
                    except Exception as e:
                        if system_logger:
                            system_logger.warning(
                                "[MTF] %s interval'i hazırlanırken hata: %s",
                                itv,
                                e,
                            )

            # 4) MTF ensemble skoru
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
                                system_logger.warning(
                                    "[MTF] Interval=%s için kullanılabilir feature yok, skip ediliyor.",
                                    itv,
                                )
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
                        system_logger.warning(
                            "[MTF] Ensemble hesaplanırken hata: %s",
                            e,
                        )
                    p_used = p_single
                    mtf_debug = None
                    p_1m = p_5m = p_15m = p_1h = None

            # 5) MTF Whale sinyali -> whale_meta
            whale_meta: Dict[str, Any] = {
                "direction": "none",
                "score": 0.0,
                "per_tf": {},
            }

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
                        system_logger.warning(
                            "[WHALE] MTF whale hesaplanırken hata: %s",
                            e,
                        )

            # 6) ATR Hesabı
            atr_period = int(os.getenv("ATR_PERIOD", "14"))
            atr_value = compute_atr_from_klines(raw_df, period=atr_period)

            # 7) Extra meta paket – probs + extra
            probs = {
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

            if system_logger:
                system_logger.info(
                    "[HYBRID] mode=%s n_samples=%d n_features=%d "
                    "p_sgd_mean=%.4f, p_lstm_mean=%.4f, p_hybrid_mean=%.4f, "
                    "best_auc=%.4f, best_side=%s",
                    debug_single.get("mode", "unknown"),
                    len(X_live),
                    X_live.shape[1],
                    float(debug_single.get("p_sgd_mean", 0.0)),
                    float(debug_single.get("p_lstm_mean", 0.5)),
                    float(debug_single.get("p_hybrid_mean", p_used)),
                    best_auc,
                    best_side,
                )

            # 8) Sinyal üretimi
            long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
            short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))

            if p_used >= long_thr:
                signal = "long"
            elif p_used <= short_thr:
                signal = "short"
            else:
                signal = "hold"

            # ------------------------------
            # 9) Çoklu TF trend filtresi + mikro filtre
            # ------------------------------
            # p_1m, p_5m, p_15m, p_1h zaten yukarıda set edildi (MTF varsa)

            # --- Trend filtresi (1h/15m hard veto) ---
            if p_1h is not None and p_15m is not None:
                if signal == "long" and not (p_1h > 0.6 and p_15m > 0.5):
                    if system_logger:
                        system_logger.info(
                            "[TREND_FILTER] 1h/15m filtre nedeniyle LONG -> HOLD "
                            "(p_1h=%.4f, p_15m=%.4f)",
                            p_1h,
                            p_15m,
                        )
                    signal = "hold"
                elif signal == "short" and not (p_1h < 0.4 and p_15m < 0.5):
                    if system_logger:
                        system_logger.info(
                            "[TREND_FILTER] 1h/15m filtre nedeniyle SHORT -> HOLD "
                            "(p_1h=%.4f, p_15m=%.4f)",
                            p_1h,
                            p_15m,
                        )
                    signal = "hold"

            # --- 1m mikro filtre (hafif fren) ---
            micro_conf_scale = 1.0
            if signal == "long" and isinstance(p_1m, float) and p_1m < 0.30:
                micro_conf_scale = 0.7  # %30 küçült
            elif signal == "short" and isinstance(p_1m, float) and p_1m > 0.70:
                micro_conf_scale = 0.7

            # Model confidence factor'u mikro filtre ile birleştir
            effective_model_conf = float(model_conf_factor) * micro_conf_scale

            # Whale meta güvenli çek
            whale_dir = None
            whale_score = None
            if isinstance(whale_meta, dict):
                whale_dir = whale_meta.get("direction")
                whale_score = whale_meta.get("score")

            # Genişletilmiş SIGNAL log'u
            if system_logger:
                system_logger.info(
                    "[SIGNAL] p_used=%.4f, long_thr=%.3f, short_thr=%.3f, "
                    "signal=%s, model_conf_factor=%.3f, effective_conf=%.3f, "
                    "p_1m=%s, p_5m=%s, p_15m=%s, p_1h=%s, "
                    "whale_dir=%s, whale_score=%s",
                    p_used,
                    long_thr,
                    short_thr,
                    signal,
                    float(model_conf_factor),
                    effective_model_conf,
                    f"{p_1m:.4f}" if isinstance(p_1m, float) else "None",
                    f"{p_5m:.4f}" if isinstance(p_5m, float) else "None",
                    f"{p_15m:.4f}" if isinstance(p_15m, float) else "None",
                    f"{p_1h:.4f}" if isinstance(p_1h, float) else "None",
                    whale_dir if whale_dir is not None else "None",
                    f"{whale_score:.3f}" if isinstance(whale_score, (int, float)) else "None",
                )

            # extra paketinde sadece model_confidence_factor'ü güncelle
            extra["model_confidence_factor"] = effective_model_conf

            # Label diagnostic
            last_label = build_labels(feat_df).iloc[-1]
            if system_logger:
                system_logger.info(
                    "[LABEL_CHECK] last_label(horizon=1)=%s (1=up,0=down)",
                    last_label,
                )

            # 10) Kararı TradeExecutor'a gönder
            last_price = float(raw_df["close"].iloc[-1])

            await trade_executor.execute_decision(
                signal=signal,
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
    """
    Asenkron ana giriş:
      - ENV yüklenir
      - logger initialize edilir
      - trading objeleri oluşturulur
      - bot_loop başlatılır
    """
    global system_logger
    global BINANCE_API_KEY, BINANCE_API_SECRET
    global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS, DRY_RUN

    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")

    if system_logger is None:
        print("[MAIN] system_logger init edilemedi!")
        return

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

    HYBRID_MODE = get_bool_env("HYBRID_MODE", HYBRID_MODE)
    TRAINING_MODE = get_bool_env("TRAINING_MODE", TRAINING_MODE)
    USE_MTF_ENS = get_bool_env("USE_MTF_ENS", USE_MTF_ENS)
    DRY_RUN = get_bool_env("DRY_RUN", DRY_RUN)

    if not (BINANCE_API_KEY and BINANCE_API_SECRET):
        system_logger.warning(
            "[BINANCE] API key/secret env'de yok. DRY_RUN modunda çalıştığından emin ol."
        )

    system_logger.info(
        "[MAIN] TRAINING_MODE=%s -> %s",
        TRAINING_MODE,
        "Offline/eğitim modu" if TRAINING_MODE else "Normal çalışma modu (trade logic aktif)",
    )
    system_logger.info(
        "[MAIN] HYBRID_MODE=%s -> %s",
        HYBRID_MODE,
        "LSTM+SGD hibrit skor kullanılacak (mümkünse)."
        if HYBRID_MODE
        else "Sadece SGD/online skor kullanılacak.",
    )

    trading_objects = create_trading_objects()
    await bot_loop(trading_objects)


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
            pass  # Windows vs.

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


if __name__ == "__main__":
    main()
