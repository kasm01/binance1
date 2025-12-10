import os
import asyncio
import logging
import signal
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from config.load_env import load_environment_variables
from config import config
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.position_manager import PositionManager
from core.trade_executor import TradeExecutor
from data.online_learning import OnlineLearner  # şimdilik sadece placeholder
from models.hybrid_inference import HybridModel, HybridMultiTFModel
from core.binance_client import create_binance_client
from models.hybrid_inference import HybridModel, HybridMultiTFModel

# Config değerlerini çek
USE_TESTNET = getattr(config, "USE_TESTNET", False)

BINANCE_API_KEY = getattr(config, "BINANCE_API_KEY", None)
BINANCE_API_SECRET = getattr(config, "BINANCE_API_SECRET", None)

REDIS_URL = getattr(config, "REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = getattr(config, "REDIS_KEY_PREFIX", "bot:positions")

SYMBOL = getattr(config, "SYMBOL", "BTCUSDT")


# ----------------------------------------------------------------------
# Global logger
# ----------------------------------------------------------------------
system_logger: Optional[logging.Logger] = None

# Loop bekleme süresi (sn)
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))


# ----------------------------------------------------------------------
# Env yardımcı fonksiyon
# ----------------------------------------------------------------------
def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


# ----------------------------------------------------------------------
# Global env flag’ler (async_main içinde tekrar güncellenecek)
# ----------------------------------------------------------------------
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

    # --------------------------------------------------
    # 1) Zaman kolonlarını normalize et (open_time / close_time)
    #    - Eğer kolon numeric ise: unit="ms" ile epoch'tan çevir
    #    - Değilse: direkt to_datetime
    #    Sonuçta: saniye bazlı epoch float (Unix time)
    # --------------------------------------------------
    for col in ["open_time", "close_time"]:
        if col not in df.columns:
            continue

        # dtype numeric mi?
        if pd.api.types.is_numeric_dtype(df[col]):
            # Binance'ten gelen ms epoch
            dt = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            # Offline CSV'de string timestamp ise
            dt = pd.to_datetime(df[col], utc=True)

        # nanosecond -> second
        df[col] = dt.astype("int64") / 1e9

    # --------------------------------------------------
    # 2) Temel numeric cast
    # --------------------------------------------------
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
    int_cols = [
        "number_of_trades",
    ]

    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)  # model için float da iş görür

    # ignore yoksa ekle
    if "ignore" not in df.columns:
        df["ignore"] = 0.0

    # --------------------------------------------------
    # 3) Teknik feature'lar
    # --------------------------------------------------
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["volume"].rolling(10).std()

    # Dummy ekstra feature (modelle uyum için)
    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    # --------------------------------------------------
    # 4) NaN temizliği
    # --------------------------------------------------
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
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Kline fetch helper.

    - Eğer client None ise:
        -> DRY_RUN / offline mod varsayılır
        -> data/offline_cache/{symbol}_{interval}_6m.csv dosyasından okur
    - Eğer client varsa:
        -> Binance'ten async get_klines ile veri çeker
    """

    # ---------------------------------------------------------
    # 1) OFFLINE / DRY_RUN MODU (client is None)
    # ---------------------------------------------------------
    if client is None:
        csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
        if not os.path.exists(csv_path):
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

        # Eğer limit'ten fazla satır varsa sadece son 'limit' satırı al
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        logger.info(
            "[DATA] OFFLINE mod: %s dosyasından kline yüklendi. shape=%s",
            csv_path,
            df.shape,
        )
        return df

    # ---------------------------------------------------------
    # 2) ONLINE MOD (Binance Async Client ile)
    # ---------------------------------------------------------
    try:
        klines = await client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
    except Exception as e:
        logger.error("[DATA] Binance get_klines hatası: %s", e)
        raise

    # klines -> DataFrame'e çevir
    # python-binance standard kline yapısı:
    # [
    #   [
    #     open_time, open, high, low, close, volume,
    #     close_time, quote_asset_volume, number_of_trades,
    #     taker_buy_base_volume, taker_buy_quote_volume, ignore
    #   ],
    #   ...
    # ]
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

    # Tip düzeltmeleri
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
    int_cols = [
        "open_time",
        "close_time",
        "number_of_trades",
    ]

    for c in float_cols:
        df[c] = df[c].astype(float)

    for c in int_cols:
        df[c] = df[c].astype(int)

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

    # -----------------------------
    # Temel parametreler
    # -----------------------------
    symbol = getattr(config, "SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "5m")

    # DRY_RUN flag
    DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

    # -----------------------------
    # Binance client
    # -----------------------------
    client = create_binance_client(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=USE_TESTNET,
        logger=system_logger,
    )

    # --------------------------------------------------
    # Risk Manager
    # --------------------------------------------------

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
        equity_start_of_day=equity_start_of_day,  # ✅ DOĞRU İSİM
        logger=system_logger,                      # ✅ RiskManager logger alıyor
    )
    # -----------------------------
    # Position Manager (Redis + opsiyonel Postgres)
    # -----------------------------
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

    # ---------------------------------------------------------
    #  Model (HybridModel + opsiyonel MTF ensemble)
    # ---------------------------------------------------------

    # Tek timeframe hibrit model
    hybrid_model = HybridModel(
        interval=interval,
        model_dir="models",
    )

    # Eğer sınıfta böyle bir attribute varsa, env’den gelen flag ile set et
    try:
        if hasattr(hybrid_model, "use_lstm_hybrid"):
            hybrid_model.use_lstm_hybrid = HYBRID_MODE
    except Exception:
        # Sıkıntı olursa sessizce geç, en kötü sadece SGD kullanır
        pass

    # Multi-timeframe ensemble modeli (opsiyonel)
    mtf_model = None
    if USE_MTF_ENS:
        mtf_intervals = ["1m", "5m", "15m", "1h"]
        mtf_model = HybridMultiTFModel(
            intervals=mtf_intervals,
            model_dir="models",
        )
        try:
            if hasattr(mtf_model, "use_lstm_hybrid"):
                mtf_model.use_lstm_hybrid = HYBRID_MODE
        except Exception:
            pass


    # -----------------------------
    # Whale detector
    # -----------------------------
    try:
        from core.whale_detector import WhaleDetector
        # Yeni WhaleDetector API'sinde symbol/logger yok, basit init
        whale_detector = WhaleDetector()
        system_logger.info("[WHALE] WhaleDetector başarıyla init edildi.")
    except Exception as e:
        system_logger.warning("[WHALE] WhaleDetector init edilemedi: %s", e)
        whale_detector = None

    # -----------------------------
    # Trade Executor – YENİ İMZAYA GÖRE
    # -----------------------------
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
        client=client,                    # <-- sadece BURADA client
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
        "mtf_model": mtf_model,
        "whale_detector": whale_detector,
    }


# ----------------------------------------------------------------------
# Ana trading loop
# ----------------------------------------------------------------------
async def bot_loop(objs: Dict[str, Any]) -> None:
    """
    Ana trading loop'u:
      - Kline fetch
      - Feature build
      - Hybrid / MTF prediction
      - ATR & whale meta
      - TradeExecutor.execute_decision
    """
    client = objs["client"]
    risk_manager = objs["risk_manager"]
    trade_executor = objs["trade_executor"]
    whale_detector = objs.get("whale_detector")
    hybrid_model = objs["hybrid_model"]
    mtf_model = objs.get("mtf_model")

    symbol = getattr(config, "SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "5m")
    data_limit = 500

    global system_logger
    system_logger.info(
        "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s, USE_MTF_ENS=%s)",
        symbol,
        interval,
        TRAINING_MODE,
        HYBRID_MODE,
        USE_MTF_ENS,
    )

    while True:
        try:
            # ------------------------------
            # 1) KLINES -> FEATURES
            # ------------------------------
            raw_df = await fetch_klines(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=data_limit,
                logger=system_logger,
            )

            feat_df = build_features(raw_df)
            system_logger.info(
                "[FE] Features DF shape: %s, columns=%s",
                feat_df.shape,
                list(feat_df.columns),
            )

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
            X_live = feat_df[feature_cols].tail(500)

            # --------------------------------
            # 2) Hybrid / MTF Prediction
            # --------------------------------
            mtf_debug = None
            if USE_MTF_ENS and mtf_model is not None:
                mtf_feat_dict: Dict[str, pd.DataFrame] = {}
                for itv in ["1m", "5m", "15m", "1h"]:
                    raw_m = await fetch_klines(
                        client=client,
                        symbol=symbol,
                        interval=itv,
                        limit=data_limit,
                        logger=system_logger,
                    )
                    f_m = build_features(raw_m)
                    mtf_feat_dict[itv] = f_m[feature_cols].tail(500)

                p_live, mtf_debug = mtf_model.predict_proba_multi(mtf_feat_dict)
                p_single = hybrid_model.predict_proba_single(X_live)[0]

                system_logger.info(
                    "[HYBRID-MTF] ensemble_p=%.4f (USE_MTF_ENS=True) | single_tf_p=%.4f",
                    p_live,
                    p_single,
                )
            else:
                p_live = hybrid_model.predict_proba_single(X_live)[0]
                p_single = p_live

            p_used = float(p_live)

            # RiskManager'dan model meta çek
            model_conf_factor = getattr(risk_manager, "model_confidence_factor", 1.0)
            best_auc = getattr(risk_manager, "best_auc", 0.5)
            best_side = getattr(risk_manager, "best_side", "long")

            system_logger.info(
                "[HYBRID] mode=sgd_only n_samples=%d n_features=%d "
                "p_sgd_mean=%.4f, p_lstm_mean=%.4f, p_hybrid_mean=%.4f, "
                "best_auc=%.4f, best_side=%s",
                len(X_live),
                X_live.shape[1],
                float(hybrid_model.last_debug.get("p_sgd_mean", 0.0)),
                float(hybrid_model.last_debug.get("p_lstm_mean", 0.5)),
                float(hybrid_model.last_debug.get("p_hybrid_mean", p_used)),
                best_auc,
                best_side,
            )

            # ------------------------------
            # 3) ATR Hesabı
            # ------------------------------
            atr_period = int(os.getenv("ATR_PERIOD", "14"))
            atr_value = compute_atr_from_klines(raw_df, period=atr_period)

            # ------------------------------
            # 4) Whale Durumu
            # ------------------------------
            whale_meta = None
            if whale_detector is not None:
                try:
                    # data/whale_detector.py içindeki from_klines(df) kullanılıyor
                    whale_signal = whale_detector.from_klines(raw_df)

                    whale_meta = {
                        "direction": whale_signal.direction,
                        "score": whale_signal.score,
                        "reason": whale_signal.reason,
                        "meta": whale_signal.meta,
                    }
                except Exception as e:
                    system_logger.warning("[WHALE] from_klines hata: %s", e)
                    whale_meta = None

            # ------------------------------
            # 5) Extra meta paket
            # ------------------------------
            extra = {
                "model_confidence_factor": model_conf_factor,
                "best_auc": best_auc,
                "best_side": best_side,
                "mtf_debug": mtf_debug,
                "whale_meta": whale_meta,
                "atr": atr_value,  # ATR TradeExecutor'a taşınacak
            }

            probs = {
                "p_used": p_used,
                "p_single": p_single,
                "p_sgd_mean": float(hybrid_model.last_debug.get("p_sgd_mean", 0.0)),
                "p_lstm_mean": float(hybrid_model.last_debug.get("p_lstm_mean", 0.5)),
            }

            # ------------------------------
            # 6) Sinyal üretimi (model tabanlı)
            # ------------------------------
            long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
            short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))

            if p_used >= long_thr:
                signal = "long"
            elif p_used <= short_thr:
                signal = "short"
            else:
                signal = "hold"

            # Trend filtresi (1h / 15m)
            p_1h = None
            p_15m = None
            if mtf_debug and "per_interval" in mtf_debug:
                p_1h = mtf_debug["per_interval"].get("1h", {}).get("p_last")
                p_15m = mtf_debug["per_interval"].get("15m", {}).get("p_last")

            if p_1h is not None and p_15m is not None:
                if signal == "long" and not (p_1h > 0.6 and p_15m > 0.5):
                    system_logger.info(
                        "[TREND_FILTER] 1h/15m filtre nedeniyle LONG sinyali HOLD'a çekildi (p_1h=%.4f, p_15m=%.4f).",
                        p_1h,
                        p_15m,
                    )
                    signal = "hold"
                elif signal == "short" and not (p_1h < 0.4 and p_15m < 0.5):
                    system_logger.info(
                        "[TREND_FILTER] 1h/15m filtre nedeniyle SHORT sinyali HOLD'a çekildi (p_1h=%.4f, p_15m=%.4f).",
                        p_1h,
                        p_15m,
                    )
                    signal = "hold"

            whale_dir = whale_meta["direction"] if isinstance(whale_meta, dict) else None
            whale_score = whale_meta["score"] if isinstance(whale_meta, dict) else None

            system_logger.info(
                "[SIGNAL] p_used=%.4f, long_thr=%.3f, short_thr=%.3f, "
                "signal=%s, model_conf_factor=%.3f, p_1h=%s, p_15m=%s, "
                "whale_dir=%s, whale_score=%s",
                p_used,
                long_thr,
                short_thr,
                signal,
                model_conf_factor,
                f"{p_1h:.4f}" if p_1h is not None else "None",
                f"{p_15m:.4f}" if p_15m is not None else "None",
                whale_dir if whale_dir is not None else "None",
                f"{whale_score:.3f}" if isinstance(whale_score, (int, float)) else "None",
            )

            # ------------------------------
            # 7) Label check (diagnostic)
            # ------------------------------
            last_label = build_labels(feat_df).iloc[-1]
            system_logger.info(
                "[LABEL_CHECK] last_label(horizon=1)=%s (1=up,0=down)",
                last_label,
            )

            # ------------------------------
            # 8) Kararı TradeExecutor'a gönder
            # ------------------------------
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

    # ENV yükle (.env vs.)
    load_environment_variables()

    # Logger kurulumu
    setup_logger()
    system_logger = logging.getLogger("system")

    if system_logger is None:
        print("[MAIN] system_logger init edilemedi!")
        return

    # Env değerlerini güncelle
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

    # Trading objeleri
    trading_objects = create_trading_objects()

    # Ana loop
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
            # Windows vs.
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


if __name__ == "__main__":
    main()
