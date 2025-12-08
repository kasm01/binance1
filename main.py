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
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kline DF -> feature DF
    offline_train_hybrid.py ile birebir uyumlu, böylece
    offline/online tutarlılığı bozulmaz.
    """
    df = df.copy()

    # Zaman kolonlarını saniyeye çevir
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], unit="ms", utc=True)
            # FutureWarning için aynı patterni kullanıyoruz
            df[col] = dt.view("int64") / 1e9  # saniye

    # Float’a cast
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(float)

    # Basit fiyat/vol feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["volume"].rolling(10).mean()

    # Placeholder / genişleme alanı
    df["dummy_extra"] = 0.0

    # NaN doldurma (offline_train_hybrid ile aynı pattern)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

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
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Binance'ten kline çeker ve DataFrame döner.
    client: core.binance_client.create_binance_client ile oluşturulan client
    """
    klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
    # Beklenen kolon sırası: [open_time, open, high, low, close, volume, close_time,
    # quote_asset_volume, number_of_trades, taker_buy_base_volume,
    # taker_buy_quote_volume, ignore]
    cols = [
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
    df = pd.DataFrame(klines, columns=cols)
    if logger:
        logger.info(
            "[DATA] Fetched klines: symbol=%s interval=%s shape=%s",
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
    Binance client, RiskManager, PositionManager, TradeExecutor,
    HybridModel ve MTF modelini ayağa kaldırır.
    """
    global system_logger, BINANCE_API_KEY, BINANCE_API_SECRET

    if system_logger is None:
        raise RuntimeError("system_logger init edilmemiş!")

    symbol = getattr(config, "SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "5m")

    # Binance client
    client = create_binance_client(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=get_bool_env("BINANCE_TESTNET", False),
        logger=system_logger,
    )

    # Risk Manager
    risk_manager = RiskManager()

    # Position Manager (Redis + opsiyonel Postgres)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    position_manager = PositionManager(
        redis_url=redis_url,
        risk_manager=risk_manager,
        logger=system_logger,
    )

    # Trade Executor (gerçek emir / dry-run + whale/risk aware)
    trade_executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        position_manager=position_manager,
        symbol=symbol,
        dry_run=DRY_RUN,
        logger=system_logger,
    )

    # Online learner (şimdilik sadece placeholder, istersen sonra bağlarız)
    online_learner = None
    try:
        online_learner = OnlineLearner(
            symbol=symbol,
            interval=interval,
            logger=system_logger,
        )
    except Exception as e:
        system_logger.warning(
            "[ONLINE_LEARNER] Init başarısız, devre dışı bırakılıyor: %s", e
        )
        online_learner = None

    # Ana HybridModel (interval = INTERVAL, örn: 5m)
    hybrid_model = HybridModel(interval=interval, logger=system_logger)

    # Multi-timeframe ensemble modeli (1m,5m,15m,1h)
    mtf_model: Optional[HybridMultiTFModel] = None
    if USE_MTF_ENS:
        try:
            mtf_model = HybridMultiTFModel(
                intervals=["1m", "5m", "15m", "1h"],
                logger=system_logger,
            )
            system_logger.info(
                "[HYBRID-MTF] Multi-timeframe ensemble aktif. Intervals=%s",
                ",".join(["1m", "5m", "15m", "1h"]),
            )
        except Exception as e:
            system_logger.warning(
                "[HYBRID-MTF] Init başarısız, ensemble devre dışı: %s", e
            )
            mtf_model = None

    # Whale detector şimdilik yok, ileride eklenecek
    whale_detector = None

    return {
        "client": client,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "online_learner": online_learner,
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
            whale_meta = whale_detector.get_last_state() if whale_detector else None

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

            system_logger.info(
                "[SIGNAL] p_used=%.4f, long_thr=%.3f, short_thr=%.3f, "
                "signal=%s, model_conf_factor=%.3f, p_1h=%s, p_15m=%s",
                p_used,
                long_thr,
                short_thr,
                signal,
                model_conf_factor,
                f"{p_1h:.4f}" if p_1h is not None else "None",
                f"{p_15m:.4f}" if p_15m is not None else "None",
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
