import os
import asyncio
import signal
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Config & Core imports
# ----------------------------------------------------------------------
try:
    # Yeni yapı
    from config.load_env import load_environment_variables
except ImportError:
    # Eski yapı veya lokal kullanım için fallback
    def load_environment_variables() -> Dict[str, str]:
        return dict(os.environ)

from core.logger import setup_logger
from core.risk_manager import RiskManager  # risk tarafında kullanmak istersen
from models.hybrid_inference import HybridModel, HybridMultiTFModel

# ----------------------------------------------------------------------
# Global logger
# ----------------------------------------------------------------------
system_logger: Optional[logging.Logger] = None

# ----------------------------------------------------------------------
# ENV / sabitler
# ----------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5m")

TRAINING_MODE = os.getenv("TRAINING_MODE", "false").lower() == "true"
HYBRID_MODE = os.getenv("HYBRID_MODE", "true").lower() == "true"

# ------------------------------------------------------------
# Threshold ve Multi-Timeframe ensemble ayarları
# ------------------------------------------------------------
LONG_THRESHOLD = float(os.getenv("LONG_THRESHOLD", "0.5"))
SHORT_THRESHOLD = float(os.getenv("SHORT_THRESHOLD", "0.3"))

# Multi-timeframe ensemble kullanılsın mı?
USE_MTF_ENS = os.getenv("USE_MTF_ENS", "false").lower() == "true"

# Ensemble’da kullanılacak interval listesi
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]

# Loop bekleme süresi (saniye)
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))

# DRY-RUN: API key yoksa gerçek emir yok, sadece log
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true" or not (
    BINANCE_API_KEY and BINANCE_API_SECRET
)


# ----------------------------------------------------------------------
# Basit feature engineering helper
#  -> offline_eval_hybrid / offline_backtest_hybrid ile aynı schema
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitimde kullanılan feature setine yakın, sabit bir şema üretir.
    HybridModel._prepare_feature_matrix bu kolonları bekliyor:

      ['open_time', 'open', 'high', 'low', 'close', 'volume',
       'close_time', 'quote_asset_volume', 'number_of_trades',
       'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore',
       'hl_range', 'oc_change', 'return_1', 'return_3', 'return_5',
       'ma_5', 'ma_10', 'ma_20', 'vol_10', 'dummy_extra']
    """
    df = raw_df.copy()

    # Zaten offline_cache içinde bu kolonlar var:
    # ['open_time', 'open', 'high', 'low', 'close', 'volume',
    #  'close_time', 'quote_asset_volume', 'number_of_trades',
    #  'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']

    # Price / volume feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    # Eğitime uyum için ekstra dummy kolon
    df["dummy_extra"] = 0.0

    # NA temizliği
    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Basit label helper (offline eval / log için)
# ----------------------------------------------------------------------
def build_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Eğitimde kullanılan label mantığı:

        future_close = df["close"].shift(-horizon)
        ret = future_close / df["close"] - 1.0
        y = (ret > 0.0).astype(int)

    Burada sadece *opsiyonel* olarak log / offline kıyas için kullanıyoruz.
    """
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    y = (ret > 0.0).astype(int)
    return y


# ----------------------------------------------------------------------
# Data fetch helpers
# ----------------------------------------------------------------------
def load_offline_klines_from_cache(
    symbol: str, interval: str, limit: int = 500
) -> pd.DataFrame:
    """
    Offline cache'den veriyi okur (ör: data/offline_cache/BTCUSDT_5m_6m.csv)
    Development / test için güvenli.
    """
    cache_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Offline cache bulunamadı: {cache_path}")

    df = pd.read_csv(cache_path)
    df = df.tail(limit).reset_index(drop=True)
    return df


async def fetch_raw_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Şu anda *sadece offline cache* kullanıyoruz.

    İLERİDE:
        - Buraya kendi Binance client'ını (websocket veya REST) entegre edebilirsin.
        - Örneğin data/klines.py içinde bir fetch_live_klines fonksiyonun varsa
          onu çağıracak şekilde güncelleyebilirsin.
    """
    # TODO: Canlı veri için kendi client'ını entegre et
    return load_offline_klines_from_cache(symbol, interval, limit)


# ----------------------------------------------------------------------
# Trading nesneleri
# ----------------------------------------------------------------------
def create_trading_objects() -> Dict[str, Any]:
    global system_logger

    # ------------------------------------------------------------
    # RiskManager
    # ------------------------------------------------------------
    risk_manager = RiskManager(logger=system_logger)

    # ------------------------------------------------------------
    # Tek-timeframe HybridModel (INTERVAL için)
    # ------------------------------------------------------------
    hybrid_model = HybridModel(
        model_dir="models",
        interval=INTERVAL,
        logger=system_logger,
    )

    # Meta’dan model güven katsayısı türet
    best_auc = float(hybrid_model.meta.get("best_auc", 0.5))
    best_side = hybrid_model.meta.get("best_side", "best")

    # AUC 0.5 → 1.0, AUC 0.7 → ~1.4, AUC 0.8 → ~1.6 gibi
    model_confidence_factor = 1.0 + max(0.0, best_auc - 0.5) * 2.0

    system_logger.info(
        "[RISK] Using offline meta for interval=%s: best_auc=%.4f, best_side=%s, "
        "model_confidence_factor=%.3f",
        INTERVAL,
        best_auc,
        best_side,
        model_confidence_factor,
    )

    # ------------------------------------------------------------
    # Multi-timeframe Hybrid ensemble (opsiyonel)
    # ------------------------------------------------------------
    mtf_model = None
    if USE_MTF_ENS:
        try:
            mtf_model = HybridMultiTFModel(
                model_dir="models",
                intervals=MTF_INTERVALS,
                logger=system_logger,
            )
            system_logger.info(
                "[HYBRID-MTF] Multi-timeframe ensemble aktif. Intervals=%s",
                ",".join(MTF_INTERVALS),
            )
        except Exception as e:
            system_logger.warning(
                "[HYBRID-MTF] Ensemble modeli init edilemedi: %s. Tek timeframe ile devam.",
                e,
            )
            mtf_model = None

    # ------------------------------------------------------------
    # TradeExecutor
    # ------------------------------------------------------------
    try:
        from core.trade_executor import TradeExecutor  # type: ignore

        trade_executor = TradeExecutor(
            risk_manager=risk_manager,
            logger=system_logger,
            dry_run=DRY_RUN,
        )
    except Exception as e:
        system_logger.warning(
            "[MAIN] TradeExecutor init edilemedi (%s). Emir gönderimi devre dışı.", e
        )
        trade_executor = None

    return {
        "risk_manager": risk_manager,
        "hybrid_model": hybrid_model,
        "mtf_model": mtf_model,
        "trade_executor": trade_executor,
        "model_confidence_factor": model_confidence_factor,
        "best_auc": best_auc,
        "best_side": best_side,
    }


# ----------------------------------------------------------------------
# Ana bot loop
# ----------------------------------------------------------------------
async def bot_loop(trading_objects: Dict[str, Any]) -> None:
    global system_logger

    hybrid_model: HybridModel = trading_objects["hybrid_model"]
    trade_executor = trading_objects["trade_executor"]
    model_confidence_factor: float = trading_objects["model_confidence_factor"]
    best_auc: float = trading_objects["best_auc"]
    best_side: str = trading_objects["best_side"]

    system_logger.info(
        "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s)",
        SYMBOL,
        INTERVAL,
        TRAINING_MODE,
        HYBRID_MODE,
    )

    while True:
        try:
            system_logger.info(
                "[DATA] Starting data pipeline for %s (%s, limit=%d)",
                SYMBOL,
                INTERVAL,
                500,
            )
            raw_df = await fetch_raw_klines(SYMBOL, INTERVAL, limit=500)

            # Feature engineering
            feat_df = build_features(raw_df)
            system_logger.info(
                "[FE] Features DF shape: %s, columns=%s",
                feat_df.shape,
                list(feat_df.columns),
            )

            # Hybrid model tahmini
            p_arr, debug = hybrid_model.predict_proba(feat_df)

            if p_arr is None or len(p_arr) == 0:
                system_logger.warning(
                    "[PRED] Empty probability array, skipping this iteration."
                )
                await asyncio.sleep(LOOP_SLEEP_SECONDS)
                continue

            # Son barın skoru
            p_hybrid = float(p_arr[-1])

            # debug içinden SGD ortalamasını çek
            p_sgd_mean = float(debug.get("p_sgd_mean", 0.5))
            p_lstm_mean = float(debug.get("p_lstm_mean", 0.5))
            mode = debug.get("mode", "unknown")

# ------------------------------------------------------------
# PRED LOG
# ------------------------------------------------------------
system_logger.info(
    "[PRED] p_sgd=%.4f, p_hybrid=%.4f (HYBRID_MODE=%s)",
    p_sgd,
    p_hybrid,
    HYBRID_MODE,
)

# ------------------------------------------------------------
# Multi-Timeframe ensemble kullanımı (varsa)
# ------------------------------------------------------------
# Varsayılan olarak tek interval hibrit skorunu kullanıyoruz
p_used = float(p_hybrid if HYBRID_MODE else p_sgd)
mtf_debug = None

if USE_MTF_ENS and mtf_model is not None:
    try:
        # Şimdilik sadece mevcut interval'in feature DF'ini veriyoruz.
        # İleride 1m/15m/1h için de feature pipeline ekleyip buraya koyacağız.
        X_dict = {INTERVAL: features_df}

        ensemble_p, mtf_debug = mtf_model.predict_proba_multi(X_dict)
        p_used = float(ensemble_p)

        system_logger.info(
            "[HYBRID-MTF] ensemble_p=%.4f (INTERVAL=%s, USE_MTF_ENS=%s)",
            p_used,
            INTERVAL,
            USE_MTF_ENS,
        )
    except Exception as e:
        system_logger.warning(
            "[HYBRID-MTF] ensemble hesaplaması başarısız, single TF kullanılacak: %s",
            e,
        )

# ------------------------------------------------------------
# Threshold bazlı sinyal kararı
# ------------------------------------------------------------
if p_used >= LONG_THRESHOLD:
    signal = "long"
elif p_used <= SHORT_THRESHOLD:
    signal = "short"
else:
    signal = "hold"

system_logger.info(
    "[SIGNAL] p_used=%.4f, long_thr=%.3f, short_thr=%.3f, signal=%s, model_conf_factor=%.3f",
    p_used,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    signal,
    model_confidence_factor,
)

            # ------------------------------------------------------------------
            # Opsiyonel: label ile kısa bir realtime kıyas (sadece log)
            # ------------------------------------------------------------------
            # Burada horizon=1 için future_close kullanarak ret ve y hesaplıyoruz.
            # Canlı trade'i etkilemez, sadece loglama amaçlı.
            try:
                y_series = build_labels(raw_df["close"], horizon=1)
                # Feature DF dropna yüzünden ufak kayma olabilir;
                # son bar için geleceği bilmiyoruz, o yüzden -2 ile hizalayabiliriz.
                if len(y_series) > 2:
                    last_label = int(y_series.iloc[-2])
                    system_logger.info(
                        "[LABEL_CHECK] last_label(horizon=1)=%d (1=up,0=down)",
                        last_label,
                    )
            except Exception as e:
                system_logger.warning(
                    "[LABEL_CHECK] label hesaplanırken hata: %s", e
                )

            # ------------------------------------------------------------------
            # Trade execution
            # ------------------------------------------------------------------
            if TRAINING_MODE:
                system_logger.info(
                    "[MAIN] TRAINING_MODE aktif, execute_decision çağrılmayacak. "
                    "Sinyal sadece loglandı."
                )
            else:
                last_price = float(raw_df["close"].iloc[-1])

                if trade_executor is not None and hasattr(
                    trade_executor, "execute_decision"
                ):
                    # Güvenlik için p_sgd/p_hybrid'i tekrar al
                    p_sgd_val = float(p_sgd_mean)
                    p_hybrid_val = float(p_hybrid)

                    extra = {
                        "model_confidence_factor": float(model_confidence_factor),
                        "best_auc": float(best_auc),
                        "best_side": best_side,
                    }

                    # TradeExecutor imzan senin projende farklıysa sadece bu çağrıyı
                    # kendi versiyonuna uyarlaman yeterli.
                    await trade_executor.execute_decision(
                        signal=signal,
                        symbol=SYMBOL,
                        price=last_price,
                        size=None,  # TODO: RiskManager ile dinamik position size
                        interval=INTERVAL,
                        training_mode=TRAINING_MODE,
                        hybrid_mode=HYBRID_MODE,
                        probs={
                            "p_sgd": p_sgd_val,
                            "p_hybrid": p_hybrid_val,
                        },
                        extra=extra,
                    )
                else:
                    system_logger.info(
                        "[EXEC] TradeExecutor yok veya execute_decision tanımlı değil; "
                        "sinyal sadece loglandı."
                    )

        except asyncio.CancelledError:
            system_logger.info("[MAIN] Bot loop cancelled, shutting down gracefully.")
            break
        except Exception as e:
            if system_logger:
                system_logger.exception("[MAIN] Error in bot_loop: %s", e)
            else:
                print(f"[MAIN] Error in bot_loop: {e}")

        # Sonraki loop için bekleme
        await asyncio.sleep(LOOP_SLEEP_SECONDS)


# ----------------------------------------------------------------------
# Async main & entrypoint
# ----------------------------------------------------------------------
async def async_main() -> None:
    global system_logger

    # ENV yükle
    load_environment_variables()

    # Logger hazırla
    system_logger = setup_logger()
    system_logger.info("[LOGGER] Loggers initialized (system, error, trades).")

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


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown için sinyal handler
    stop_event = asyncio.Event()

    def _signal_handler(sig_num, _frame):
        if system_logger:
            system_logger.info("[MAIN] Signal received (%s), shutting down...", sig_num)
        loop.call_soon_threadsafe(stop_event.set)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        # Windows vs. uyumsuz ortamlar için
        pass

    try:
        loop.run_until_complete(async_main())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
