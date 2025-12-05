import os
import asyncio
import signal
import json
import logging
from datetime import datetime
from typing import Dict, Any

import pandas as pd

from config.load_env import load_environment_variables
import config as app_config
from data.online_learning import OnlineLearner
from models.hybrid_inference import HybridModel
from core.logger import setup_logger
from core.risk_manager import RiskManager
# Binance client helper
try:
    from core.binance_client import create_binance_client  # Eğer projede varsa buradan al
except ModuleNotFoundError:
    # Fallback: doğrudan python-binance Client kullan
    from binance.client import Client

    def create_binance_client(cfg) -> Client:
        """
        Basit Binance client oluşturucu.
        Ortam değişkenlerinden API key/secret okur,
        config içinden USE_TESTNET vb. alabilir.
        """
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        use_testnet = getattr(cfg, "USE_TESTNET", True)

        client = Client(api_key, api_secret, testnet=use_testnet)
        return client

# PositionManager ve TradeExecutor bazı projelerde core altında değil, trading altında.
# Önce core'dan import etmeyi dene, olmazsa trading'den al.
try:
    from core.position_manager import PositionManager  # varsa buradan
except ModuleNotFoundError:
    from trading.position_manager import PositionManager  # yoksa buradan

try:
    from core.trade_executor import TradeExecutor  # varsa buradan
except ModuleNotFoundError:
    from trading.trade_executor import TradeExecutor  # yoksa buradan

# Global logger referansı (setup_logger() çağrıldıktan sonra dolacak)
system_logger: logging.Logger | None = None

# Loop bekleme süresi (saniye)
LOOP_SLEEP_SECONDS = 60


# ----------------------------------------------------------------------
# Basit feature engineering helper
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kendi içimizde minimal bir feature builder.
    Dıştaki data/feature_engineering bağımlılığını kaldırmak için kullanıyoruz.
    İstersen bunu daha sonra gerçek feature pipeline'ınla değiştirebilirsin.
    """
    df = raw_df.copy()

    # Zamanı index yap
    if "open_time" in df.columns:
        df.set_index("open_time", inplace=True)

    # Basit fiyat özellikleri
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change()
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # Basit hareketli ortalamalar
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()

    # Volatilite benzeri
    df["vol_10"] = df["return_1"].rolling(window=10).std()
    # Ek dummy feature (online model 22 feature bekliyor)
    df["dummy_extra"] = 0.0

    # NaN temizle
    df = df.dropna()

    # Hybrid / SGD: datetime kolonlarını numeric'e (ns -> int64) çevir
    try:
        for col in list(df.columns):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # pandas datetime64[ns] -> int64 (nanosecond)
                df[col] = df[col].astype("int64")
    except Exception:
        # Her ihtimale karşı sessiz geç; hata olursa HYBRID fallback zaten devreye girer
        pass

    return df.reset_index()


# ----------------------------------------------------------------------
# Binance klines fetch helper (async)
# ----------------------------------------------------------------------
async def fetch_klines(client, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Senkron python-binance client.get_klines çağrısını async hale getiren helper.
    """
    loop = asyncio.get_event_loop()

    def _fetch():
        return client.get_klines(symbol=symbol, interval=interval, limit=limit)

    klines = await loop.run_in_executor(None, _fetch)

    if not klines:
        return pd.DataFrame()

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

    # Zaman ve numerik cast
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


# ----------------------------------------------------------------------
# Trading objelerini oluştur
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Trading objelerini oluştur
# ----------------------------------------------------------------------
def create_trading_objects() -> Dict[str, Any]:
    global system_logger

    # Ortam değişkenleri
    env_vars, missing_vars = load_environment_variables()
    if missing_vars:
        logging.getLogger("system").warning(
            "[load_env] WARNING: Missing environment variables: %s",
            missing_vars,
        )

    # Logger
    setup_logger()
    system_logger = logging.getLogger("system")
    error_logger = logging.getLogger("error")

    # DRY_RUN uyarısı
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    if not api_key or not api_secret:
        system_logger.warning(
            "[BINANCE] API key/secret env'de yok. DRY_RUN modunda çalıştığından emin ol."
        )

    # Symbol & interval
    symbol_env = os.environ.get("SYMBOL")
    interval_env = os.environ.get("INTERVAL")

    SYMBOL = symbol_env or getattr(app_config, "SYMBOL", "BTCUSDT")
    INTERVAL = interval_env or getattr(app_config, "INTERVAL", "1m")

    # Training mode (sadece eğitim, gerçek trade yok)
    training_mode_env = os.environ.get("TRAINING_MODE", "").lower()
    TRAINING_MODE = training_mode_env in ("1", "true", "yes", "y", "on")

    if TRAINING_MODE:
        system_logger.info("[MAIN] TRAINING_MODE=true -> Sadece eğitim/log, gerçek trade YOK.")
    else:
        system_logger.info("[MAIN] TRAINING_MODE=false -> Normal çalışma modu (trade logic aktif).")

    # Hibrit mod flag (sadece karar verirken kullanıyoruz; modelin içinde ayrıca use_lstm_hybrid var)
    hybrid_mode_env = os.environ.get("HYBRID_MODE", "").lower()
    HYBRID_MODE = hybrid_mode_env in ("1", "true", "yes", "y", "on")

    if HYBRID_MODE:
        system_logger.info("[MAIN] HYBRID_MODE=true -> LSTM+SGD hibrit skor kullanılacak (mümkünse).")
    else:
        system_logger.info("[MAIN] HYBRID_MODE=false -> Sadece SGD skoru kullanılacak.")

    # Binance client
    binance_client = create_binance_client(app_config)

    # Online learner (SGD vb.)
    online_learner = OnlineLearner(
        model_dir="models",
        base_model_name="online_model",
        interval=INTERVAL,
        n_classes=2,
    )

    # Hibrit model (LSTM + SGD)
    # DİKKAT: hybrid_inference.HybridModel imzası sadece (model_dir, interval, logger) alıyor.
    hybrid_model = HybridModel(
        model_dir="models",
        interval=INTERVAL,
        logger=system_logger,
    )

    # ------------------------------------------------------------------
    # Offline meta'dan AUC -> model_confidence_factor
    # Önce OnlineLearner içinden almaya çalış, yoksa JSON dosyasına düş.
    # ------------------------------------------------------------------
    model_confidence_factor = 1.0
    best_auc = 0.6

    meta_source = {}
    try:
        if hasattr(online_learner, "offline_meta"):
            meta_source = getattr(online_learner, "offline_meta") or {}
        elif hasattr(online_learner, "meta"):
            meta_source = getattr(online_learner, "meta") or {}

        if isinstance(meta_source, dict) and meta_source:
            best_auc = float(meta_source.get("best_auc", best_auc))
            model_confidence_factor = max(0.5, min(1.5, (best_auc - 0.5) * 4 + 0.8))
            system_logger.info(
                "[RISK] Using offline meta from OnlineLearner for interval=%s: best_auc=%.4f, model_confidence_factor=%.3f",
                INTERVAL,
                best_auc,
                model_confidence_factor,
            )
        else:
            meta_path = os.path.join("models", f"model_meta_{INTERVAL}.json")
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                best_auc = float(meta.get("best_auc", best_auc))
                model_confidence_factor = max(0.5, min(1.5, (best_auc - 0.5) * 4 + 0.8))
                system_logger.info(
                    "[RISK] Loaded offline meta from %s: best_auc=%.4f, model_confidence_factor=%.3f",
                    meta_path,
                    best_auc,
                    model_confidence_factor,
                )
            except FileNotFoundError:
                system_logger.warning(
                    "[RISK] model_meta_%s.json bulunamadı, model_confidence_factor=1.0 kullanılacak.",
                    INTERVAL,
                )
            except Exception as e:
                error_logger.exception(
                    "[RISK] Offline meta okunurken hata: %s, model_confidence_factor=1.0 kullanılacak.",
                    e,
                )
    except Exception as e:
        error_logger.exception(
            "[RISK] Meta kaynağı okunurken hata: %s, model_confidence_factor=1.0 kullanılacak.",
            e,
        )

    # ------------------------------------------------------------------
    # Risk Manager
    # ------------------------------------------------------------------
    try:
        risk_manager = RiskManager(logger=system_logger)
    except TypeError:
        risk_manager = RiskManager()
        if hasattr(risk_manager, "logger"):
            risk_manager.logger = system_logger
        elif hasattr(risk_manager, "set_logger"):
            risk_manager.set_logger(system_logger)

    # AUC'tan gelen güven faktörünü risk manager'a ver
    if hasattr(risk_manager, "set_model_confidence_factor"):
        risk_manager.set_model_confidence_factor(model_confidence_factor)

    # ------------------------------------------------------------------
    # Position Manager
    # ------------------------------------------------------------------
    try:
        position_manager = PositionManager(logger=system_logger)
    except TypeError:
        position_manager = PositionManager()
        if hasattr(position_manager, "logger"):
            position_manager.logger = system_logger
        elif hasattr(position_manager, "set_logger"):
            position_manager.set_logger(system_logger)

    # ------------------------------------------------------------------
    # Trade Executor
    # ------------------------------------------------------------------
    trade_executor = TradeExecutor(
        client=binance_client,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    return {
        "SYMBOL": SYMBOL,
        "INTERVAL": INTERVAL,
        "TRAINING_MODE": TRAINING_MODE,
        "HYBRID_MODE": HYBRID_MODE,
        "binance_client": binance_client,
        "online_learner": online_learner,
        "hybrid_model": hybrid_model,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
    }


# ----------------------------------------------------------------------
# Ana bot loop
# ----------------------------------------------------------------------
async def bot_loop(trading_objects: Dict[str, Any]) -> None:
    system_logger = logging.getLogger("system")

    SYMBOL = trading_objects["SYMBOL"]
    INTERVAL = trading_objects["INTERVAL"]
    TRAINING_MODE = trading_objects.get("TRAINING_MODE", False)
    HYBRID_MODE = trading_objects.get("HYBRID_MODE", False)
    binance_client = trading_objects["binance_client"]
    online_learner = trading_objects["online_learner"]
    hybrid_model = trading_objects["hybrid_model"]
    risk_manager = trading_objects["risk_manager"]
    position_manager = trading_objects["position_manager"]
    trade_executor = trading_objects["trade_executor"]

    system_logger.info(
        "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s)",
        SYMBOL,
        INTERVAL,
        TRAINING_MODE,
    )

    while True:
        try:
            system_logger.info(
                "[DATA] Starting data pipeline for %s (%s, limit=%d)",
                SYMBOL,
                INTERVAL,
                500,
            )

            # 1) Klines çek
            raw_df = await fetch_klines(
                client=binance_client,
                symbol=SYMBOL,
                interval=INTERVAL,
                limit=500,
            )

            if raw_df.empty:
                system_logger.warning("[DATA] Empty klines received, skipping iteration.")
                await asyncio.sleep(LOOP_SLEEP_SECONDS)
                continue

            # 2) Feature engineering
            features_df = build_features(raw_df)

            if features_df.empty:
                system_logger.warning("[FE] Features DF empty, skipping iteration.")
                await asyncio.sleep(LOOP_SLEEP_SECONDS)
                continue

            system_logger.info(
                "[FE] Features DF shape: %s, columns=%s",
                features_df.shape,
                list(features_df.columns),
            )

            # Son satırı tahmin için kullan
            X_live = features_df.iloc[[-1]].copy()

            # 3) Online model (SGD) proba
            try:
                proba_sgd = online_learner.predict_proba(X_live)[0][1]
            except Exception as e:
                logging.getLogger("error").exception(
                    "[ONLINE] predict_proba sırasında hata: %s", e
                )
                await asyncio.sleep(LOOP_SLEEP_SECONDS)
                continue

            # 4) Hybrid model proba (LSTM + SGD)
            try:
                # Sadece HYBRID_MODE=True VE hybrid_model.use_lstm_hybrid=True ise
                # hibrit skor hesapla; aksi halde direkt SGD kullan.
                if HYBRID_MODE and getattr(hybrid_model, "use_lstm_hybrid", False):
                    # HybridModel: (p_hybrid_vec, debug_info) döner
                    p_hybrid_vec, hybrid_debug = hybrid_model.predict_proba(X_live)
                    proba_hybrid = float(p_hybrid_vec[0])

                    system_logger.info(
                        "[HYBRID] p_sgd_mean=%.4f, p_lstm_mean=%.4f, p_hybrid_mean=%.4f, best_auc=%.4f, best_side=%s",
                        hybrid_debug.get("p_sgd_mean", 0.0),
                        hybrid_debug.get("p_lstm_mean", 0.0),
                        hybrid_debug.get("p_hybrid_mean", 0.0),
                        hybrid_debug.get("best_auc", 0.0),
                        hybrid_debug.get("best_side", "n/a"),
                    )
                else:
                    # HYBRID_MODE=false veya LSTM hibrit devre dışı ise
                    # hibrit skor = online SGD skoru
                    proba_hybrid = proba_sgd
            except Exception as e:
                logging.getLogger("error").exception(
                    "[HYBRID] predict_proba sırasında hata: %s", e
                )
                # Hibrit hata verirse fallback olarak sgd kullan
                proba_hybrid = proba_sgd

            system_logger.info(
                "[PRED] p_sgd=%.4f, p_hybrid=%.4f (HYBRID_MODE=%s)", proba_sgd, proba_hybrid, HYBRID_MODE
            )

            # 5) Risk manager'dan model confidence factor al
            model_conf_factor = 1.0
            if hasattr(risk_manager, "get_model_confidence_factor"):
                model_conf_factor = risk_manager.get_model_confidence_factor()

            # Basit sinyal örneği: p_hybrid > 0.55 long, < 0.45 short, arası hold
            signal = "hold"
            if proba_hybrid > 0.55:
                signal = "long"
            elif proba_hybrid < 0.45:
                signal = "short"

            system_logger.info(
                "[SIGNAL] signal=%s, model_conf_factor=%.3f", signal, model_conf_factor
            )

            # 6) TradeExecutor varsa, sinyali ona pasla
            # TRAINING_MODE=true ise sadece log, trade yok.
            if TRAINING_MODE:
                system_logger.info(
                    "[MAIN] TRAINING_MODE aktif, execute_decision çağrılmayacak. Sinyal sadece loglandı."
                )
            else:
                if hasattr(trade_executor, "execute_decision"):
                    # Fiyat için son close
                    last_price = float(raw_df["close"].iloc[-1])

                    # Güvenli olması için p_sgd/p_hybrid/model_confidence_factor/best_auc/best_side değişkenlerini locals()'tan çek
                    p_sgd_val = float(locals().get("p_sgd", 0.5))
                    p_hybrid_val = float(locals().get("p_hybrid", 0.5))
                    mcf_val = float(locals().get("model_confidence_factor", 1.0))
                    best_auc_val = float(locals().get("best_auc", 0.0))
                    best_side_val = locals().get("best_side", "hold")

                    await trade_executor.execute_decision(
                        signal=signal,
                        symbol=SYMBOL,
                        price=last_price,
                        size=None,  # TODO: RiskManager içinden dinamik position size
                        interval=INTERVAL,
                        training_mode=TRAINING_MODE,
                        hybrid_mode=HYBRID_MODE,
                        probs={
                            "p_sgd": p_sgd_val,
                            "p_hybrid": p_hybrid_val,
                        },
                        extra={
                            "model_confidence_factor": mcf_val,
                            "best_auc": best_auc_val,
                            "best_side": best_side_val,
                        },
                    )
                else:
                    system_logger.info(
                        "[TRADE] TradeExecutor.execute_decision bulunamadı, trade atlanıyor."
                    )

        except asyncio.CancelledError:
            system_logger.info("[MAIN] bot_loop cancelled, shutting down...")
            break
        except Exception as e:
            logging.getLogger("error").exception(
                "[MAIN] Error in bot_loop: %s", e
            )

        await asyncio.sleep(LOOP_SLEEP_SECONDS)


# ----------------------------------------------------------------------
# main / async_main
# ----------------------------------------------------------------------
async def async_main() -> None:
    trading_objects = create_trading_objects()

    loop_task = asyncio.create_task(bot_loop(trading_objects))

    # Graceful shutdown
    loop = asyncio.get_running_loop()

    stop_event = asyncio.Event()

    def _signal_handler():
        logging.getLogger("system").info(
            "[MAIN] Shutdown signal received, cancelling bot_loop..."
        )
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows vb. ortamlar için
            pass

    await stop_event.wait()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        logging.getLogger("system").info("[MAIN] bot_loop cancelled.")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logging.getLogger("system").info("[MAIN] KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()

