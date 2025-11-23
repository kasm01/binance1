import asyncio
import signal
from contextlib import suppress
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from aiohttp import web

from config.load_env import load_environment_variables

from core.logger import setup_logger
from core.exceptions import (
    BinanceBotException,
    ConfigException,
    DataLoadingException,
    DataProcessingException,
    ModelTrainingException,
    OnlineLearningException,
    PredictionException,
    SignalGenerationException,
    PipelineException,
    LabelGenerationException,
)

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.batch_learning import BatchLearner
from data.online_learning import OnlineLearner
from data.anomaly_detection import AnomalyDetector

from trading.strategy_engine import StrategyEngine
from trading.risk_manager import RiskManager
from trading.capital_manager import CapitalManager
from trading.position_manager import PositionManager
from trading.trade_executor import TradeExecutor

# ---------------------------------------------------------
# Global logger
# ---------------------------------------------------------

LOGGER = setup_logger("system")

# ---------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------


def _get_env_int(env_vars: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(env_vars.get(key, str(default)))
    except Exception:
        return default


def _get_env_float(env_vars: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(env_vars.get(key, str(default)))
    except Exception:
        return default


# ---------------------------------------------------------
# DATA PIPELINE
# ---------------------------------------------------------


async def run_data_pipeline(env_vars: Dict[str, str]) -> pd.DataFrame:
    """
    Tam veri pipeline:
      1) Kline verisini DataLoader ile çek
      2) FeatureEngineer ile feature üret
      3) AnomalyDetector ile outlier temizliği
      4) 'label' kolonunu üret ve son satırı düşür
    """
    symbol = env_vars.get("SYMBOL", "BTCUSDT")
    interval = env_vars.get("INTERVAL", "1m")
    limit = _get_env_int(env_vars, "LIMIT", 1000)

    LOGGER.info(
        "[DATA] Starting data pipeline for %s (%s, limit=%d)",
        symbol,
        interval,
        limit,
    )

    try:
        # 1) DataLoader
        data_loader = DataLoader(env_vars)

        candidate_methods = (
            "load_and_cache_klines",
            "load_and_cache_ohlcv",
            "load_klines",
            "load_and_cache",
            "load",
        )
        load_method = None

        for name in candidate_methods:
            if hasattr(data_loader, name):
                load_method = getattr(data_loader, name)
                LOGGER.info("[DATA] Using DataLoader.%s", name)
                break

        if load_method is None:
            raise DataLoadingException(
                "DataLoader has no suitable load method; tried: "
                + ", ".join(candidate_methods)
            )

        # 2) Data yükle
        code_vars = load_method.__code__.co_varnames
        kwargs: Dict[str, Any] = {}

        if "symbol" in code_vars:
            kwargs["symbol"] = symbol
        if "interval" in code_vars:
            kwargs["interval"] = interval
        if "limit" in code_vars:
            kwargs["limit"] = limit

        raw_df = load_method(**kwargs)

        if raw_df is None or raw_df.empty:
            raise DataLoadingException("No data returned from DataLoader.")

        LOGGER.info("[DATA] Raw DF shape: %s", raw_df.shape)

        # 3) Feature engineering
        fe = FeatureEngineer(df=raw_df)
        features_df = fe.transform()

        LOGGER.info(
            "[FE] Features DF shape: %s, columns=%s",
            features_df.shape,
            list(features_df.columns),
        )

        # 4) Anomali tespiti (IsolationForest vs.)
        try:
            anomaly_detector = AnomalyDetector(features_df)
            features_df = anomaly_detector.detect_and_handle_anomalies()
        except Exception as e:
            LOGGER.warning(
                "[DATA] Anomaly detection failed, using original features_df: %r",
                e,
                exc_info=True,
            )

        # 5) Label üretimi
        if "close" not in features_df.columns:
            raise LabelGenerationException(
                "Column 'close' not found in features_df for label generation."
            )

        df = features_df.copy()

        # Basit label: Bir sonraki barın kapanışı şimdiki kapanıştan yüksekse => 1 (BUY), değilse 0 (SELL/HOLD)
        df["future_close"] = df["close"].shift(-1)
        df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

        df["label"] = (df["future_return"] > 0).astype(int)

        # Son satırın future_close'u NaN olacağı için düşür
        df = df.dropna(subset=["future_close"]).copy()
        df = df.drop(columns=["future_close", "future_return"])

        LOGGER.info(
            "[FE] Final Features DF shape (with label): %s, columns=%s",
            df.shape,
            list(df.columns),
        )

        if df.empty:
            raise DataProcessingException("Final features dataframe is empty after label generation.")

        return df

    except BinanceBotException:
        # Zaten bizden atanmış custom exception, aynen yukarı fırlat
        raise
    except Exception as e:
        LOGGER.error(
            "💥 [PIPELINE] Unexpected error in data pipeline: %s",
            e,
            exc_info=True,
        )
        raise DataProcessingException(f"Data pipeline failed: {e}") from e


# ---------------------------------------------------------
# MODEL TRAINING (Batch + Online)
# ---------------------------------------------------------


def _split_features_labels(
    clean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    clean_df içinden feature kolonları ve label kolonunu ayırır.
    Assumption:
      - Label kolonu: 'label'
    """
    if "label" not in clean_df.columns:
        raise ModelTrainingException("Column 'label' not found in dataframe.")

    label_col = "label"

    numeric_cols = clean_df.select_dtypes(
        include=["float64", "float32", "int64", "int32"]
    ).columns.tolist()

    feature_cols = [c for c in numeric_cols if c != label_col]

    if not feature_cols:
        raise ModelTrainingException("No feature columns found for training.")

    X = clean_df[feature_cols].copy()
    y = clean_df[label_col].copy()

    return X, y, feature_cols


def train_models_and_update_state(
    clean_df: pd.DataFrame, env_vars: Dict[str, str]
) -> None:
    """
    - BatchLearner ile batch model eğitir
    - OnlineLearner ile initial_fit + partial_update yapar
    """
    try:
        X, y, feature_cols = _split_features_labels(clean_df)
        n_samples, n_features = X.shape

        LOGGER.info(
            "[MODEL] Training batch model on %d samples, %d features. Using %d feature columns.",
            n_samples,
            n_features,
            len(feature_cols),
        )

        model_dir = env_vars.get("MODEL_DIR", "models")
        batch_model_name = env_vars.get("BATCH_MODEL_NAME", "batch_model")
        online_model_name = env_vars.get("ONLINE_MODEL_NAME", "online_model")

        # --- Batch Learner ---
        batch_learner = BatchLearner(
            X=X,
            y=y,
            model_dir=model_dir,
            base_model_name=batch_model_name,
            logger=LOGGER,
        )
        _ = batch_learner.fit()

        # --- Online Learner ---
        LOGGER.info("[ONLINE] Initializing OnlineLearner with batch data.")

        online_learner = OnlineLearner(
            model_dir=model_dir,
            base_model_name=online_model_name,
            n_classes=2,
            logger=LOGGER,
        )

        # İlk eğitim (tüm batch)
        online_learner.feature_columns = feature_cols
        online_learner.initial_fit(X, y)

        # Son N örnekle incremental update (örneğin son 100 bar)
        tail_n = min(100, len(clean_df))
        if tail_n > 0:
            X_tail = X.iloc[-tail_n:]
            y_tail = y.iloc[-tail_n:]
            online_learner.partial_update(X_tail, y_tail)

    except (DataProcessingException, ModelTrainingException, OnlineLearningException) as e:
        LOGGER.error("💥 [MODEL] Known model pipeline error: %s", e, exc_info=True)
        raise
    except Exception as e:
        LOGGER.error(
            "💥 [MODEL] Unexpected error in train_models_and_update_state: %s",
            e,
            exc_info=True,
        )
        raise ModelTrainingException(f"Unexpected training error: {e}") from e


# ---------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------


def generate_signal(clean_df: pd.DataFrame, env_vars: Dict[str, str]) -> str:
    """
    - Son bar için feature'ları alır
    - Online modelden BUY olasılığını (class=1) hesaplar
    - BUY / SELL / HOLD kararı verip döner
    """
    try:
        if len(clean_df) == 0:
            LOGGER.warning("[SIGNAL] Empty dataframe, cannot generate signal.")
            return "HOLD"

        X, y, feature_cols = _split_features_labels(clean_df)

        # Sadece son bar için feature
        X_live = X.iloc[[-1]]

        model_dir = env_vars.get("MODEL_DIR", "models")
        online_model_name = env_vars.get("ONLINE_MODEL_NAME", "online_model")

        buy_threshold = _get_env_float(env_vars, "BUY_THRESHOLD", 0.6)
        sell_threshold = _get_env_float(env_vars, "SELL_THRESHOLD", 0.4)

        online_learner = OnlineLearner(
            model_dir=model_dir,
            base_model_name=online_model_name,
            n_classes=2,
            logger=LOGGER,
        )

        online_learner.feature_columns = feature_cols

        p_buy = online_learner.predict_proba_live(X_live)

        LOGGER.info(
            "[SIGNAL] p_buy=%.4f (BUY_THRESHOLD=%.2f, SELL_THRESHOLD=%.2f)",
            p_buy,
            buy_threshold,
            sell_threshold,
        )

        if p_buy >= buy_threshold:
            signal = "BUY"
        elif p_buy <= sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        LOGGER.info("[SIGNAL] Generated trading signal: %s", signal)
        return signal

    except Exception as e:
        LOGGER.error(
            "💥 [SIGNAL] Error while generating signal: %s",
            e,
            exc_info=True,
        )
        raise SignalGenerationException(f"Signal generation failed: {e}") from e


# ---------------------------------------------------------
# TRADING OBJE AYARLARI
# ---------------------------------------------------------


def init_trading_objects(env_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Trading ile ilgili objeleri bir kere oluşturur:
      - CapitalManager
      - RiskManager
      - PositionManager
      - TradeExecutor
      - StrategyEngine (model tabanlı strateji)
    """
    total_capital = float(env_vars.get("TOTAL_CAPITAL", 1000.0))
    max_risk_per_trade = float(env_vars.get("MAX_RISK_PER_TRADE", 0.02))

    capital_manager = CapitalManager(total_capital=total_capital)
    risk_manager = RiskManager(max_risk_per_trade=max_risk_per_trade)
    position_manager = PositionManager()
    trade_executor = TradeExecutor()
    strategy_engine = StrategyEngine()

    LOGGER.info(
        "[TRADING_INIT] total_capital=%.4f, max_risk_per_trade=%.4f",
        total_capital,
        max_risk_per_trade,
    )

    return {
        "capital_manager": capital_manager,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "strategy_engine": strategy_engine,
    }


# ---------------------------------------------------------
# TRADING: LONG / SHORT + POZİSYON YÖNETİMİ
# ---------------------------------------------------------


def handle_trading(
    model_signal: str,
    latest_row: pd.Series,
    env_vars: Dict[str, str],
    trading_objects: Dict[str, Any],
) -> None:
    """
    BUY/SELL/HOLD sinyaline göre LONG/SHORT pozisyon yönetimi.

    Anlam haritası:
      - BUY  -> LONG tarafına geç
        * SHORT pozisyon varsa önce kapat, sonra LONG aç
      - SELL -> SHORT tarafına geç
        * LONG pozisyon varsa önce kapat, sonra SHORT aç
      - HOLD -> hiçbir pozisyon değişikliği yapma
    """
    symbol = env_vars.get("SYMBOL", "BTCUSDT")
    price = float(latest_row["close"])

    capital_manager: CapitalManager = trading_objects["capital_manager"]
    risk_manager: RiskManager = trading_objects["risk_manager"]
    position_manager: PositionManager = trading_objects["position_manager"]
    trade_executor: TradeExecutor = trading_objects["trade_executor"]

    open_positions = position_manager.get_open_positions()

    # Açık pozisyonların snapshot'ını logla
    if open_positions:
        LOGGER.info("[TRADE] Open positions snapshot: %s", open_positions)
    else:
        LOGGER.info("[TRADE] No open positions currently.")

    # HOLD -> hiçbir şey yapma
    if model_signal == "HOLD":
        LOGGER.info("[TRADE] HOLD signal received, no action taken.")
        return

    # Pozisyonları side'a göre ayır
    long_positions = {
        pid: pos for pid, pos in open_positions.items()
        if str(pos.get("side", "LONG")).upper() == "LONG"
    }
    short_positions = {
        pid: pos for pid, pos in open_positions.items()
        if str(pos.get("side", "SHORT")).upper() == "SHORT"
    }

    # Yardımcı: belirli side'daki pozisyonları kapat
    def close_side_positions(side: str, positions: Dict[str, Dict[str, Any]]) -> None:
        for pos_id, pos in list(positions.items()):
            qty = float(pos["qty"])
            LOGGER.info(
                "[TRADE] Closing %s position %s | symbol=%s, qty=%.6f",
                side, pos_id, symbol, qty,
            )
            try:
                _ = trade_executor.close_position(
                    symbol=symbol,
                    quantity=qty,
                    position_side=side.upper(),
                )
            except Exception as e:
                LOGGER.error(
                    "[TRADE] Error while sending close order for %s: %s",
                    pos_id,
                    e,
                    exc_info=True,
                )

            # PnL hesapla ve sermayeyi serbest bırak
            pnl = position_manager.close_position(pos_id, exit_price=price)
            capital_to_release = qty * price
            capital_manager.release(capital_to_release)

            LOGGER.info(
                "[TRADE] %s position %s closed. PnL=%.4f, released_capital=%.4f",
                side,
                pos_id,
                pnl if pnl is not None else 0.0,
                capital_to_release,
            )

    # BUY -> LONG'a geç (SHORT varsa kapat)
    if model_signal == "BUY":
        if long_positions:
            LOGGER.info(
                "[TRADE] BUY signal but LONG positions already open, skipping new LONG entry."
            )
            return

        if short_positions:
            LOGGER.info(
                "[TRADE] BUY signal -> closing existing SHORT positions before opening LONG."
            )
            close_side_positions("SHORT", short_positions)

        # Yeni LONG aç
        risk_pct = float(
            env_vars.get("RISK_PER_TRADE", env_vars.get("MAX_RISK_PER_TRADE", 0.01))
        )
        available_capital = capital_manager.available_capital
        capital_to_use = capital_manager.allocate(risk_pct)

        if capital_to_use <= 0:
            LOGGER.warning("[TRADE] capital_to_use <= 0 for LONG, skipping trade.")
            return

        # Pozisyon nominal büyüklüğü: kullanılan sermaye
        position_notional = capital_to_use

        # Risk kontrolü (toplam sermayeye göre)
        if not risk_manager.check_risk(
            capital=capital_manager.total_capital,
            position_qty=position_notional,
        ):
            LOGGER.warning(
                "[TRADE] Risk limit exceeded for LONG, releasing capital and skipping trade."
            )
            capital_manager.release(capital_to_use)
            return

        qty = capital_to_use / price

        LOGGER.info(
            "[TRADE] Opening LONG position | symbol=%s, qty=%.6f, price=%.2f, notional=%.2f",
            symbol,
            qty,
            price,
            position_notional,
        )

        try:
            order = trade_executor.create_market_order(
                symbol=symbol,
                side="BUY",
                quantity=qty,
                position_side="LONG",
            )
        except Exception as e:
            LOGGER.error(
                "[TRADE] Error while creating LONG order: %s", e, exc_info=True
            )
            capital_manager.release(capital_to_use)
            return

        if order:
            pos_id = position_manager.open_position(
                symbol=symbol,
                qty=qty,
                side="LONG",
                price=price,
            )
            LOGGER.info(
                "[TRADE] LONG position opened with id=%s (orderId=%s)",
                pos_id,
                order.get("orderId"),
            )
        else:
            LOGGER.error("[TRADE] LONG order returned empty result, releasing capital.")
            capital_manager.release(capital_to_use)

        return

    # SELL -> SHORT'a geç (LONG varsa kapat)
    if model_signal == "SELL":
        if short_positions:
            LOGGER.info(
                "[TRADE] SELL signal but SHORT positions already open, skipping new SHORT entry."
            )
            return

        if long_positions:
            LOGGER.info(
                "[TRADE] SELL signal -> closing existing LONG positions before opening SHORT."
            )
            close_side_positions("LONG", long_positions)

        # Yeni SHORT aç
        risk_pct = float(
            env_vars.get("RISK_PER_TRADE", env_vars.get("MAX_RISK_PER_TRADE", 0.01))
        )
        available_capital = capital_manager.available_capital
        capital_to_use = capital_manager.allocate(risk_pct)

        if capital_to_use <= 0:
            LOGGER.warning("[TRADE] capital_to_use <= 0 for SHORT, skipping trade.")
            return

        position_notional = capital_to_use

        if not risk_manager.check_risk(
            capital=capital_manager.total_capital,
            position_qty=position_notional,
        ):
            LOGGER.warning(
                "[TRADE] Risk limit exceeded for SHORT, releasing capital and skipping trade."
            )
            capital_manager.release(capital_to_use)
            return

        qty = capital_to_use / price

        LOGGER.info(
            "[TRADE] Opening SHORT position | symbol=%s, qty=%.6f, price=%.2f, notional=%.2f",
            symbol,
            qty,
            price,
            position_notional,
        )

        try:
            order = trade_executor.create_market_order(
                symbol=symbol,
                side="SELL",
                quantity=qty,
                position_side="SHORT",
            )
        except Exception as e:
            LOGGER.error(
                "[TRADE] Error while creating SHORT order: %s", e, exc_info=True
            )
            capital_manager.release(capital_to_use)
            return

        if order:
            pos_id = position_manager.open_position(
                symbol=symbol,
                qty=qty,
                side="SHORT",
                price=price,
            )
            LOGGER.info(
                "[TRADE] SHORT position opened with id=%s (orderId=%s)",
                pos_id,
                order.get("orderId"),
            )
        else:
            LOGGER.error(
                "[TRADE] SHORT order returned empty result, releasing capital."
            )
            capital_manager.release(capital_to_use)

        return


# ---------------------------------------------------------
# BOT LOOP
# ---------------------------------------------------------


async def run_data_and_model_pipeline(
    env_vars: Dict[str, str],
    trading_objects: Dict[str, Any],
) -> None:
    """
    Tek bir cycle:
      - Data pipeline
      - Model training (batch + online)
      - Signal generation
      - Sinyale göre LONG/SHORT/HOLD trade yönetimi
    """
    clean_df = await run_data_pipeline(env_vars)

    # Model eğit
    train_models_and_update_state(clean_df, env_vars)

    # Sinyal üret (online model)
    signal = generate_signal(clean_df, env_vars)

    # En son satırı trading için kullan (fiyat vs.)
    latest_row = clean_df.iloc[-1]

    # Trading akışını çalıştır
    handle_trading(
        model_signal=signal,
        latest_row=latest_row,
        env_vars=env_vars,
        trading_objects=trading_objects,
    )


async def bot_loop(env_vars: Dict[str, str]) -> None:
    """
    Arka planda sürekli çalışan ana bot loop'u.
    """
    LOGGER.info("🚀 [BOT] Binance1-Pro core bot_loop started.")
    interval_sec = _get_env_int(env_vars, "BOT_LOOP_INTERVAL", 60)

    # Trading objelerini bir kere oluştur
    trading_objects = init_trading_objects(env_vars)

    while True:
        try:
            await run_data_and_model_pipeline(env_vars, trading_objects)
        except Exception as e:
            LOGGER.error("💥 [BOT] Unexpected error in bot_loop: %s", e, exc_info=True)
        finally:
            await asyncio.sleep(interval_sec)


# ---------------------------------------------------------
# Aiohttp HTTP Server
# ---------------------------------------------------------

routes = web.RouteTableDef()


@routes.get("/")
async def root(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "status": "ok",
            "message": "Binance1-Pro bot is running.",
        }
    )


@routes.get("/health")
async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy"})


def create_app(env_vars: Dict[str, str]) -> web.Application:
    app = web.Application()
    app["env_vars"] = env_vars
    app.add_routes(routes)

    async def on_startup(app: web.Application):
        LOGGER.info(
            "🌐 [MAIN] Starting HTTP server on 0.0.0.0:8080 (ENV=%s)",
            env_vars.get("ENVIRONMENT", "unknown"),
        )
        LOGGER.info("🔁 [MAIN] Starting background bot_loop task...")
        app["bot_task"] = asyncio.create_task(bot_loop(app["env_vars"]))

    async def on_cleanup(app: web.Application):
        LOGGER.info("[MAIN] Cleanup: cancelling bot_loop task...")
        bot_task = app.get("bot_task")
        if bot_task:
            bot_task.cancel()
            with suppress(asyncio.CancelledError):
                await bot_task

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


def main() -> None:
    # Environment değişkenlerini yükle
    try:
        env_vars = load_environment_variables()
    except Exception as e:
        LOGGER.error(
            "💥 [MAIN] Failed to load environment variables: %s", e, exc_info=True
        )
        raise ConfigException(f"Failed to load environment variables: {e}") from e

    app = create_app(env_vars)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_shutdown(loop, s)),
        )

    web.run_app(app, host="0.0.0.0", port=8080)


async def _shutdown(loop: asyncio.AbstractEventLoop, sig: signal.Signals) -> None:
    LOGGER.info("[MAIN] Received exit signal %s, shutting down...", sig.name)
    tasks = [
        t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()
    ]
    for task in tasks:
        task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks)
    loop.stop()


if __name__ == "__main__":
    main()

