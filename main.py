import asyncio
import logging
import os
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
    BinanceAPIException,
)


from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.labels import LabelGenerator
from data.batch_learning import BatchLearner
from data.online_learning import OnlineLearner
from data.anomaly_detection import AnomalyDetector


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
    Tek cycle için data pipeline:

      1) Binance'ten / cache'den ham veriyi çek
      2) Gerekirse anomaly detection uygula
      3) Feature engineering ile özellik dataframe'i üret
      4) (Label kolonları artık data/feature_engineering veya başka bir yerde üretilmiş varsayılıyor)
    """
    try:
        symbol = env_vars.get("SYMBOL", "BTCUSDT")
        interval = env_vars.get("INTERVAL", "1m")
        limit = _get_env_int(env_vars, "KLINES_LIMIT", 1000)

        LOGGER.info(
            "[DATA] Starting data pipeline for %s (%s, limit=%s)",
            symbol,
            interval,
            limit,
        )

        # --- 1) Data Loading ---
        data_loader = DataLoader(
            symbol=symbol,
            interval=interval,
            limit=limit,
            logger=LOGGER,
        )

        raw_df = await data_loader.load()
        if raw_df is None or len(raw_df) == 0:
            raise DataLoadingException("DataLoader returned empty dataframe.")

        LOGGER.info("[DATA] Raw DF shape: %s", raw_df.shape)

        # --- 2) (Opsiyonel) External Data Merge ---
        external_url = env_vars.get("EXTERNAL_DATA_URL")
        if external_url:
            try:
                raw_df = await data_loader.merge_external_data(raw_df, external_url)
                LOGGER.info("[DATA] After external merge DF shape: %s", raw_df.shape)
            except Exception as e:
                LOGGER.warning(
                    "[DATA] External data merge failed, continuing without it: %s",
                    e,
                    exc_info=True,
                )
        else:
            LOGGER.warning(
                "[DATA] No EXTERNAL_DATA_URL provided; skipping external data merge."
            )

        # --- 3) Anomaly Detection (opsiyonel) ---
        try:
            use_anomaly = env_vars.get("USE_ANOMALY_DETECTION", "false").lower() == "true"
            if use_anomaly:
                detector = AnomalyDetector(df=raw_df, logger=LOGGER)
                clean_df = detector.filter_anomalies()
                LOGGER.info(
                    "[DATA] Anomaly detection applied. Clean DF shape: %s",
                    clean_df.shape,
                )
            else:
                clean_df = raw_df
                LOGGER.info(
                    "[DATA] Anomaly detection disabled. Using raw DF shape: %s",
                    clean_df.shape,
                )
        except Exception as e:
            LOGGER.warning(
                "[DATA] Anomaly detection failed, using raw data: %s", e, exc_info=True
            )
            clean_df = raw_df

        # --- 4) Feature Engineering ---
        try:
            feature_engineer = FeatureEngineer(df=clean_df, logger=LOGGER)
            features_df = feature_engineer.transform()

            # Özellik dataframe'ini logla (şekil + kolonlar)
            LOGGER.info(
                "[FE] Features DF shape: %s, columns=%s",
                features_df.shape,
                list(features_df.columns),
            )
        except Exception as e:
            LOGGER.error("[FE] Feature engineering failed: %s", e, exc_info=True)
            raise FeatureEngineeringException(f"Feature engineering failed: {e}") from e

        # NOT: Eskiden burada LabelGenerator vardı:
        # from data.labels import LabelGenerator
        # label_generator = LabelGenerator(df=features_df, horizon=label_horizon, logger=LOGGER)
        # features_with_labels = label_generator.generate()
        #
        # Artık label kolonlarının (ör: 'label', 'future_return') feature pipeline içinde
        # üretildiğini varsayıyoruz. Bu yüzden features_df doğrudan geri dönüyor.

        if len(features_df) == 0:
            raise DataProcessingException("Features dataframe is empty after pipeline.")

        return features_df

    except BinanceBotException:
        # Zaten custom exception; yeniden fırlat
        raise
    except Exception as e:
        LOGGER.error("💥 [PIPELINE] Unexpected error in data pipeline: %s", e, exc_info=True)
        raise DataProcessingException(f"Data pipeline failed: {e}") from e



# ---------------------------------------------------------
# MODEL TRAINING (Batch + Online)
# ---------------------------------------------------------

def _split_features_labels(clean_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    clean_df içinden feature kolonları ve label kolonunu ayırır.
    Assumption:
      - Label kolonu: 'label'
    """
    if "label" not in clean_df.columns:
        raise ModelTrainingException("Column 'label' not found in dataframe.")

    label_col = "label"

    # Feature kolonlarını: tüm sayısal kolonlar - label
    numeric_cols = clean_df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [label_col]]

    if not feature_cols:
        raise ModelTrainingException("No feature columns found for training.")

    X = clean_df[feature_cols].copy()
    y = clean_df[label_col].copy()

    return X, y, feature_cols


def train_models_and_update_state(clean_df: pd.DataFrame, env_vars: Dict[str, str]) -> None:
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
        batch_model = batch_learner.fit()
        # batch_model RAM'de, ayrıca models/batch_model.joblib olarak kaydedilmeli

        # --- Online Learner ---
        LOGGER.info("[ONLINE] Initializing OnlineLearner with batch data.")

        online_learner = OnlineLearner(
            model_dir=model_dir,
            base_model_name=online_model_name,
            n_classes=2,
            logger=LOGGER,
        )

        # İlk eğitim (tüm batch)
        online_learner.initial_fit(X, y)

        # İsteğe bağlı: son N örnekle incremental update (örneğin son 100 bar)
        tail_n = min(100, len(clean_df))
        if tail_n > 0:
            X_tail = X.iloc[-tail_n:]
            y_tail = y.iloc[-tail_n:]
            online_learner.partial_update(X_tail, y_tail)

    except (DataProcessingException, ModelTrainingException, OnlineLearningException) as e:
        LOGGER.error("💥 [MODEL] Known model pipeline error: %s", e, exc_info=True)
        raise
    except Exception as e:
        LOGGER.error("💥 [MODEL] Unexpected error in train_models_and_update_state: %s", e, exc_info=True)
        raise ModelTrainingException(f"Unexpected training error: {e}") from e


# ---------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------

def generate_signal(clean_df: pd.DataFrame, env_vars: Dict[str, str]) -> None:
    """
    - Son bar için feature'ları alır
    - Online modelden BUY olasılığını (class=1) hesaplar
    - BUY / SELL / HOLD kararı verip loglar
    """
    try:
        if len(clean_df) == 0:
            LOGGER.warning("[SIGNAL] Empty dataframe, cannot generate signal.")
            return

        X, y, feature_cols = _split_features_labels(clean_df)

        # Sadece son bar için feature (shape: (1, n_features))
        X_live = X.iloc[[-1]]

        model_dir = env_vars.get("MODEL_DIR", "models")
        online_model_name = env_vars.get("ONLINE_MODEL_NAME", "online_model")

        BUY_THRESHOLD = _get_env_float(env_vars, "BUY_THRESHOLD", 0.6)
        SELL_THRESHOLD = _get_env_float(env_vars, "SELL_THRESHOLD", 0.4)

        # OnlineLearner mevcut modeli diskten yükleyecek
        online_learner = OnlineLearner(
            model_dir=model_dir,
            base_model_name=online_model_name,
            n_classes=2,
            logger=LOGGER,
        )

        # Feature kolon hizalaması
        online_learner.feature_columns = feature_cols

        # Tek bir scalar BUY olasılığı (class=1)
        p_buy = online_learner.predict_proba_live(X_live)

        LOGGER.info(
            "[SIGNAL] p_buy=%.4f (BUY_THRESHOLD=%.2f, SELL_THRESHOLD=%.2f)",
            p_buy,
            BUY_THRESHOLD,
            SELL_THRESHOLD,
        )

        if p_buy >= BUY_THRESHOLD:
            signal = "BUY"
        elif p_buy <= SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        LOGGER.info("[SIGNAL] Generated trading signal: %s", signal)

    except Exception as e:
        LOGGER.error("💥 [SIGNAL] Error while generating signal: %s", e, exc_info=True)
        raise SignalGenerationException(f"Signal generation failed: {e}") from e



# ---------------------------------------------------------
# BOT LOOP
# ---------------------------------------------------------

async def run_data_and_model_pipeline(env_vars: Dict[str, str]) -> None:
    """
    Tek bir cycle:
      - Data pipeline
      - Model training (batch + online)
      - Signal generation
    """
    clean_df = await run_data_pipeline(env_vars)
    train_models_and_update_state(clean_df, env_vars)
    generate_signal(clean_df, env_vars)


async def bot_loop(env_vars: Dict[str, str]) -> None:
    """
    Arka planda sürekli çalışan ana bot loop'u.
    """
    LOGGER.info("🚀 [BOT] Binance1-Pro core bot_loop started.")
    interval_sec = _get_env_int(env_vars, "BOT_LOOP_INTERVAL", 60)

    while True:
        try:
            await run_data_and_model_pipeline(env_vars)
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
    try:
        env_vars = load_environment_variables()
    except Exception as e:
        LOGGER.error("💥 [MAIN] Failed to load environment variables: %s", e, exc_info=True)
        raise EnvironmentException(f"Failed to load environment variables: {e}") from e

    app = create_app(env_vars)

    # Graceful shutdown için sinyal yakalama
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(_shutdown(loop, s)),
        )

    web.run_app(app, host="0.0.0.0", port=8080)


async def _shutdown(loop: asyncio.AbstractEventLoop, sig: signal.Signals) -> None:
    LOGGER.info("[MAIN] Received exit signal %s, shutting down...", sig.name)
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks)
    loop.stop()


if __name__ == "__main__":
    main()

