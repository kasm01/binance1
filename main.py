import asyncio
import logging
import os
import signal
from typing import Dict, List

import numpy as np
import pandas as pd
from aiohttp import web
from joblib import load as joblib_load

from config.load_env import load_environment_variables
from core.logger import setup_logger
from core.exceptions import DataProcessingException, ModelTrainingException
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.batch_learning import BatchLearner
from data.online_learning import OnlineLearner

# -----------------------------------------------------------------------------
# Global config & logger
# -----------------------------------------------------------------------------

ENV_VARS: Dict[str, str] = load_environment_variables()
logger = setup_logger(logger_name="system")


# -----------------------------------------------------------------------------
# HTTP Handlers (Cloud Run health / root)
# -----------------------------------------------------------------------------

async def handle_root(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "message": "Binance1-Pro bot running"})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "healthy"})


async def init_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/healthz", handle_health)
    return app


# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

async def run_data_pipeline(env_vars: Dict[str, str]) -> pd.DataFrame:
    """
    1) Binance'ten ham veriyi çeker
    2) (Varsa) harici veri ile merge eder
    3) Feature engineering uygular
    4) Label ve future_return üretir
    """
    symbol = env_vars.get("SYMBOL", "BTCUSDT")
    interval = env_vars.get("INTERVAL", "1m")
    limit = int(env_vars.get("HISTORY_LIMIT", "1000"))

    data_loader = DataLoader(env_vars=env_vars, logger=logger)

    # --- 1) Binance verisi ---
    logger.info("[DATA] Fetching %s klines from Binance for %s (%s)", limit, symbol, interval)
    raw_df = data_loader.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)
    if raw_df is None or raw_df.empty:
        raise DataProcessingException("[DATA] Binance returned empty dataframe.")
    logger.info("[DATA] Raw DF shape: %s", raw_df.shape)

    # --- 2) Harici veri (opsiyonel) ---
    external_url = env_vars.get("EXTERNAL_DATA_URL")
    if external_url:
        try:
            # DataLoader içindeki imzaya göre bu fonksiyonu düzenledik;
            # burada sadece URL zorunlu, diğerleri DataLoader içinde env'den okunuyor.
            ext_df = data_loader.fetch_external_data(url=external_url)
            # İstersen burada raw_df ile merge mantığını ekleyebilirsin.
            # Şimdilik sadece log atıp devam edelim:
            logger.info("[DATA] External data fetched, shape: %s", getattr(ext_df, "shape", None))
        except Exception as e:
            logger.error(
                "[DATA] Error in fetch_external_data, continuing with Binance data only: %s",
                e,
                exc_info=True,
            )
    else:
        logger.warning("[DATA] No EXTERNAL_DATA_URL provided; skipping external data merge.")

    # --- 3) Feature engineering ---
    try:
        feature_engineer = FeatureEngineer(df=raw_df, logger=logger)
        features_df = feature_engineer.transform()
        if features_df is None or features_df.empty:
            raise DataProcessingException("[FE] FeatureEngineer returned empty dataframe.")
        logger.info("[FE] Features DF shape: %s", features_df.shape)
    except DataProcessingException:
        # FeatureEngineer zaten DataProcessingException raise ediyorsa aynen yukarı fırlatalım
        raise
    except Exception as e:
        logger.error("[FE] Unexpected error in FeatureEngineer.transform(): %s", e, exc_info=True)
        raise DataProcessingException(f"Feature engineering failed: {e}") from e

    # --- 4) Label & future_return ---
    df = features_df.copy()

    # Close'un numerik olduğundan emin ol
    if not np.issubdtype(df["close"].dtype, np.number):
        df["close"] = pd.to_numeric(df["close"], errors="coerce")

    horizon = int(env_vars.get("LABEL_HORIZON", "10"))  # kaç bar sonrasına bakarak future_return
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

    # future_return'i hesaplayamayan satırları at (tail kısmı)
    df = df.dropna(subset=["future_return"])

    # Label: gelecekteki getirisi pozitifse 1, değilse 0
    df["label"] = (df["future_return"] > 0).astype(int)

    n = len(df)
    pos = int(df["label"].sum())
    neg = n - pos
    mean_fr = df["future_return"].mean()
    std_fr = df["future_return"].std()
    pos_ratio = pos / n if n > 0 else 0.0

    logger.info(
        "[LABEL] future_return mean=%.4f, std=%.4f, positive ratio=%.3f (%.1f%%), pos=%d, neg=%d, n=%d",
        mean_fr,
        std_fr,
        pos_ratio,
        pos_ratio * 100,
        pos,
        neg,
        n,
    )

    return df


# -----------------------------------------------------------------------------
# Model training & state update
# -----------------------------------------------------------------------------

def train_models_and_update_state(clean_df: pd.DataFrame, env_vars: Dict[str, str]) -> None:
    """
    Batch ve online modelleri eğitir, model dosyalarını kaydeder.
    """
    # Özellik kolonları: label & future_return hariç her şey
    feature_columns: List[str] = [
        c for c in clean_df.columns if c not in ("future_return", "label")
    ]

    X = clean_df[feature_columns]
    y = clean_df["label"]

    n_samples, n_features = X.shape
    logger.info(
        "[MODEL] Training batch model on %d samples, %d features. Using %d feature columns.",
        n_samples,
        n_features,
        len(feature_columns),
    )

    model_dir = env_vars.get("MODEL_DIR", "models")
    batch_model_name = env_vars.get("BATCH_MODEL_NAME", "batch_model")
    online_model_name = env_vars.get("ONLINE_MODEL_NAME", "online_model")

    os.makedirs(model_dir, exist_ok=True)

    # --- Batch model ---
    try:
        batch_learner = BatchLearner(
            X=X,
            y=y,
            model_dir=model_dir,
            base_model_name=batch_model_name,
            logger=logger,
        )
        batch_model = batch_learner.fit()
    except ModelTrainingException:
        raise
    except Exception as e:
        logger.error("[BATCH] Unexpected error while training batch model: %s", e, exc_info=True)
        raise ModelTrainingException(f"Batch model training failed: {e}") from e

    # --- Online model ---
    try:
        logger.info("[ONLINE] Initializing OnlineLearner with batch data.")
        n_classes = len(np.unique(y.values))
        online_learner = OnlineLearner(
            model_dir=model_dir,
            base_model_name=online_model_name,
            n_classes=n_classes,
            logger=logger,
        )
        # initial_fit içinde:
        #  - feature_columns set ediliyor,
        #  - partial_fit yapılıyor,
        #  - model dosyası kaydediliyor.
        online_learner.initial_fit(X, y)
    except Exception as e:
        logger.error("[ONLINE] Unexpected error while initializing OnlineLearner: %s", e, exc_info=True)
        # Online öğrenme başarısız olsa bile, batch model kaydedilmiş durumda;
        # pipeline'ın tamamen çökmesini istemiyorsak burada fatal yapmayabiliriz.
        # Ama şimdilik yukarı fırlatalım:
        raise ModelTrainingException(f"Online model initialization failed: {e}") from e


# -----------------------------------------------------------------------------
# Signal generation
# -----------------------------------------------------------------------------

def generate_signal(clean_df: pd.DataFrame, env_vars: Dict[str, str]) -> None:
    """
    Eğitilmiş model(ler) ile en güncel satır için sinyal üretir.
    - Önce online modeli (varsa) kullanır
    - Yoksa batch modeline düşer
    - Çıkan olasılıktan BUY / SELL / HOLD kararı çıkarır
    """
    model_dir = env_vars.get("MODEL_DIR", "models")
    batch_model_name = env_vars.get("BATCH_MODEL_NAME", "batch_model")
    online_model_name = env_vars.get("ONLINE_MODEL_NAME", "online_model")

    online_model_path = os.path.join(model_dir, f"{online_model_name}.joblib")
    batch_model_path = os.path.join(model_dir, f"{batch_model_name}.joblib")

    model = None
    model_source = None

    # Önce online modeli dene
    if os.path.exists(online_model_path):
        try:
            model = joblib_load(online_model_path)
            model_source = "online"
            logger.info("[SIGNAL] Loaded online model from %s", online_model_path)
        except Exception as e:
            logger.error("[SIGNAL] Failed to load online model (%s): %s", online_model_path, e, exc_info=True)

    # Online yoksa veya yüklenemezse batch model
    if model is None and os.path.exists(batch_model_path):
        try:
            model = joblib_load(batch_model_path)
            model_source = "batch"
            logger.info("[SIGNAL] Loaded batch model from %s", batch_model_path)
        except Exception as e:
            logger.error("[SIGNAL] Failed to load batch model (%s): %s", batch_model_path, e, exc_info=True)

    if model is None:
        logger.error("[SIGNAL] No model available to generate signal (online/batch both missing).")
        return

    # Aynı feature kolonları: future_return ve label hariç
    feature_columns: List[str] = [
        c for c in clean_df.columns if c not in ("future_return", "label")
    ]

    if clean_df.empty:
        logger.error("[SIGNAL] clean_df is empty; cannot generate signal.")
        return

    # En son satır (en güncel bar)
    X_live = clean_df[feature_columns].iloc[[-1]]  # DataFrame olarak (1, n_features)
    try:
        proba = model.predict_proba(X_live)  # Şekil genelde (1, n_classes)
    except Exception as e:
        logger.error("[SIGNAL] model.predict_proba failed: %s", e, exc_info=True)
        return

    # --- PROBA → scalar p_buy (sadece class=1 olasılığı) ---
    # Burada asıl hatayı düzeltiyoruz:
    #   TypeError: only length-1 arrays can be converted to Python scalars
    # Çünkü predict_proba çıktısı bir array; doğrudan float() yapamayız.
    if proba is None:
        logger.error("[SIGNAL] predict_proba returned None.")
        return

    proba = np.asarray(proba)

    # class=1'in indexini güvenli şekilde bul
    try:
        if hasattr(model, "classes_") and 1 in model.classes_:
            buy_idx = int(np.where(model.classes_ == 1)[0][0])
        else:
            # Eğer classes_ yoksa veya içinde 1 yoksa:
            # son kolonu "BUY" varsayıyoruz
            buy_idx = -1

        if proba.ndim == 2:
            p_buy = float(proba[0, buy_idx])
        else:
            # Olağan dışı durum: (n,) gibi tek boyutlu array
            p_buy = float(np.ravel(proba)[buy_idx])
    except Exception as e:
        logger.error("[SIGNAL] Failed to extract BUY probability from predict_proba output: %s", e, exc_info=True)
        return

    logger.info("[SIGNAL] Model source=%s, p_buy=%.4f", model_source, p_buy)

    # Threshold'lar
    buy_threshold = float(env_vars.get("BUY_THRESHOLD", "0.6"))
    sell_threshold = float(env_vars.get("SELL_THRESHOLD", "0.4"))

    action = "HOLD"
    if p_buy >= buy_threshold:
        action = "BUY"
    elif p_buy <= sell_threshold:
        action = "SELL"

    logger.info(
        "[SIGNAL] Decision=%s (p_buy=%.4f, buy_th=%.3f, sell_th=%.3f)",
        action,
        p_buy,
        buy_threshold,
        sell_threshold,
    )

    # Burada gerçek trade executer / risk manager entegrasyonunu çağırabilirsin.
    # Örneğin:
    # trade_executor = TradeExecutor(env_vars=env_vars, logger=logger)
    # trade_executor.execute(action=action, prob=p_buy, latest_row=clean_df.iloc[-1])
    # Şimdilik sadece log atmakla yetiniyoruz.


# -----------------------------------------------------------------------------
# Orkestrasyon: data + model + signal
# -----------------------------------------------------------------------------

async def run_data_and_model_pipeline(env_vars: Dict[str, str]) -> None:
    """
    Tam pipeline:
      1) Data pipeline
      2) Batch & online model eğitimi
      3) Signal üretimi
    """
    clean_df = await run_data_pipeline(env_vars)
    train_models_and_update_state(clean_df, env_vars)
    generate_signal(clean_df, env_vars)


# -----------------------------------------------------------------------------
# Bot loop (background task)
# -----------------------------------------------------------------------------

async def bot_loop() -> None:
    logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")
    interval = int(ENV_VARS.get("BOT_LOOP_INTERVAL", "60"))

    while True:
        try:
            await run_data_and_model_pipeline(ENV_VARS)
        except Exception as e:
            logger.error("💥 [BOT] Unexpected error in bot_loop: %s", e, exc_info=True)
        await asyncio.sleep(interval)


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------

async def main() -> None:
    env_name = ENV_VARS.get("ENVIRONMENT", "production")
    port = int(os.getenv("PORT", "8080"))

    logger.info("🌐 [MAIN] Starting HTTP server on 0.0.0.0:%d (ENV=%s)", port, env_name)

    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info("🔁 [MAIN] Starting background bot_loop task...")
    bot_task = asyncio.create_task(bot_loop())

    stop_event = asyncio.Event()

    def _handle_shutdown():
        logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
        if not bot_task.done():
            bot_task.cancel()
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_shutdown)
        except NotImplementedError:
            # Windows vb. ortamlarda signal handler olmayabilir, sorun değil.
            pass

    try:
        await stop_event.wait()
    finally:
        if not bot_task.done():
            bot_task.cancel()
        await runner.cleanup()
        logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
