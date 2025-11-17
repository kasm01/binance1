import os
import asyncio
import logging
from typing import Optional

from aiohttp import web
import pandas as pd
import numpy as np

# ------------------------------
# Core & Config
# ------------------------------
from config.credentials import Credentials
from config.settings import Settings
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler

# ------------------------------
# Data & ML
# ------------------------------
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.batch_learning import BatchLearner
from data.online_learning import OnlineLearner


logger = logging.getLogger("binance1_pro_main")

# ------------------------------
# Global model state
# ------------------------------
batch_model: Optional[object] = None
online_learner: Optional[OnlineLearner] = None


def run_data_pipeline(symbol: str, interval: str = "1m", limit: int = 500):
    """
    Sync çalışan data + feature + anomaly + model pipeline.
    bot_loop içinde asyncio.to_thread(...) ile çağrılıyor.
    """
    global batch_model, online_learner

    try:
        # 1) Veri çek
        system_logger.info(
            f"[DATA] Fetching {limit} klines from Binance for {symbol} ({interval})"
        )

        data_loader = DataLoader(api_keys={})
        raw_df = data_loader.fetch_binance_data(
            symbol=symbol, interval=interval, limit=limit
        )

        if raw_df is None or raw_df.empty:
            system_logger.warning("[DATA] Empty DataFrame returned from Binance.")
            return

        # Tip dönüşümleri
        for col in ["open", "high", "low", "close", "volume"]:
            if col in raw_df.columns:
                try:
                    raw_df[col] = raw_df[col].astype(float)
                except Exception as e:
                    logger.warning(f"[DATA] Failed to cast column {col} to float: {e}")

        # Zaman index'i
        if "open_time" in raw_df.columns:
            try:
                raw_df["open_time"] = pd.to_datetime(raw_df["open_time"], unit="ms")
                raw_df.set_index("open_time", inplace=True)
            except Exception as e:
                logger.warning(f"[DATA] Failed to set datetime index: {e}")

        system_logger.info(f"[DATA] Raw DF shape: {raw_df.shape}")

        # 2) Feature engineering
        fe = FeatureEngineer(raw_data=raw_df)
        features_df = fe.transform()
        system_logger.info(f"[FE] Features DF shape: {features_df.shape}")

        if features_df is None or features_df.empty:
            system_logger.warning("[FE] Empty features_df, skipping further steps.")
            return

        # 3) Anomali tespiti
        detector = AnomalyDetector(features_df=features_df)
        clean_df = detector.remove_anomalies()
        system_logger.info(
            f"[ANOM] Clean DF shape: {clean_df.shape} "
            f"(removed {len(features_df) - len(clean_df)} rows)"
        )

        if clean_df is None or clean_df.empty:
            system_logger.warning("[ANOM] Empty clean_df after anomaly removal.")
            return

        # -----------------------------
        # 4) Label (target) üretimi
        #    - future_return = (future_close - close) / close
        #    - target = 1  if future_return > up_thresh
        #              0  otherwise
        # -----------------------------
        if "close" not in clean_df.columns:
            system_logger.warning(
                "[LABEL] 'close' column not found in clean_df; skipping model training."
            )
            return

        df_model = clean_df.copy()
        label_horizon = 5  # 5 bar sonrası (1m ise ~5 dakika)
        df_model["future_close"] = df_model["close"].shift(-label_horizon)
        df_model["future_return"] = (
            df_model["future_close"] - df_model["close"]
        ) / df_model["close"]

        up_thresh = 0.002  # %0.2 üzeri hareketleri '1' say
        df_model["target"] = (df_model["future_return"] > up_thresh).astype(int)

        # Geleceği bilmeyen satırları bırak (son label_horizon satırı düşer)
        df_model = df_model.dropna(subset=["future_close", "future_return", "target"])

        if len(df_model) < 200:
            system_logger.info(
                f"[LABEL] Not enough samples for training (have {len(df_model)}, need >= 200)."
            )
            return

        # Label istatistiklerini logla
        pos_ratio = float(df_model["target"].mean())
        num_pos = int(df_model["target"].sum())
        num_neg = int(len(df_model) - num_pos)
        system_logger.info(
            "[LABEL] future_return mean=%.4f, std=%.4f, positive ratio=%.3f (%.1f%%), "
            "pos=%d, neg=%d, n=%d",
            float(df_model["future_return"].mean()),
            float(df_model["future_return"].std()),
            pos_ratio,
            pos_ratio * 100,
            num_pos,
            num_neg,
            len(df_model),
        )

        if num_pos < 20:
            system_logger.warning(
                "[LABEL] Too few positive samples (%d); training this loop may be unstable.",
                num_pos,
            )

        # -----------------------------
        # 5) X / y hazırlanması (OnlineLearner için)
        #    !! Gelecek bilgisi içeren kolonları feature'dan çıkar !!
        # -----------------------------
        df_for_xy = df_model.dropna().copy()

        numeric_cols = df_for_xy.select_dtypes(
            include=["float32", "float64", "int32", "int64"]
        ).columns.tolist()

        # Feature olarak kullanılmayacak kolonlar
        exclude_cols = ["target", "future_close", "future_return"]
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if not numeric_cols:
            system_logger.warning(
                "[MODEL] No numeric feature columns left after exclusion; skipping training."
            )
            return

        X = df_for_xy[numeric_cols].values
        y = df_for_xy["target"].astype(int).values

        system_logger.info(
            f"[MODEL] Training batch model on {X.shape[0]} samples, "
            f"{X.shape[1]} features. Using {len(numeric_cols)} feature columns."
        )

        # -----------------------------
        # 6) Batch training
        # -----------------------------
        try:
            batch_learner = BatchLearner(features_df=df_for_xy, target_column="target")
            batch_model_local = batch_learner.train()
            batch_model = batch_model_local
        except Exception as e:
            logger.exception(f"[MODEL] BatchLearner training failed: {e}")
            return

        # -----------------------------
        # 7) Online learner init / update
        # -----------------------------
        if online_learner is None:
            system_logger.info("[ONLINE] Initializing OnlineLearner with batch data.")
            online_local = OnlineLearner(base_model=batch_model, classes=(0, 1))
            try:
                online_local.initialize_with_batch(X, y)
                online_learner = online_local
            except Exception as e:
                logger.exception(f"[ONLINE] initialize_with_batch failed: {e}")
                return
        else:
            tail_n = min(50, X.shape[0])
            X_new = X[-tail_n:]
            y_new = y[-tail_n:]
            try:
                online_learner.partial_update(X_new, y_new)
                system_logger.info(
                    f"[ONLINE] partial_update done on last {tail_n} samples."
                )
            except Exception as e:
                logger.exception(f"[ONLINE] partial_update failed: {e}")

        # -----------------------------
        # 8) Son bar için sinyal logla
        # -----------------------------
        try:
            X_last = X[-1:].copy()
            if online_learner is not None:
                proba = online_learner.predict_proba(X_last)[0, 1]
            elif batch_model is not None and hasattr(batch_model, "predict_proba"):
                proba = batch_model.predict_proba(X_last)[0, 1]
            else:
                proba = 0.5

            system_logger.info(
                f"[SIGNAL] Latest p_buy={proba:.3f} for {symbol} "
                f"(up_thresh={up_thresh:.4f}, horizon={label_horizon})"
            )
        except Exception as e:
            logger.warning(f"[SIGNAL] Could not compute latest signal: {e}")

    except Exception as e:
        logger.exception(
            f"[PIPELINE] Error in data/feature/anomaly/model pipeline for {symbol}: {e}"
        )



# ------------------------------
# Bot Loop
# ------------------------------
async def bot_loop():
    """
    Binance1-Pro botunun çekirdek döngüsü.

    Şu an:
      - Belirli aralıklarla Binance'ten veri çekiyor
      - Feature üretiyor
      - Anomali temizliği yapıyor
      - Label üretip Batch + Online model eğitiyor
      - En son p_buy oranını logluyor
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    symbols = Settings.TRADE_SYMBOLS or ["BTCUSDT"]
    symbol = symbols[0]
    interval = "1m"
    limit = 500

    while True:
        try:
            await asyncio.to_thread(run_data_pipeline, symbol, interval, limit)
            system_logger.info(
                "⏱ [BOT] Heartbeat - bot_loop running with data+model pipeline."
            )
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")
            break
        except Exception as e:
            logger.exception(f"[BOT] Unexpected error in bot_loop: {e}")

        # Binance rate-limit'e saygı: her loop sonrası 60 sn bekle
        await asyncio.sleep(60)


# ------------------------------
# HTTP Handlers (Cloud Run)
# ------------------------------
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text="OK")


async def ready_handler(request: web.Request) -> web.Response:
    return web.Response(text="READY")


async def start_background_tasks(app: web.Application):
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    app["bot_task"] = asyncio.create_task(bot_loop())


async def cleanup_background_tasks(app: web.Application):
    system_logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
    bot_task: asyncio.Task = app.get("bot_task")
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            system_logger.info("✅ [MAIN] bot_loop cancelled gracefully.")


async def create_app() -> web.Application:
    """
    Hem health endpoint'lerini hem de background bot'u yöneten aiohttp uygulaması.
    """
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


def main():
    """
    Cloud Run için entry point.
    """
    setup_logger("binance1_pro_entry")
    GlobalExceptionHandler.register()
    Credentials.validate()

    port = int(os.environ.get("PORT", "8080"))
    system_logger.info(
        f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{port} (ENV={Settings.ENV})"
    )

    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
