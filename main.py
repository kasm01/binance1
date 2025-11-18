import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from aiohttp import web

# ------------------------------
# Core & Config
# ------------------------------
from config.credentials import Credentials
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler

# ------------------------------
# Data Modules
# ------------------------------
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.batch_learning import BatchLearner
from data.online_learning import OnlineLearner

# ------------------------------
# Models
# ------------------------------
from models.fallback_model import FallbackModel

logger = logging.getLogger("binance1_pro_main")

# Global state (online model vb.)
STATE: Dict[str, Any] = {
    "online_learner": None,
    "batch_model": None,
    "feature_columns": None,
    "fallback_model": FallbackModel(default_proba=0.5),
}

# ------------------------------
# Yardımcı fonksiyonlar
# ------------------------------


def build_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    up_thresh: float = 0.002,
) -> pd.DataFrame:
    """
    future_return ve binary target üretir.
      - future_return: close(t+h) / close(t) - 1
      - target: future_return > up_thresh -> 1, else 0
    """
    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("build_labels: 'close' column not in DataFrame")

    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1.0

    valid_mask = df["future_return"].notnull()
    labeled = df[valid_mask].copy()

    labeled["target"] = (labeled["future_return"] > up_thresh).astype(int)

    # Label istatistiklerini logla
    n = len(labeled)
    if n > 0:
        mean_ret = labeled["future_return"].mean()
        std_ret = labeled["future_return"].std()
        pos = int(labeled["target"].sum())
        neg = n - pos
        pos_ratio = pos / n if n > 0 else 0.0

        system_logger.info(
            "[LABEL] future_return mean=%.4f, std=%.4f, positive ratio=%.3f (%.1f%%), "
            "pos=%d, neg=%d, n=%d",
            mean_ret,
            std_ret,
            pos_ratio,
            pos_ratio * 100,
            pos,
            neg,
            n,
        )
    else:
        system_logger.warning("[LABEL] No valid rows after future_return computation.")

    return labeled


def get_feature_columns(df: pd.DataFrame, extra_drop: Optional[List[str]] = None) -> List[str]:
    """
    Sayısal feature kolonlarını seçer.
    target, future_return vb. label kolonlarını hariç tutar.
    """
    drop_cols = {"target", "future_return"}
    if extra_drop:
        drop_cols.update(extra_drop)

    numeric = df.select_dtypes(
        include=["float32", "float64", "int32", "int64"]
    ).copy()

    feature_cols = [c for c in numeric.columns if c not in drop_cols]

    return feature_cols


async def run_data_and_model_pipeline() -> None:
    """
    Tek bir tur için:
      - Binance'ten kline verisi çek
      - Feature üret
      - Anomali temizle
      - Label üret
      - Batch model eğit
      - OnlineLearner güncelle
      - Son bar için p_buy logla
    """
    # 1️⃣ Credentials / env
    Credentials.validate()
    env_vars = dict(os.environ)

    symbol = env_vars.get("SYMBOL", "BTCUSDT")
    interval = env_vars.get("INTERVAL", "1m")
    limit = int(env_vars.get("LIMIT", "500"))

    # 2️⃣ Data yükleme
    system_logger.info(
        "[DATA] Fetching %d klines from Binance for %s (%s)",
        limit,
        symbol,
        interval,
    )

    data_loader = DataLoader(api_keys=env_vars)
    # Senin DataLoader'ında load_recent_data varsa onu kullan,
    # yoksa fetch_binance_data'yı kullanacak şekilde try/except koyuyoruz.
    try:
        if hasattr(data_loader, "load_recent_data"):
            raw_df = data_loader.load_recent_data(symbol=symbol, interval=interval, limit=limit)
        else:
            raw_df = data_loader.fetch_binance_data(symbol=symbol, interval=interval, limit=limit)
    except Exception as e:
        system_logger.error(f"[DATA] Error while fetching Binance data: {e}")
        return

    if raw_df is None or len(raw_df) == 0:
        system_logger.warning("[DATA] Empty DataFrame from Binance, skipping this cycle.")
        return

    # Kullandığın kolon düzenine uyum (eğer load_recent_data zaten DF veriyorsa bu blok gerekmeyebilir)
    if isinstance(raw_df.columns[0], int):
        # Eski fetch_binance_data formatı: 12 kolon
        raw_df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]

    # float'a çevir
    for col in ["open", "high", "low", "close", "volume"]:
        raw_df[col] = raw_df[col].astype(float)

    system_logger.info("[DATA] Raw DF shape: %s", raw_df.shape)

    # 3️⃣ Feature engineering
    fe = FeatureEngineer(raw_data=raw_df)
    feat_df = fe.transform()
    if feat_df is None or len(feat_df) == 0:
        system_logger.warning("[FE] Empty features DF, skipping this cycle.")
        return

    system_logger.info("[FE] Features DF shape: %s", feat_df.shape)

    # 4️⃣ Anomali tespiti
    anom = AnomalyDetector(features_df=feat_df)
    clean_df = anom.remove_anomalies()

    if clean_df is None or len(clean_df) == 0:
        system_logger.warning("[ANOM] Empty clean DF after anomaly removal, skipping.")
        return

    system_logger.info(
        "[ANOM] Clean DF shape: %s (removed %d rows)",
        clean_df.shape,
        len(feat_df) - len(clean_df),
    )

    # 5️⃣ Label üretimi
    horizon = int(env_vars.get("LABEL_HORIZON", "5"))
    up_thresh = float(env_vars.get("LABEL_UP_THRESH", "0.002"))  # ~0.2%

    labeled_df = build_labels(clean_df, horizon=horizon, up_thresh=up_thresh)

    # Çok az veri varsa eğitme
    if len(labeled_df) < 200:
        system_logger.warning(
            "[LABEL] Not enough labeled samples (%d) for training, skipping.",
            len(labeled_df),
        )
        return

    # 6️⃣ Feature kolonları seç
    feature_cols = get_feature_columns(labeled_df)
    if not feature_cols:
        system_logger.error("[MODEL] No numeric feature columns found, aborting training.")
        return

    X = labeled_df[feature_cols].values
    y = labeled_df["target"].astype(int).values

    n_samples, n_features = X.shape
    system_logger.info(
        "[MODEL] Training batch model on %d samples, %d features. Using %d feature columns.",
        n_samples,
        n_features,
        len(feature_cols),
    )

    # BatchLearner, features_df içinde target kolonunu bekliyor
    batch_input = labeled_df[feature_cols + ["target"]].copy()
    batch_learner = BatchLearner(features_df=batch_input, target_column="target")
    batch_model = batch_learner.train()

    if batch_model is None:
        system_logger.error("[MODEL] Batch model training failed (None), skipping.")
        return

    # Modeli global state içine yaz
    STATE["batch_model"] = batch_model
    STATE["feature_columns"] = feature_cols

    # 7️⃣ OnlineLearner init / update
    online_learner: Optional[OnlineLearner] = STATE.get("online_learner")

    if online_learner is None:
        system_logger.info("[ONLINE] Initializing OnlineLearner with batch data.")
        online_learner = OnlineLearner(base_model=batch_model, classes=(0, 1))
        # İlk fit tüm veriye
        online_learner.initialize_with_batch(X, y)
        STATE["online_learner"] = online_learner
    else:
        # Son 50 örneği ile kısmi güncelleme
        window = min(50, len(X))
        X_new = X[-window:]
        y_new = y[-window:]
        online_learner.partial_update(X_new, y_new)
        system_logger.info("[ONLINE] partial_update done on last %d samples.", window)

    # 8️⃣ En son bar için sinyal üretimi
    # Son satırın feature'ları
    last_row = labeled_df.iloc[[-1]]  # DataFrame olarak tut
    X_last = last_row[feature_cols].values

    p_buy = 0.5
    source = "fallback"

    try:
        if online_learner is not None:
            proba = online_learner.predict_proba(X_last)
            # shape: (1,2) -> [p0, p1]
            p_buy = float(proba[0, 1])
            source = "online"
        elif batch_model is not None and hasattr(batch_model, "predict_proba"):
            proba = batch_model.predict_proba(X_last)
            p_buy = float(proba[0, 1])
            source = "batch"
        else:
            # fallback model
            fb = STATE["fallback_model"]
            proba = fb.predict_proba(X_last)
            p_buy = float(proba[0, 1])
            source = "fallback"
    except Exception as e:
        system_logger.error(f"[SIGNAL] Error while computing probability: {e}")
        # fallback'a dön
        fb = STATE["fallback_model"]
        proba = fb.predict_proba(X_last)
        p_buy = float(proba[0, 1])
        source = "fallback-error"

    # Basit karar
    decision = "HOLD"
    if p_buy > 0.6:
        decision = "BUY"
    elif p_buy < 0.4:
        decision = "SELL"

    system_logger.info(
        "[SIGNAL] Latest p_buy=%.4f for %s (up_thresh=%.4f, horizon=%d, source=%s, decision=%s)",
        p_buy,
        symbol,
        up_thresh,
        horizon,
        source,
        decision,
    )


# ------------------------------
# Bot Loop
# ------------------------------


async def bot_loop():
    """
    Ana periyodik bot döngüsü.
    Her turda:
      - Veri + feature + anomaly + label + model + sinyal pipeline
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    while True:
        try:
            await run_data_and_model_pipeline()

            system_logger.info(
                "⏱ [BOT] Heartbeat - bot_loop running with data+model pipeline."
            )

        except Exception as e:
            logger.exception(f"💥 [BOT] Unexpected error in bot_loop: {e}")

        # Bir sonraki turdan önce bekleme (örn. 60 sn)
        await asyncio.sleep(60)


# ------------------------------
# HTTP / Health Endpoints (Cloud Run)
# ------------------------------


async def health_handler(request):
    return web.Response(text="OK")


async def ready_handler(request):
    return web.Response(text="READY")


async def on_startup(app: web.Application):
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    app["bot_task"] = asyncio.create_task(bot_loop())


async def on_cleanup(app: web.Application):
    system_logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
    task = app.get("bot_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def main():
    """
    Cloud Run için entry point.
    """
    setup_logger("binance1_pro_entry")
    GlobalExceptionHandler.register()

    port = int(os.environ.get("PORT", "8080"))
    env_name = os.getenv("ENV", "production")

    system_logger.info(
        f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{port} (ENV={env_name})"
    )

    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
