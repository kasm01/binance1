import os
import asyncio
import logging
from typing import Dict, Any, Tuple, List

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

# ------------------------------
# Global state
# ------------------------------
STATE: Dict[str, Any] = {
    "online_learner": None,
    "batch_model": None,
    "feature_columns": None,
    "fallback_model": FallbackModel(default_proba=0.5),
}


# ------------------------------
# Yardımcı: Label üretimi
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

    # Gelecek getiriyi hesapla
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1.0

    # NaN olan satırları at (geleceği olmayan son satırlar)
    valid_mask = df["future_return"].notnull()
    labeled = df[valid_mask].copy()

    # Binary hedef
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
            pos_ratio * 100.0,
            pos,
            neg,
            n,
        )
    else:
        system_logger.warning("[LABEL] No valid rows after labeling (n=0).")

    return labeled


# ------------------------------
# Yardımcı: Data pipeline
# ------------------------------
async def run_data_pipeline(env_vars: Dict[str, str]) -> pd.DataFrame:
    """
    1) Binance'ten son veriyi çek
    2) Feature engineering uygula
    3) Anomali temizliği yap
    """
    data_loader = DataLoader(env_vars)

    # DataLoader API'si değişmiş olabilir; esnek şekilde yüklemeyi dene
    load_method = None
    for name in ("load_recent_data", "load_data", "load"):
        if hasattr(data_loader, name):
            load_method = getattr(data_loader, name)
            system_logger.info("[DATA] Using DataLoader.%s()", name)
            break

    if load_method is None:
        # Hata verirken mevcut attribute'ları da loglayalım ki debug kolay olsun
        available = [a for a in dir(data_loader) if not a.startswith("_")]
        system_logger.error(
            "[DATA] DataLoader has no method load_recent_data/load_data/load. Available: %s",
            available,
        )
        raise AttributeError(
            "DataLoader is missing load_recent_data/load_data/load. "
            "Check data/data_loader.py"
        )

    # 1) Data yükleme
    raw_df = load_method()  # DataLoader içi zaten [DATA] loglarını basıyor olmalı

    # 2) Feature engineering
    feature_engineer = FeatureEngineer(raw_df)
    features_df = feature_engineer.transform()
    # FeatureEngineer içinde [FE] logları yazılıyor.

    # 3) Anomali tespiti / temizliği
    anomaly_detector = AnomalyDetector(features_df)
    clean_df = anomaly_detector.remove_anomalies()
    # AnomalyDetector içinde [ANOM] logları yazılıyor.

    return clean_df


# ------------------------------
# Yardımcı: X, y, feature listesi
# ------------------------------
def get_feature_target_matrices(
    labeled_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    labeled_df içinden X (features), y (target) ve feature_columns listesini çıkarır.
    """
    # target ve future_return dışındaki tüm numeric kolonları feature olarak al
    drop_cols = {"future_return", "target"}
    feature_cols = [c for c in labeled_df.columns if c not in drop_cols]

    X = labeled_df[feature_cols].copy()
    y = labeled_df["target"].copy()

    # Numerik olmayanları zorla float'a çevir (gerekirse)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = y.astype(int)

    return X, y, feature_cols


# ------------------------------
# Yardımcı: Son bar için sinyal
# ------------------------------
def compute_latest_signal(
    X: pd.DataFrame,
    feature_cols: List[str],
    horizon: int,
    up_thresh: float,
) -> None:
    """
    Son satır için p_buy hesapla, source ve karar logla.
    """
    if X.empty:
        system_logger.warning("[SIGNAL] X is empty, cannot compute latest signal.")
        return

    latest_features = X.iloc[[-1]]  # shape (1, n_features)

    source = "fallback"
    p_buy = 0.5

    # Önce online_learner varsa onu kullan
    if STATE["online_learner"] is not None:
        online = STATE["online_learner"]
        try:
            proba = online.predict_proba(latest_features)  # shape (1, 2)
            p_buy = float(proba[0, 1])
            source = "online"
        except Exception as e:
            system_logger.exception(
                "[SIGNAL] Error using OnlineLearner predict_proba: %s", e
            )

    # OnlineLearner yoksa ama batch_model varsa onu kullan
    elif STATE["batch_model"] is not None:
        batch_model = STATE["batch_model"]
        try:
            proba = batch_model.predict_proba(latest_features)
            p_buy = float(proba[0, 1])
            source = "batch"
        except Exception as e:
            system_logger.exception(
                "[SIGNAL] Error using batch_model predict_proba: %s", e
            )

    # Hiçbiri yoksa fallback model
    else:
        fallback = STATE["fallback_model"]
        try:
            proba = fallback.predict_proba(latest_features)
            p_buy = float(proba[0, 1])
            source = "fallback"
        except Exception as e:
            system_logger.exception(
                "[SIGNAL] Error using fallback_model predict_proba: %s", e
            )

    decision = "BUY" if p_buy >= 0.5 else "SELL"

    system_logger.info(
        "[SIGNAL] Latest p_buy=%.4f for BTCUSDT (up_thresh=%.4f, horizon=%d, source=%s, decision=%s)",
        p_buy,
        up_thresh,
        horizon,
        source,
        decision,
    )


# ------------------------------
# Tam data + model pipeline
# ------------------------------
async def run_data_and_model_pipeline() -> None:
    """
    Tam pipeline:
      - Data load + FE + Anomali
      - Label üret
      - Batch LightGBM train
      - OnlineLearner init / update
      - Son bar için sinyal hesapla
    """
    env_vars = dict(os.environ)

    # Label parametreleri (ENV üzerinden override edilebilir)
    horizon = int(env_vars.get("BINANCE1_LABEL_HORIZON", "5"))
    up_thresh = float(env_vars.get("BINANCE1_UP_THRESH", "0.002"))

    # 1) Data pipeline
    clean_df = await run_data_pipeline(env_vars)

    # 2) Label pipeline
    labeled_df = build_labels(clean_df, horizon=horizon, up_thresh=up_thresh)

    if labeled_df.empty or "target" not in labeled_df.columns:
        system_logger.warning(
            "[MODEL] Not enough labeled data or 'target' column missing. Skipping training."
        )
        return

    # 3) X, y ve feature listesi
    X, y, feature_cols = get_feature_target_matrices(labeled_df)
    n_samples, n_features = X.shape

    if n_samples < 100:
        system_logger.warning(
            "[MODEL] Too few samples for training (n=%d). Skipping.", n_samples
        )
        return

    system_logger.info(
        "[MODEL] Training batch model on %d samples, %d features. Using %d feature columns.",
        n_samples,
        n_features,
        len(feature_cols),
    )

    # 4) Batch training (LightGBM)
    batch_learner = BatchLearner()
    batch_model = batch_learner.train(X, y)

    # Global state güncelle
    STATE["batch_model"] = batch_model
    STATE["feature_columns"] = feature_cols

    # 5) Online learner init / update
    if STATE["online_learner"] is None:
        # İlk defa initialize
        online_learner = OnlineLearner()
        online_learner.initialize_from_batch(X, y, batch_model)
        STATE["online_learner"] = online_learner
    else:
        # Son 50 örnekle partial update
        online_learner = STATE["online_learner"]
        tail_size = min(50, len(X))
        X_tail = X.iloc[-tail_size:, :]
        y_tail = y.iloc[-tail_size:]
        online_learner.partial_update(X_tail, y_tail)

    # 6) Son bar için sinyal
    compute_latest_signal(X, feature_cols, horizon=horizon, up_thresh=up_thresh)


# ------------------------------
# Bot Loop
# ------------------------------
async def bot_loop():
    """
    Cloud Run içinde arka planda sürekli çalışan bot döngüsü.
    Her çalışmada:
      - Data + model pipeline çalıştırır
      - Hata olursa loglar ama servis ayakta kalır
    """
    # Credentials check (env OK mu?)
    Credentials.validate()

    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    while True:
        try:
            await run_data_and_model_pipeline()
            system_logger.info(
                "⏱ [BOT] Heartbeat - bot_loop running with data+model pipeline."
            )
        except Exception as e:
            system_logger.exception("[BOT] Unexpected error in bot_loop: %s", e)

        # 1 dakika bekle, sonra tekrar
        await asyncio.sleep(60)


# ------------------------------
# HTTP Health Endpoints (Cloud Run)
# ------------------------------
async def health_handler(request):
    """
    Cloud Run health check endpoint.
    Sadece 'OK' döner.
    """
    return web.Response(text="OK")


async def ready_handler(request):
    """
    Opsiyonel readiness endpoint.
    Şimdilik basit 'READY' cevabı döner.
    """
    return web.Response(text="READY")


async def start_background_bot(app: web.Application):
    """
    App startup sırasında bot_loop görevini başlatır.
    """
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    app["bot_task"] = asyncio.create_task(bot_loop())


async def stop_background_bot(app: web.Application):
    """
    App cleanup sırasında bot_loop görevini durdurur.
    """
    system_logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
    bot_task = app.get("bot_task")
    if bot_task:
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")


async def create_app() -> web.Application:
    """
    Hem health endpoint'lerini hem de background bot_loop'u yöneten aiohttp uygulaması.
    """
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/healthz", health_handler)
    app.router.add_get("/ready", ready_handler)

    app.on_startup.append(start_background_bot)
    app.on_cleanup.append(stop_background_bot)

    return app


def main():
    """
    Cloud Run için entry point.

    - PORT env değişkenini alır (Cloud Run bunu otomatik set eder, default: 8080)
    - aiohttp HTTP server'ı başlatır
    - Binance1-Pro botunu background task olarak çalıştırır
    """
    # Logger ve global exception handler
    setup_logger("system")
    GlobalExceptionHandler.register()

    port = int(os.environ.get("PORT", "8080"))
    env_name = os.environ.get("ENV", "production")

    system_logger.info(
        "🌐 [MAIN] Starting HTTP server on 0.0.0.0:%d (ENV=%s)", port, env_name
    )

    web.run_app(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
