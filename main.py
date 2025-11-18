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
from config.credentials import Credentials  # Şimdilik ENV üzerinden okuyoruz, ama kalsın
from core.logger import setup_logger, system_logger
from core.exceptions import GlobalExceptionHandler  # Kullanmak istersen hazır
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

# Global state
STATE: Dict[str, Any] = {
    "online_learner": None,
    "batch_model": None,
    "feature_columns": None,
    "fallback_model": FallbackModel(default_proba=0.5),
    "last_signal": None,
}

# -------------------------------------------------
# Label oluşturma
# -------------------------------------------------
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
            pos_ratio * 100.0,
            pos,
            neg,
            n,
        )
    else:
        system_logger.warning("[LABEL] No valid samples after labeling.")

    return labeled


# -------------------------------------------------
# ENV helper
# -------------------------------------------------
def load_env_vars() -> Dict[str, str]:
    env_vars = dict(os.environ)

    # Varsayılanlar
    env_vars.setdefault("BINANCE_SYMBOL", "BTCUSDT")
    env_vars.setdefault("BINANCE_INTERVAL", "1m")
    env_vars.setdefault("BOT_LOOP_INTERVAL", "60")  # saniye
    env_vars.setdefault("UP_THRESH", "0.002")
    env_vars.setdefault("LABEL_HORIZON", "5")

    return env_vars


# -------------------------------------------------
# Data Pipeline
# -------------------------------------------------
async def run_data_pipeline(env_vars: Dict[str, str]) -> pd.DataFrame:
    """
    1) Binance'ten son veriyi çek (fetch_binance_data)
    2) (Varsa) harici feature'ları ekle (fetch_external_data)
    3) Feature engineering + anomali temizliği
    """
    data_loader = DataLoader(env_vars)

    symbol = env_vars.get("BINANCE_SYMBOL", "BTCUSDT")
    interval = env_vars.get("BINANCE_INTERVAL", "1m")

    if not hasattr(data_loader, "fetch_binance_data"):
        available = [a for a in dir(data_loader) if not a.startswith("_")]
        system_logger.error(
            "[DATA] DataLoader has no method fetch_binance_data. Available: %s",
            available,
        )
        raise AttributeError(
            "DataLoader is missing fetch_binance_data. "
            "Check data/data_loader.py"
        )

    # 🔧 ÖNCEKİ HATA: symbol parametresi verilmeden çağrılıyordu
    try:
        # Muhtemel imza: (symbol, interval='1m', limit=500, ...)
        raw_df = data_loader.fetch_binance_data(symbol=symbol, interval=interval)
    except TypeError:
        # Eğer sadece (symbol, ...) kabul ediyorsa
        raw_df = data_loader.fetch_binance_data(symbol)

    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        raise ValueError("[DATA] fetch_binance_data returned empty or invalid DataFrame")

    system_logger.info("[DATA] Raw DF shape: %s", raw_df.shape)

    # 2) Ek harici veri varsa merge etmeyi dene (opsiyonel)
    if hasattr(data_loader, "fetch_external_data"):
        try:
            ext_df = data_loader.fetch_external_data()
            if isinstance(ext_df, pd.DataFrame) and not ext_df.empty:
                if "open_time" in raw_df.columns and "open_time" in ext_df.columns:
                    raw_df = raw_df.merge(ext_df, on="open_time", how="left")
                    system_logger.info(
                        "[DATA] Merged external data on 'open_time'. Raw DF shape now: %s",
                        raw_df.shape,
                    )
                else:
                    raw_df = pd.concat(
                        [raw_df.reset_index(drop=True), ext_df.reset_index(drop=True)],
                        axis=1,
                    )
                    system_logger.info(
                        "[DATA] Concatenated external data by index. Raw DF shape now: %s",
                        raw_df.shape,
                    )
        except Exception as e:
            system_logger.exception(
                "[DATA] Error in fetch_external_data, continuing with Binance data only: %s",
                e,
            )

    # 3) Feature engineering
    feature_engineer = FeatureEngineer(raw_df)
    features_df = feature_engineer.transform()
    # FeatureEngineer içinde [FE] logları yazılıyor.

    # 4) Anomali tespiti / temizliği
    anomaly_detector = AnomalyDetector(features_df)
    clean_df = anomaly_detector.remove_anomalies()
    # AnomalyDetector içinde [ANOM] logları yazılıyor.

    return clean_df


# -------------------------------------------------
# Batch + Online learning
# -------------------------------------------------
def train_models_and_update_state(
    clean_df: pd.DataFrame,
    env_vars: Dict[str, str],
) -> None:
    """
    - Label üret
    - Batch model eğit
    - OnlineLearner'ı initialize/partial_update
    - STATE içindeki modelleri güncelle
    """
    up_thresh = float(env_vars.get("UP_THRESH", "0.002"))
    horizon = int(env_vars.get("LABEL_HORIZON", "5"))

    labeled_df = build_labels(clean_df, horizon=horizon, up_thresh=up_thresh)

    if labeled_df.empty:
        system_logger.warning("[MODEL] No labeled data available, skipping training.")
        return

    # Target ve feature kolonlarını ayır
    # 'future_return' ve 'target' dışındaki numeric kolonları feature yap
    ignore_cols = {"target", "future_return"}
    feature_cols: List[str] = [
        c for c in labeled_df.columns
        if c not in ignore_cols and np.issubdtype(labeled_df[c].dtype, np.number)
    ]

    if not feature_cols:
        system_logger.error("[MODEL] No numeric feature columns found, cannot train.")
        return

    X = labeled_df[feature_cols]
    y = labeled_df["target"].astype(int)

    n_samples, n_features = X.shape
    system_logger.info(
        "[MODEL] Training batch model on %d samples, %d features. Using %d feature columns.",
        n_samples,
        n_features,
        len(feature_cols),
    )

    # 🔧 ÖNCEKİ HATA: BatchLearner(features_df=..., target_column=...) kullanılıyordu.
    # Şimdi X, y ve feature_columns ile çağırıyoruz.
    batch_learner = BatchLearner(
        X=X,
        y=y,
        feature_columns=feature_cols,
    )
    batch_model = batch_learner.fit()

    STATE["batch_model"] = batch_model
    STATE["feature_columns"] = feature_cols

    # Online learner init / update
    if STATE["online_learner"] is None:
        system_logger.info("[ONLINE] Initializing OnlineLearner with batch data.")
        online_learner = OnlineLearner(
            base_model=batch_model,
            feature_columns=feature_cols,
        )
        # İlk full data ile initialize
        online_learner.initial_fit(X, y)
        STATE["online_learner"] = online_learner
    else:
        # Son 50 sample ile incremental update
        online_learner: OnlineLearner = STATE["online_learner"]
        chunk_size = 50 if len(X) > 50 else len(X)
        if chunk_size > 0:
            X_tail = X.iloc[-chunk_size:]
            y_tail = y.iloc[-chunk_size:]
            online_learner.partial_update(X_tail, y_tail)
        system_logger.info("[ONLINE] partial_update done on last %d samples.", chunk_size)


# -------------------------------------------------
# Signal generation
# -------------------------------------------------
def generate_signal(
    df: pd.DataFrame,
    env_vars: Dict[str, str],
) -> Dict[str, Any]:
    """
    Son satır için BUY/SELL sinyali üretir.
    """
    symbol = env_vars.get("BINANCE_SYMBOL", "BTCUSDT")
    up_thresh = float(env_vars.get("UP_THRESH", "0.002"))
    horizon = int(env_vars.get("LABEL_HORIZON", "5"))

    feature_cols: Optional[List[str]] = STATE.get("feature_columns")
    if not feature_cols:
        system_logger.warning(
            "[SIGNAL] No feature_columns in STATE, using fallback model for %s",
            symbol,
        )
        p_buy = STATE["fallback_model"].predict_proba({})
        decision = "BUY" if p_buy >= 0.5 else "SELL"
        system_logger.info(
            "[SIGNAL] Latest p_buy=%.4f for %s (up_thresh=%.4f, horizon=%d, source=fallback, decision=%s)",
            p_buy,
            symbol,
            up_thresh,
            horizon,
            decision,
        )
        signal = {
            "symbol": symbol,
            "p_buy": p_buy,
            "decision": decision,
            "source": "fallback",
        }
        STATE["last_signal"] = signal
        return signal

    if df.empty:
        system_logger.warning("[SIGNAL] Empty DataFrame, cannot generate signal.")
        p_buy = STATE["fallback_model"].predict_proba({})
        decision = "BUY" if p_buy >= 0.5 else "SELL"
        signal = {
            "symbol": symbol,
            "p_buy": p_buy,
            "decision": decision,
            "source": "fallback",
        }
        STATE["last_signal"] = signal
        return signal

    latest_row = df.iloc[-1:]
    latest_features = latest_row[feature_cols]

    online_learner: Optional[OnlineLearner] = STATE.get("online_learner")
    batch_model = STATE.get("batch_model")

    source = "fallback"
    if online_learner is not None:
        proba = online_learner.predict_proba(latest_features)[0]
        source = "online"
    elif batch_model is not None:
        proba = batch_model.predict_proba(latest_features)[:, 1][0]
        source = "batch"
    else:
        proba = STATE["fallback_model"].predict_proba({})

    p_buy = float(proba)
    decision = "BUY" if p_buy >= 0.5 else "SELL"

    system_logger.info(
        "[SIGNAL] Latest p_buy=%.4f for %s (up_thresh=%.4f, horizon=%d, source=%s, decision=%s)",
        p_buy,
        symbol,
        up_thresh,
        horizon,
        source,
        decision,
    )

    signal = {
        "symbol": symbol,
        "p_buy": p_buy,
        "decision": decision,
        "source": source,
    }
    STATE["last_signal"] = signal
    return signal


# -------------------------------------------------
# Full pipeline: data + model + signal
# -------------------------------------------------
async def run_data_and_model_pipeline(env_vars: Dict[str, str]) -> None:
    clean_df = await run_data_pipeline(env_vars)
    train_models_and_update_state(clean_df, env_vars)
    generate_signal(clean_df, env_vars)


# -------------------------------------------------
# HTTP Handlers
# -------------------------------------------------
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def status_handler(request: web.Request) -> web.Response:
    resp = {
        "online_learner": STATE["online_learner"] is not None,
        "batch_model": STATE["batch_model"] is not None,
        "feature_columns": STATE["feature_columns"],
        "last_signal": STATE["last_signal"],
    }
    return web.json_response(resp)


async def signal_handler(request: web.Request) -> web.Response:
    """
    Anlık sinyal (STATE’teki en son clean_df üzerinden değil, sadece son üretilen sinyal).
    Gerçek zamanlı pipeline çalışması bot_loop üzerinden yürütülüyor.
    """
    if STATE["last_signal"] is None:
        return web.json_response({"error": "No signal generated yet."}, status=503)
    return web.json_response(STATE["last_signal"])


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_get("/signal", signal_handler)
    return app


# -------------------------------------------------
# Bot Loop
# -------------------------------------------------
async def bot_loop():
    env_vars = load_env_vars()
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    while True:
        try:
            await run_data_and_model_pipeline(env_vars)
            system_logger.info("⏱ [BOT] Heartbeat - bot_loop running with data+model pipeline.")
        except asyncio.CancelledError:
            system_logger.info("🛑 [BOT] bot_loop cancelled, shutting down.")
            break
        except Exception as e:
            system_logger.exception("💥 [BOT] Unexpected error in bot_loop: %s", e)

        interval = int(env_vars.get("BOT_LOOP_INTERVAL", "60"))
        await asyncio.sleep(interval)


# -------------------------------------------------
# Main entrypoint
# -------------------------------------------------
async def _main_async():
    setup_logger()

    env = os.environ.get("ENV", "production")
    port = int(os.environ.get("PORT", "8080"))

    system_logger.info("🌐 [MAIN] Starting HTTP server on 0.0.0.0:%d (ENV=%s)", port, env)

    app = create_app()

    # Background bot_loop
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    loop = asyncio.get_running_loop()
    bot_task = loop.create_task(bot_loop())

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)

    try:
        await site.start()
        system_logger.info("======== Running on http://0.0.0.0:%d ========", port)
        system_logger.info("(Press CTRL+C to quit)")
        # Sonsuza kadar bekle
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        system_logger.info("🧹 [MAIN] Cleaning up background bot_loop task...")
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
    finally:
        await runner.cleanup()


def main():
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
