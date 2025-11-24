# main.py

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Any, Dict

import numpy as np
import pandas as pd
from aiohttp import web
from binance.client import Client as BinanceClient

from env.load_env import load_environment_variables
from config.settings import Config
from core.logger import setup_logger, system_logger
from core.utils import retry  # varsa, yoksa kaldır
from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager
from trading.trade_executor import TradeExecutor
from models.fallback_model import FallbackModel

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner

from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_system import AlertSystem
from tg_bot.telegram_bot import TelegramBot



# ────────────────────────────── health endpoint ──────────────────────────────

async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "binance1-pro"})


# ───────────────────── Binance client & trading obj init ─────────────────────

def create_binance_futures_client(env_vars: Dict[str, str]) -> BinanceClient:
    api_key = env_vars.get("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
    api_secret = env_vars.get("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        system_logger.warning(
            "[MAIN] BINANCE_API_KEY / BINANCE_API_SECRET not found in env. "
            "Client will not be authorized!"
        )

    client = BinanceClient(api_key, api_secret)
    # Testnet kullanıyorsan burada URL override edebilirsin:
    # client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return client


def init_trading_objects(env_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Tüm core trading objelerini initialize eder.
    main.py içinde bir kez çağrılır, bot_loop içinde kullanılır.
    """
    system_logger.info("[MAIN] Initializing trading objects...")

    # Binance client
    client = create_binance_futures_client(env_vars)

    # Data pipeline objeleri
    data_loader = DataLoader(
        client=client,
        symbol=Config.BINANCE_SYMBOL,
        interval=Config.BINANCE_INTERVAL,
        use_cache=True,
    )
    feature_engineer = FeatureEngineer()
    anomaly_detector = AnomalyDetector()

    # Online model + fallback
    online_learner = OnlineLearner(
        model_dir="models",
        base_model_name="online_model",
        n_classes=2,
    )
    fallback_model = FallbackModel(default_proba=0.5)

    # Risk & pozisyon yönetimi
    risk_manager = RiskManager(
        max_risk_per_trade=Config.MAX_RISK_PER_TRADE,
        max_daily_loss_pct=Config.MAX_DAILY_LOSS_PCT,
        state_file=os.path.join("logs", "risk_state.json"),
    )
    position_manager = PositionManager(log_path=os.path.join("logs", "trades.log"))

    # Trade executor
    trade_executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    # Monitoring & Telegram
    performance_tracker = PerformanceTracker()
    alert_system = AlertSystem()
    telegram_bot = TelegramBot()  # istersen main dışında ayrı süreçte koşuturabilirsin.

    objects = {
        "client": client,
        "data_loader": data_loader,
        "feature_engineer": feature_engineer,
        "anomaly_detector": anomaly_detector,
        "online_learner": online_learner,
        "fallback_model": fallback_model,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "performance_tracker": performance_tracker,
        "alert_system": alert_system,
        "telegram_bot": telegram_bot,
    }

    system_logger.info("[MAIN] Trading objects initialized successfully.")
    return objects


# ───────────────────────────── sinyal üretim katmanı ─────────────────────────────

def compute_p_buy(
    online_learner: OnlineLearner,
    fallback_model: FallbackModel,
    X_live: pd.DataFrame,
) -> float:
    """
    Online modelden p_buy hesaplar, hata olursa fallback modeli kullanır.
    """
    try:
        probs = online_learner.predict_proba(X_live)

        # probs şekli değişken olabilir: scalar / 1D / 2D
        if isinstance(probs, (list, np.ndarray)):
            probs = np.array(probs)
            if probs.ndim == 2:  # (n_samples, 2) gibi
                p_buy = float(probs[-1, 1])
            else:  # (n_samples,)
                p_buy = float(probs[-1])
        else:
            p_buy = float(probs)

        system_logger.info(
            f"[SIGNAL] p_buy={p_buy:.4f} (source=ONLINE, "
            f"BUY_THRESHOLD={Config.BUY_THRESHOLD:.2f}, "
            f"SELL_THRESHOLD={Config.SELL_THRESHOLD:.2f})"
        )
        return p_buy

    except Exception as e:
        system_logger.exception(
            f"[SIGNAL] Online model prediction failed, using fallback. Error: {e}"
        )
        probs = fallback_model.predict_proba(X_live.values)
        if isinstance(probs, (list, np.ndarray)):
            probs = np.array(probs)
            if probs.ndim == 2:
                p_buy = float(probs[-1, 1])
            else:
                p_buy = float(probs[-1])
        else:
            p_buy = float(probs)

        system_logger.info(
            f"[SIGNAL] p_buy={p_buy:.4f} (source=FALLBACK, "
            f"BUY_THRESHOLD={Config.BUY_THRESHOLD:.2f}, "
            f"SELL_THRESHOLD={Config.SELL_THRESHOLD:.2f})"
        )
        return p_buy


def generate_trading_signal(p_buy: float) -> str:
    """
    Basit kural:
      p_buy >= BUY_THRESHOLD  => BUY
      p_buy <= SELL_THRESHOLD => SELL
      aksi                     => HOLD
    """
    if p_buy >= Config.BUY_THRESHOLD:
        signal = "BUY"
    elif p_buy <= Config.SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    system_logger.info(f"[SIGNAL] Generated trading signal: {signal}")
    return signal


# ───────────────────────────── LONG/SHORT yönetimi ─────────────────────────────

def manage_positions_for_signal(
    trade_executor: TradeExecutor,
    position_manager: PositionManager,
    risk_manager: RiskManager,
    symbol: str,
    signal: str,
    current_price: float,
) -> None:
    """
    Gelen sinyale göre LONG/SHORT pozisyonlarını yönetir.

    - BUY: SHORT varsa kapat, LONG yoksa aç
    - SELL: LONG varsa kapat, SHORT yoksa aç
    - HOLD: hiçbir şey yapma (istersen SL/TP yönetimi ekleyebilirsin)
    """

    signal = signal.upper()
    long_pos = position_manager.get_position(symbol, "LONG")
    short_pos = position_manager.get_position(symbol, "SHORT")

    # Günlük zarar limiti aşıldıysa: yeni trade açma, istersen tüm pozisyonları kapat
    if risk_manager.trading_halted:
        system_logger.warning(
            "[MAIN] Trading halted for today by risk manager (MAX_DAILY_LOSS reached)."
        )
        # Burada istersen tüm pozisyonları anında kapat:
        trade_executor.flatten_all_positions({symbol: current_price})
        return

    if signal == "BUY":
        # Önce ters yönlü pozisyonu kapat (SHORT)
        if short_pos:
            trade_executor.close_position(
                symbol=symbol, direction="SHORT", exit_price=current_price
            )

        # LONG yoksa aç
        if not long_pos:
            trade_executor.open_position_from_signal(
                symbol=symbol,
                direction="LONG",
                entry_price=current_price,
                stop_loss_pct=Config.STOP_LOSS_PCT,
                leverage=Config.DEFAULT_LEVERAGE,
            )

    elif signal == "SELL":
        # Önce ters yönlü pozisyonu kapat (LONG)
        if long_pos:
            trade_executor.close_position(
                symbol=symbol, direction="LONG", exit_price=current_price
            )

        # SHORT yoksa aç
        if not short_pos:
            trade_executor.open_position_from_signal(
                symbol=symbol,
                direction="SHORT",
                entry_price=current_price,
                stop_loss_pct=Config.STOP_LOSS_PCT,
                leverage=Config.DEFAULT_LEVERAGE,
            )

    else:  # HOLD
        system_logger.info("[MAIN] HOLD signal -> no new position opened/closed.")


# ───────────────────────────── data + model pipeline ─────────────────────────────

def run_data_and_model_pipeline(
    trading_objects: Dict[str, Any],
    symbol: str,
    interval: str,
    limit: int,
) -> Dict[str, Any]:
    """
    1) Binance'ten kline verisini çek
    2) Feature engineering
    3) Anomali filtresi
    4) Online model initial_fit / partial_update
    5) Son bar için X_live, current_price, p_buy döndür

    Bu fonksiyon synchronous, bot_loop içinde çağrılıyor.
    """
    data_loader: DataLoader = trading_objects["data_loader"]
    feature_engineer: FeatureEngineer = trading_objects["feature_engineer"]
    anomaly_detector: AnomalyDetector = trading_objects["anomaly_detector"]
    online_learner: OnlineLearner = trading_objects["online_learner"]

    system_logger.info(
        f"[DATA] Starting data pipeline for {symbol} "
        f"({interval}, limit={limit})"
    )

    # 1) Kline verisi
    df_raw = data_loader.load_and_cache_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    system_logger.info(f"[DATA] Raw DF shape: {df_raw.shape}")

    if df_raw is None or df_raw.empty:
        raise RuntimeError("Empty dataframe from DataLoader.load_and_cache_klines")

    # 2) Feature engineering
    df_features = feature_engineer.build_features(df_raw)
    system_logger.info(
        f"[FE] Features DF shape: {df_features.shape}, "
        f"columns={list(df_features.columns)}"
    )

    # 3) Anomali filtresi
    df_clean = anomaly_detector.filter_anomalies(df_features)
    system_logger.info(
        f"[ANOM] After anomaly filter: {df_clean.shape[0]} rows remain."
    )

    # Yeterli veri yoksa devam etme
    if df_clean.shape[0] < 100:
        raise RuntimeError("Not enough samples after anomaly filtering.")

    # 'label' sütunu varsa ayır
    if "label" in df_clean.columns:
        feature_cols = [c for c in df_clean.columns if c not in ("open_time", "close_time", "label")]
        X = df_clean[feature_cols]
        y = df_clean["label"]
    else:
        feature_cols = [c for c in df_clean.columns if c not in ("open_time", "close_time")]
        X = df_clean[feature_cols]
        y = None

    # 4) Online learner initial_fit / partial_update
    if not online_learner.is_initialized:
        if y is None:
            raise RuntimeError("OnlineLearner initial_fit requires 'label' column.")
        system_logger.info(
            f"[ONLINE] initial_fit called with {X.shape[0]} samples, {X.shape[1]} features."
        )
        online_learner.initial_fit(X, y)
    else:
        # Son 100 bar ile partial update
        if y is not None:
            X_chunk = X.tail(100)
            y_chunk = y.tail(100)
            system_logger.info(
                f"[ONLINE] partial_update called with {X_chunk.shape[0]} samples, "
                f"{X_chunk.shape[1]} features."
            )
            online_learner.partial_update(X_chunk, y_chunk)

    # 5) Son barı X_live olarak al
    X_live = X.tail(1)
    current_price = float(df_clean["close"].iloc[-1])

    return {
        "X_live": X_live,
        "current_price": current_price,
    }


# ───────────────────────────── bot_loop (async) ─────────────────────────────

async def bot_loop(app: web.Application) -> None:
    """
    Ana trading döngüsü.
    """
    env_vars: Dict[str, str] = app["env_vars"]
    trading_objects: Dict[str, Any] = init_trading_objects(env_vars)

    symbol = Config.BINANCE_SYMBOL
    interval = Config.BINANCE_INTERVAL
    limit = Config.KLINES_LIMIT

    online_learner: OnlineLearner = trading_objects["online_learner"]
    fallback_model: FallbackModel = trading_objects["fallback_model"]
    trade_executor: TradeExecutor = trading_objects["trade_executor"]
    position_manager: PositionManager = trading_objects["position_manager"]
    risk_manager: RiskManager = trading_objects["risk_manager"]

    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")

    while True:
        try:
            pipeline_result = run_data_and_model_pipeline(
                trading_objects=trading_objects,
                symbol=symbol,
                interval=interval,
                limit=limit,
            )

            X_live = pipeline_result["X_live"]
            current_price = pipeline_result["current_price"]

            # Sinyal üret
            p_buy = compute_p_buy(
                online_learner=online_learner,
                fallback_model=fallback_model,
                X_live=X_live,
            )
            signal = generate_trading_signal(p_buy)

            # Pozisyon yönetimi
            manage_positions_for_signal(
                trade_executor=trade_executor,
                position_manager=position_manager,
                risk_manager=risk_manager,
                symbol=symbol,
                signal=signal,
                current_price=current_price,
            )

        except asyncio.CancelledError:
            system_logger.info("[MAIN] bot_loop cancelled by asyncio (shutdown).")
            break
        except Exception as e:
            system_logger.exception(f"[MAIN] Error in bot_loop iteration: {e}")

        await asyncio.sleep(Config.MAIN_LOOP_SLEEP)


# ───────────────────────────── aiohttp app setup ─────────────────────────────

def create_app() -> web.Application:
    # Env değişkenlerini yükle
    env_vars = load_environment_variables()

    # Logging setup
    setup_logger()
    system_logger.info(
        f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{os.getenv('PORT', '8080')} "
        f"(ENV={env_vars.get('ENV', 'unknown')})"
    )

    app = web.Application()
    app["env_vars"] = env_vars

    # Health endpoints
    app.router.add_get("/", health)
    app.router.add_get("/healthz", health)

    async def on_startup(app: web.Application):
        system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
        app["bot_task"] = asyncio.create_task(bot_loop(app))

    async def on_cleanup(app: web.Application):
        system_logger.info("[MAIN] Cleanup: cancelling bot_loop task...")
        bot_task = app.get("bot_task")
        if bot_task:
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def main() -> None:
    app = create_app()
    port = int(os.getenv("PORT", "8080"))

    # Cloud Run için signal handler zorunlu değil ama local debug’da iş görür
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            # Windows vs.
            pass

    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

