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
from binance.exceptions import BinanceAPIException


from config.settings import Config
from core.logger import setup_logger, system_logger
from core.notifier import Notifier

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner

from models.fallback_model import FallbackModel
from monitoring.performance_tracker import PerformanceTracker
from monitoring.alert_system import AlertSystem
from tg_bot.telegram_bot import TelegramBot
from env.load_env import load_environment_variables



from trading.risk_manager import RiskManager
from trading.position_manager import PositionManager
from trading.trade_executor import TradeExecutor


# ────────────────────────────── Basit health endpoint ──────────────────────────────

async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "service": "binance1-pro"})



# ───────────────────── Binance Futures client & trading obj init ───────────────────

def create_binance_futures_client(env_vars: Dict[str, str]) -> BinanceClient:
    api_key = env_vars.get("BINANCE_API_KEY") or os.getenv("BINANCE_API_KEY")
    api_secret = env_vars.get("BINANCE_API_SECRET") or os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        system_logger.warning(
            "[MAIN] BINANCE_API_KEY / BINANCE_API_SECRET not found in env. "
            "Futures client will be unauthenticated!"
        )

    client = BinanceClient(api_key, api_secret)
    # Testnet kullanacaksan:
    # client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return client


def init_trading_objects(env_vars: Dict[str, str]) -> Dict[str, Any]:
    """
    Tüm core trading objelerini initialize eder.
    Binance futures_account için self-check yapar ama HİÇBİR ZAMAN
    exception fırlatıp servisin çökmesine izin vermez.
    """
    system_logger.info("[MAIN] Initializing trading objects...")

    # Config log
    system_logger.info(
        "[MAIN] Config -> SYMBOL=%s | INTERVAL=%s | KLINES_LIMIT=%d | "
        "BUY_THRESHOLD=%.2f | SELL_THRESHOLD=%.2f | "
        "MAX_RISK_PER_TRADE=%.4f | MAX_DAILY_LOSS_PCT=%.4f | "
        "STOP_LOSS_PCT=%.4f | LEVERAGE=%dx",
        Config.BINANCE_SYMBOL,
        Config.BINANCE_INTERVAL,
        Config.KLINES_LIMIT,
        Config.BUY_THRESHOLD,
        Config.SELL_THRESHOLD,
        Config.MAX_RISK_PER_TRADE,
        Config.MAX_DAILY_LOSS_PCT,
        Config.STOP_LOSS_PCT,
        Config.DEFAULT_LEVERAGE,
    )

    # Binance client
    client = create_binance_futures_client(env_vars)

    # --- SELF-CHECK: futures_account (ASLA raise etme!) ---
    try:
        account = client.futures_account()
        total_wallet_balance = account.get("totalWalletBalance")
        system_logger.info(
            "[SELF-CHECK] Binance futures_account OK | "
            "totalWalletBalance=%s",
            total_wallet_balance,
        )
    except BinanceAPIException as e:
        # Cloud Run IP'leri bazen HTML error sayfası döndüğü için burada yakalıyoruz
        msg = str(e)
        if "<!DOCTYPE html>" in msg:
            system_logger.error(
                "[SELF-CHECK] Binance futures_account HTML error page döndü. "
                "Muhtemelen Cloud Run IP aralığına özel bir engel/koruma var. "
                "Detay: %s",
                msg[:500],
            )
        else:
            system_logger.error(
                "[SELF-CHECK] Binance futures_account BinanceAPIException: %s",
                msg,
            )
        # ÖNEMLİ: burada asla raise etmiyoruz, client yine de kullanılabilir
    except Exception as e:
        system_logger.exception(
            "[SELF-CHECK] Binance futures_account beklenmeyen hata: %s", e
        )
        # Yine raise yok; sadece log.

    # Data pipeline objesi (public HTTP klines)
    data_loader = DataLoader(env_vars)

    # Online model + fallback
    online_learner = OnlineLearner(
        model_dir="models",
        base_model_name="online_model",
        n_classes=2,
        logger=system_logger,
    )
    fallback_model = FallbackModel(default_proba=0.5)

    # Risk & pozisyon yönetimi
    risk_manager = RiskManager(
        max_risk_per_trade=Config.MAX_RISK_PER_TRADE,
        max_daily_loss_pct=Config.MAX_DAILY_LOSS_PCT,
        state_file=os.path.join("logs", "risk_state.json"),
    )
    position_manager = PositionManager()

    # Trade executor
    trade_executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        position_manager=position_manager,
    )

    # Monitoring & Telegram
    performance_tracker = PerformanceTracker()
    telegram_bot = TelegramBot()  # TOKEN yoksa kendi kendine pasif kalıyor
    notifier = Notifier(telegram_bot=telegram_bot)
    alert_system = AlertSystem(notifier=notifier)

    objects = {
        "client": client,
        "data_loader": data_loader,
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

        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] == 2:
            p_buy = float(probs[-1, 1])
        else:
            p_buy = float(probs[-1])

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
        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[1] == 2:
            p_buy = float(probs[-1, 1])
        else:
            p_buy = float(probs[-1])

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


# ───────────────────────────── LONG / SHORT yönetimi ─────────────────────────────

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
    - HOLD: hiçbir şey yapma
    """

    signal = signal.upper()
    long_pos = position_manager.get_position(symbol, "LONG")
    short_pos = position_manager.get_position(symbol, "SHORT")

    # Günlük zarar limiti aşıldıysa: yeni trade açma, tüm pozisyonları kapat
    if risk_manager.trading_halted:
        system_logger.warning(
            "[MAIN] Trading halted for today by risk manager (MAX_DAILY_LOSS reached)."
        )
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


# ─────────────────────── data + model pipeline (etiket üretimli) ───────────────────────

def run_data_and_model_pipeline(
    trading_objects: Dict[str, Any],
    symbol: str,
    interval: str,
    limit: int,
) -> Dict[str, Any]:
    """
    1) Binance'ten kline verisini çek
    2) Feature engineering
    3) Basit label üretimi (return_1'in bir bar sonrası işaretine göre)
    4) Anomali filtresi
    5) Online model initial_fit / partial_update
    6) Son bar için X_live, current_price döndür
    """
    data_loader: DataLoader = trading_objects["data_loader"]
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
    fe = FeatureEngineer(df_raw)
    df_features = fe.transform()
    system_logger.info(
        f"[FE] Features DF shape: {df_features.shape}, "
        f"columns={list(df_features.columns)}"
    )

    # 3) Basit label üretimi: bir sonraki barın return_1 > 0 ise 1, else 0
    df_features["label"] = (df_features["return_1"].shift(-1) > 0).astype(int)
    df_features = df_features.dropna().copy()

    # 4) Anomali filtresi
    anom_detector = AnomalyDetector(df_features, logger=system_logger)
    df_clean = anom_detector.detect_and_handle_anomalies()
    system_logger.info(
        f"[ANOM] After anomaly filter: {df_clean.shape[0]} rows remain."
    )

    if df_clean.shape[0] < 200:
        raise RuntimeError("Not enough samples after anomaly filtering.")

    # Feature / target ayrımı
    feature_cols = [c for c in df_clean.columns if c not in ("label",)]
    X_all = df_clean[feature_cols]
    y_all = df_clean["label"]

    # Son satır -> canlı tahmin için
    X_live = X_all.tail(1)
    current_price = float(df_clean["close"].iloc[-1])

    # Eğitim verisi -> son satır hariç
    X_train = X_all.iloc[:-1]
    y_train = y_all.iloc[:-1]

    # 5) Online learner initial_fit / partial_update
    # Sınıf içinde flag yok, dinamik attribute kullanıyoruz
    has_fitted = getattr(online_learner, "has_fitted_", False)

    if not has_fitted:
        system_logger.info(
            f"[ONLINE] initial_fit with {X_train.shape[0]} samples, {X_train.shape[1]} features."
        )
        online_learner.initial_fit(X_train, y_train)
        online_learner.has_fitted_ = True
    else:
        # Son 100 bar ile güncelle
        X_chunk = X_train.tail(100)
        y_chunk = y_train.tail(100)
        system_logger.info(
            f"[ONLINE] partial_update with {X_chunk.shape[0]} samples, {X_chunk.shape[1]} features."
        )
        online_learner.partial_update(X_chunk, y_chunk)

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

            # Pozisyon yönetimi (Futures LONG/SHORT)
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

