# main.py
import asyncio
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from aiohttp import web

from config.load_env import load_environment_variables
from config.settings import Settings

from core.logger import system_logger, error_logger, trade_logger
from core.exceptions import (
    DataFetchException,
    DataProcessingException,
    FeatureEngineeringException,
    LabelGenerationException,
    ModelTrainingException,
    OnlineLearningException,
    PredictionException,
    BotLoopException,
)
from core.notifier import Notifier

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner
from data.batch_learning import BatchLearner

from models.ensemble_model import EnsembleModel
from models.fallback_model import FallbackModel

from trading.risk_manager import RiskManager
from trading.capital_manager import CapitalManager
from trading.position_manager import PositionManager
from trading.trade_executor import TradeExecutor

# ============================================================
# Global / Config
# ============================================================

ENV_VARS: Dict[str, Any] = load_environment_variables()

SYMBOL = ENV_VARS.get("SYMBOL", "BTCUSDT")
INTERVAL = ENV_VARS.get("INTERVAL", "1m")
KLINE_LIMIT = int(ENV_VARS.get("KLINE_LIMIT", 1000))

# Trade / risk konfigürasyonu
POLL_INTERVAL_SECONDS = int(ENV_VARS.get("POLL_INTERVAL_SECONDS", 60))
NOTIONAL_CAPITAL = float(ENV_VARS.get("NOTIONAL_CAPITAL", 1000.0))
RISK_PER_TRADE = float(ENV_VARS.get("RISK_PER_TRADE", 0.01))
MAX_OPEN_POSITIONS_PER_SIDE = int(ENV_VARS.get("MAX_OPEN_POSITIONS_PER_SIDE", 1))

BUY_THRESHOLD = float(ENV_VARS.get("BUY_THRESHOLD", 0.60))
SELL_THRESHOLD = float(ENV_VARS.get("SELL_THRESHOLD", 0.40))

# Global modeller
ENSEMBLE_MODEL: Optional[EnsembleModel] = None
FALLBACK_MODEL: FallbackModel = FallbackModel(default_proba=0.5)

# Trading objeleri (tek instance)
RISK_MANAGER = RiskManager(max_risk_per_trade=RISK_PER_TRADE)
CAPITAL_MANAGER = CapitalManager(total_capital=NOTIONAL_CAPITAL)
POSITION_MANAGER = PositionManager()
TRADE_EXECUTOR: Optional[TradeExecutor] = None

# Notifier (ileride Telegram bot bağlanabilir)
NOTIFIER = Notifier(telegram_bot=None)


# ============================================================
# Data Pipeline
# ============================================================

def run_data_pipeline(
    symbol: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """
    1) Kline verisini yükler (Redis cache + Binance)
    2) Feature engineering uygular
    3) Anomali tespiti (IsolationForest)
    4) Label kolonunu üretir
    5) Model için hazır final dataframe döner
    """
    system_logger.info(
        f"[DATA] Starting data pipeline for {symbol} ({interval}, limit={limit})"
    )

    # ---------------------------
    # 1) Kline verisi yükleme
    # ---------------------------
    try:
        loader = DataLoader(env_vars=ENV_VARS)
        system_logger.info("[DATA] Using DataLoader.load_and_cache_klines")
        df_raw = loader.load_and_cache_klines(symbol=symbol, interval=interval, limit=limit)
        system_logger.info(f"[DATA] Raw DF shape: {df_raw.shape}")
    except Exception as e:
        error_logger.exception(f"[DATA] Error while loading klines: {e}")
        raise DataFetchException(f"Failed to load klines: {e}") from e

    # ---------------------------
    # 2) Feature engineering
    # ---------------------------
    try:
        fe = FeatureEngineer(df_raw)
        features_df = fe.build_features()
        system_logger.info(
            f"[FE] Features DF shape: {features_df.shape}, "
            f"columns={list(features_df.columns)}"
        )
    except Exception as e:
        error_logger.exception(f"[FE] Feature engineering error: {e}")
        raise FeatureEngineeringException(f"Feature engineering failed: {e}") from e

    # ---------------------------
    # 3) Anomali tespiti / filtresi
    # ---------------------------
    try:
        anomaly_detector = AnomalyDetector(features_df=features_df)
        system_logger.info(
            f"[ANOM] Running IsolationForest on {features_df.shape[0]} samples, "
            f"{features_df.select_dtypes('number').shape[1]} numeric features."
        )
        clean_features_df = anomaly_detector.detect_and_handle_anomalies()
    except Exception as e:
        system_logger.warning(
            f"[DATA] Anomaly detection failed, using original features_df: {e}"
        )
        clean_features_df = features_df

    # ---------------------------
    # 4) Label üretimi
    # ---------------------------
    try:
        final_df = _generate_labels(clean_features_df)
        system_logger.info(
            f"[FE] Final Features DF shape (with label): {final_df.shape}, "
            f"columns={list(final_df.columns)}"
        )
    except Exception as e:
        error_logger.exception(f"[LABEL] Label generation error: {e}")
        raise LabelGenerationException(f"Label generation failed: {e}") from e

    return final_df


def _generate_labels(df: pd.DataFrame, horizon_col: str = "return_5") -> pd.DataFrame:
    """
    Basit label üretimi:
      - horizon_col > 0 => 1 (BUY)
      - horizon_col <= 0 => 0 (SELL/HOLD)
    """
    if horizon_col not in df.columns:
        raise LabelGenerationException(
            f"Label horizon_col '{horizon_col}' not found in dataframe."
        )

    df = df.copy()
    df["label"] = (df[horizon_col] > 0).astype(int)
    return df.dropna(subset=["label"])


def _split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    'label' kolonunu y'de, geri kalan numeric feature'ları X'te toplar.
    """
    if "label" not in df.columns:
        raise ModelTrainingException("Column 'label' not found in dataframe.")

    feature_cols = [
        col for col in df.columns
        if col not in ("label", "open_time", "close_time")
    ]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    return X, y, feature_cols


# ============================================================
# Model Training (Batch + Online + Ensemble)
# ============================================================

def build_ensemble_from_batch(batch_model) -> EnsembleModel:
    """
    Batch model (ör: RandomForest) ile basit bir EnsembleModel kur.
    İleride LGBM / CatBoost / LSTM de eklenebilir.
    """
    from sklearn.base import ClassifierMixin

    if batch_model is None or not isinstance(batch_model, ClassifierMixin):
        system_logger.warning(
            "[ENSEMBLE] Given batch_model is not a valid sklearn classifier; "
            "EnsembleModel will be empty."
        )
        return EnsembleModel(estimators=None)

    estimators = [("rf", batch_model)]
    ensemble = EnsembleModel(estimators=estimators)
    system_logger.info("[ENSEMBLE] EnsembleModel built with RandomForest (rf).")
    return ensemble


def train_models_and_update_state(
    clean_df: pd.DataFrame,
) -> Tuple[OnlineLearner, list]:
    """
    1) BatchLearner ile batch model train
    2) Batch modelden EnsembleModel oluştur (global ENSEMBLE_MODEL güncellenir)
    3) OnlineLearner'i batch verisi ile initial_fit + kısa partial_update
    """
    global ENSEMBLE_MODEL

    X, y, feature_cols = _split_features_labels(clean_df)

    # ---------------------------
    # 1) Batch training
    # ---------------------------
    batch_learner = BatchLearner(
        model_dir="models",
        base_model_name="batch_model",
    )

    system_logger.info(
        f"[MODEL] Training batch model on {len(X)} samples, {X.shape[1]} features. "
        f"Using {len(feature_cols)} feature columns."
    )

    batch_learner.train(X, y)
    batch_model = getattr(batch_learner, "model", None)

    if batch_model is not None:
        ENSEMBLE_MODEL = build_ensemble_from_batch(batch_model)
    else:
        system_logger.warning(
            "[MODEL] BatchLearner has no 'model' attribute; EnsembleModel will be empty."
        )
        ENSEMBLE_MODEL = EnsembleModel(estimators=None)

    # ---------------------------
    # 2) Online model init/update
    # ---------------------------
    online_learner = OnlineLearner(
        model_dir="models",
        base_model_name="online_model",
        n_classes=2,
    )
    online_learner.feature_columns = feature_cols
    system_logger.info(
        f"[ONLINE] feature_columns set with {len(feature_cols)} columns."
    )

    online_learner.initial_fit(X, y)
    system_logger.info("[ONLINE] initial_fit completed successfully.")
    online_learner.save()
    system_logger.info("[ONLINE] Online model saved to models/online_model.joblib")

    # Son 100 örnek ile küçük partial_update
    if len(X) > 100:
        recent_X = X.tail(100)
        recent_y = y.tail(100)
    else:
        recent_X = X
        recent_y = y

    online_learner.partial_update(recent_X, recent_y)
    system_logger.info("[ONLINE] partial_update completed successfully.")
    online_learner.save()
    system_logger.info("[ONLINE] Online model saved to models/online_model.joblib")

    return online_learner, feature_cols


# ============================================================
# Signal Generation (Online + Ensemble + Fallback)
# ============================================================

def generate_trading_signal(
    X_last: pd.DataFrame,
    feature_cols: list,
    online_learner: Optional[OnlineLearner] = None,
    ensemble_model: Optional[EnsembleModel] = None,
    fallback_model: Optional[FallbackModel] = None,
) -> Tuple[str, float, str]:
    """
    Öncelik: online → ensemble → fallback

    :return: (signal, p_buy, source)
    """
    if fallback_model is None:
        fallback_model = FALLBACK_MODEL

    proba = None
    source = "fallback"

    # 1) ONLINE MODEL
    try:
        if online_learner is not None:
            X_input = X_last[feature_cols]
            proba = online_learner.predict_proba(X_input)  # (1, 2) [p0, p1]
            source = "online"
    except Exception as e:
        system_logger.warning(
            f"[SIGNAL] Online model prediction failed: {e}. Will try ensemble.",
            exc_info=True,
        )
        proba = None

    # 2) ENSEMBLE MODEL
    if proba is None and ensemble_model is not None:
        try:
            X_input = X_last[feature_cols]
            proba = ensemble_model.predict_proba(X_input)
            source = "ensemble"
        except Exception as e:
            system_logger.warning(
                f"[SIGNAL] EnsembleModel prediction failed: {e}. Falling back.",
                exc_info=True,
            )
            proba = None

    # 3) FALLBACK MODEL
    if proba is None:
        X_input = X_last[feature_cols].values
        proba = fallback_model.predict_proba(X_input)
        source = "fallback"

    p_buy = float(proba[0, 1])

    system_logger.info(
        f"[SIGNAL] Source={source}, p_buy={p_buy:.4f} "
        f"(BUY_THRESHOLD={BUY_THRESHOLD}, SELL_THRESHOLD={SELL_THRESHOLD})"
    )

    if p_buy >= BUY_THRESHOLD:
        signal = "BUY"
    elif p_buy <= SELL_THRESHOLD:
        signal = "SELL"
        # short sinyal olarak yorumlayacağız
    else:
        signal = "HOLD"

    system_logger.info(f"[SIGNAL] Generated trading signal: {signal}")
    return signal, p_buy, source


# ============================================================
# Trading Layer (LONG/SHORT + Position/Capital + TradeExecutor)
# ============================================================

def _get_latest_price_from_row(row: pd.Series) -> float:
    # Varsayılan olarak 'close' sütununu kullan
    return float(row.get("close", row.get("price", 0.0)))


def _ensure_trade_executor() -> TradeExecutor:
    global TRADE_EXECUTOR
    if TRADE_EXECUTOR is None:
        TRADE_EXECUTOR = TradeExecutor(env_vars=ENV_VARS)
    return TRADE_EXECUTOR


def _find_open_positions(symbol: str, side: str) -> Dict[str, Dict[str, Any]]:
    """
    Belirli symbol + side (LONG/SHORT) için açık pozisyonları döner.
    """
    open_pos = {}
    for pos_id, pos in POSITION_MANAGER.get_open_positions().items():
        if pos.get("symbol") == symbol and pos.get("side", "").upper() == side.upper():
            open_pos[pos_id] = pos
    return open_pos


def execute_trading_logic(
    signal: str,
    X_last: pd.DataFrame,
) -> None:
    """
    Signal → LONG/SHORT pozisyon aç/kapat + loglama + TradeExecutor emirleri.
    """
    if X_last.empty:
        system_logger.warning("[TRADE] X_last is empty; skipping trade logic.")
        return

    row = X_last.iloc[0]
    current_price = _get_latest_price_from_row(row)

    if current_price <= 0:
        system_logger.warning("[TRADE] Invalid current_price <= 0; skipping.")
        return

    executor = _ensure_trade_executor()

    # Açık pozisyonları çek
    open_longs = _find_open_positions(SYMBOL, "LONG")
    open_shorts = _find_open_positions(SYMBOL, "SHORT")

    trade_logger.info(
        f"[POSITIONS] Before signal={signal} | "
        f"LONG={len(open_longs)}, SHORT={len(open_shorts)}, "
        f"all={POSITION_MANAGER.get_open_positions()}"
    )

    # Kullanılacak notional risk büyüklüğü
    position_notional = NOTIONAL_CAPITAL * RISK_PER_TRADE
    qty = position_notional / current_price  # BTC cinsinden miktar

    # Miktarı biraz yuvarla (örneğin BTC için 0.001 hassasiyet)
    qty = float(f"{qty:.3f}")

    if signal == "BUY":
        # Önce açık SHORT pozisyonları kapat
        for pos_id, pos in open_shorts.items():
            try:
                executor.close_position(
                    symbol=SYMBOL,
                    quantity=pos["qty"],
                    position_side="SHORT",
                )
                pnl = POSITION_MANAGER.close_position(pos_id, exit_price=current_price)
                CAPT_RETURN = position_notional + (pnl or 0.0)
                CAPITAL_MANAGER.release(CAPT_RETURN)
                trade_logger.info(
                    f"[TRADE] Closed SHORT pos_id={pos_id}, pnl={pnl}, "
                    f"price={current_price}"
                )
            except Exception as e:
                error_logger.error(
                    f"[TRADE] Error closing SHORT position {pos_id}: {e}",
                    exc_info=True,
                )

        # Eğer LONG pozisyon sayısı limiti aşmıyorsa yeni LONG aç
        open_longs = _find_open_positions(SYMBOL, "LONG")
        if len(open_longs) < MAX_OPEN_POSITIONS_PER_SIDE:
            if RISK_MANAGER.check_risk(capital=NOTIONAL_CAPITAL, position_qty=position_notional):
                try:
                    order = executor.create_market_order(
                        symbol=SYMBOL,
                        side="BUY",
                        quantity=qty,
                        position_side="LONG",
                        reduce_only=False,
                    )
                    pos_id = POSITION_MANAGER.open_position(
                        symbol=SYMBOL,
                        qty=qty,
                        side="LONG",
                        price=current_price,
                    )
                    CAPITAL_MANAGER.allocate(RISK_PER_TRADE)
                    trade_logger.info(
                        f"[TRADE] Opened LONG pos_id={pos_id}, "
                        f"qty={qty}, price={current_price}, order={order}"
                    )
                except Exception as e:
                    error_logger.error(
                        f"[TRADE] Error opening LONG position: {e}",
                        exc_info=True,
                    )
            else:
                system_logger.warning(
                    "[TRADE] RiskManager rejected LONG trade due to risk limit."
                )

    elif signal == "SELL":
        # Önce açık LONG pozisyonları kapat
        for pos_id, pos in open_longs.items():
            try:
                executor.close_position(
                    symbol=SYMBOL,
                    quantity=pos["qty"],
                    position_side="LONG",
                )
                pnl = POSITION_MANAGER.close_position(pos_id, exit_price=current_price)
                CAPT_RETURN = position_notional + (pnl or 0.0)
                CAPITAL_MANAGER.release(CAPT_RETURN)
                trade_logger.info(
                    f"[TRADE] Closed LONG pos_id={pos_id}, pnl={pnl}, "
                    f"price={current_price}"
                )
            except Exception as e:
                error_logger.error(
                    f"[TRADE] Error closing LONG position {pos_id}: {e}",
                    exc_info=True,
                )

        # Yeni SHORT aç (limit dahilinde)
        open_shorts = _find_open_positions(SYMBOL, "SHORT")
        if len(open_shorts) < MAX_OPEN_POSITIONS_PER_SIDE:
            if RISK_MANAGER.check_risk(capital=NOTIONAL_CAPITAL, position_qty=position_notional):
                try:
                    order = executor.create_market_order(
                        symbol=SYMBOL,
                        side="SELL",
                        quantity=qty,
                        position_side="SHORT",
                        reduce_only=False,
                    )
                    pos_id = POSITION_MANAGER.open_position(
                        symbol=SYMBOL,
                        qty=qty,
                        side="SHORT",
                        price=current_price,
                    )
                    CAPITAL_MANAGER.allocate(RISK_PER_TRADE)
                    trade_logger.info(
                        f"[TRADE] Opened SHORT pos_id={pos_id}, "
                        f"qty={qty}, price={current_price}, order={order}"
                    )
                except Exception as e:
                    error_logger.error(
                        f"[TRADE] Error opening SHORT position: {e}",
                        exc_info=True,
                    )
            else:
                system_logger.warning(
                    "[TRADE] RiskManager rejected SHORT trade due to risk limit."
                )

    else:
        # HOLD: Şimdilik açık pozisyonlara dokunma
        trade_logger.info("[TRADE] HOLD signal – no position changes.")

    # Son durum logu
    trade_logger.info(
        f"[POSITIONS] After signal={signal} | "
        f"all={POSITION_MANAGER.get_open_positions()}"
    )


# ============================================================
# Async Bot Loop
# ============================================================

async def run_data_and_model_pipeline() -> None:
    """
    Her döngüde:
      1) Data pipeline
      2) Model training (batch + online + ensemble update)
      3) Sinyal üretimi
      4) Trading logic (LONG/SHORT + TradeExecutor)
    """
    try:
        clean_df = run_data_pipeline(
            symbol=SYMBOL,
            interval=INTERVAL,
            limit=KLINE_LIMIT,
        )

        online_learner, feature_cols = train_models_and_update_state(clean_df)

        # Son satırı al (en güncel bar)
        X_last = clean_df.tail(1)

        signal, p_buy, source = generate_trading_signal(
            X_last=X_last,
            feature_cols=feature_cols,
            online_learner=online_learner,
            ensemble_model=ENSEMBLE_MODEL,
            fallback_model=FALLBACK_MODEL,
        )

        # Sinyali trading katmanına gönder
        execute_trading_logic(signal=signal, X_last=X_last)

    except BotLoopException as e:
        error_logger.error(f"[BOT] BotLoopException in pipeline: {e}", exc_info=True)
        NOTIFIER.notify_error(f"BotLoopException: {e}")
    except Exception as e:
        error_logger.error(f"[BOT] Unexpected error in pipeline: {e}", exc_info=True)
        NOTIFIER.notify_error(f"Unexpected bot error: {e}")


async def bot_loop() -> None:
    """
    Arka planda sonsuz döngü:
      - data+model+trade pipeline
      - POLL_INTERVAL_SECONDS kadar uyku
    """
    system_logger.info("🚀 [BOT] Binance1-Pro core bot_loop started.")
    while True:
        await run_data_and_model_pipeline()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


# ============================================================
# HTTP Server (aiohttp)
# ============================================================

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "symbol": SYMBOL})


async def on_startup(app: web.Application) -> None:
    system_logger.info(
        f"🌐 [MAIN] Starting HTTP server on 0.0.0.0:{Settings.PORT} (ENV={Settings.ENVIRONMENT})"
    )
    system_logger.info("🔁 [MAIN] Starting background bot_loop task...")
    app["bot_task"] = asyncio.create_task(bot_loop())


async def on_cleanup(app: web.Application) -> None:
    task = app.get("bot_task")
    if task:
        system_logger.info("[MAIN] Cleanup: cancelling bot_loop task...")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", handle_health)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app


def main() -> None:
    port = int(os.environ.get("PORT", Settings.PORT))
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

