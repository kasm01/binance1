# training/offline_pretrain_six_months.py

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from sklearn.metrics import accuracy_score

from core.logger import setup_logger, system_logger
from data.feature_engineering import FeatureEngineer
from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner


# ───────────────────── Binance veri çekme (6 ay) ─────────────────────

def _interval_to_ms(interval: str) -> int:
    """
    Binance interval string -> milisaniye
    Sadece 1m, 5m, 15m, 1h vs. için basit mapping.
    """
    mapping = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
    }
    return mapping.get(interval, 60 * 1000)  # default 1m


def fetch_futures_klines_range(
    client: Client,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 180,
    limit_per_call: int = 1500,
) -> pd.DataFrame:
    """
    Son 'days' gün için Binance Futures kline çeker (paginate).

    Not:
      - Bu fonksiyon sadece OFFLINE eğitim için.
      - Ağırlık limitine çok yüklenmemek için limit_per_call 1500 tutuldu.
    """
    end_ts = int(datetime.utcnow().timestamp() * 1000)
    start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    step = limit_per_call * _interval_to_ms(interval)

    all_klines = []
    current = start_ts

    system_logger.info(
        f"[OFFLINE] Fetching futures klines for {symbol} ({interval}), "
        f"days={days}, from={start_ts} to={end_ts}"
    )

    while current < end_ts:
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_per_call,
                startTime=current,
            )
        except Exception as e:
            system_logger.exception(f"[OFFLINE] Error fetching klines: {e}")
            break

        if not klines:
            system_logger.warning(
                f"[OFFLINE] No more klines returned at startTime={current}. Breaking loop."
            )
            break

        all_klines.extend(klines)

        last_open_time = klines[-1][0]
        next_start = last_open_time + _interval_to_ms(interval)

        if next_start <= current:
            # Güvenlik: sonsuz loop olmasın
            break

        current = next_start

    if not all_klines:
        raise RuntimeError("[OFFLINE] No klines fetched from Binance.")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(all_klines, columns=cols)

    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades"]

    for c in float_cols:
        df[c] = df[c].astype(float)
    for c in int_cols:
        df[c] = df[c].astype(int)

    system_logger.info(
        f"[OFFLINE] Fetched DF shape: {df.shape}, from ts={df['open_time'].iloc[0]} "
        f"to ts={df['open_time'].iloc[-1]}"
    )
    return df


# ───────────────────── FE + label + anomaly + split ─────────────────────

def build_training_and_validation_sets(
    df_raw: pd.DataFrame,
    valid_ratio: float = 0.2,
):
    """
    - FeatureEngineer
    - Label (next-bar up/down)
    - AnomalyDetector
    - Train / Valid split (zaman sıralı)
    """
    system_logger.info(
        f"[OFFLINE] Raw DF shape: {df_raw.shape}, columns={list(df_raw.columns)}"
    )

    fe = FeatureEngineer(df_raw)
    df_features = fe.transform()
    system_logger.info(
        f"[OFFLINE] Features DF shape: {df_features.shape}, "
        f"columns={list(df_features.columns)}"
    )

    # Label: bir sonraki barın return_1 > 0 ise 1, değilse 0
    df_features["label"] = (df_features["return_1"].shift(-1) > 0).astype(int)
    df_features = df_features.dropna().copy()

    # Anomali filtreleme
    anom = AnomalyDetector(df_features, logger=system_logger)
    df_clean = anom.detect_and_handle_anomalies()
    system_logger.info(
        f"[OFFLINE] After anomaly filter: {df_clean.shape[0]} rows remain."
    )

    if df_clean.shape[0] < 1000:
        raise RuntimeError(
            f"[OFFLINE] Too few samples after anomaly filtering: {df_clean.shape[0]}"
        )

    feature_cols = [c for c in df_clean.columns if c not in ("label",)]
    X_all = df_clean[feature_cols]
    y_all = df_clean["label"]

    n = len(X_all)
    split_idx = int(n * (1 - valid_ratio))

    X_train = X_all.iloc[:split_idx]
    y_train = y_all.iloc[:split_idx]

    X_valid = X_all.iloc[split_idx:]
    y_valid = y_all.iloc[split_idx:]

    system_logger.info(
        f"[OFFLINE] Train size={len(X_train)}, Valid size={len(X_valid)}, "
        f"Features={X_train.shape[1]}"
    )

    return X_train, y_train, X_valid, y_valid


# ───────────────────── 2000 iter offline training ─────────────────────

def run_offline_training(
    client: Client,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    days: int = 180,
    n_iterations: int = 2000,
    batch_size: int = 256,
):
    """
    Ana offline eğitim pipeline'ı:

    1) 6 aylık veri çek
    2) Feature + label + anomaly + train/valid
    3) OnlineLearner.initial_fit
    4) 2000 kez random mini-batch ile partial_update
    5) Her iterasyonda valid accuracy ve metrikleri CSV'ye logla
    """

    # 1) Veri
    df_raw = fetch_futures_klines_range(
        client=client,
        symbol=symbol,
        interval=interval,
        days=days,
    )

    # 2) Dataset
    X_train, y_train, X_valid, y_valid = build_training_and_validation_sets(df_raw)

    # 3) Online learner init
    online_learner = OnlineLearner(
        model_dir="models",
        base_model_name="online_model",
        n_classes=2,
        logger=system_logger,
    )

    system_logger.info(
        f"[OFFLINE] initial_fit on full train set: {X_train.shape[0]} samples."
    )
    online_learner.initial_fit(X_train, y_train)

    # Log dosyası
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "offline_training_6m.csv")

    # CSV header (yoksa yaz)
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "iteration,total_iters,batch_size,train_indices,"
                "valid_accuracy,train_accuracy\n"
            )

    n_train = len(X_train)
    idx_arr = np.arange(n_train)

    system_logger.info(
        f"[OFFLINE] Starting {n_iterations} iterations of partial_update..."
    )

    for it in range(1, n_iterations + 1):
        # Random mini-batch indexleri
        batch_idx = np.random.choice(idx_arr, size=min(batch_size, n_train), replace=False)
        X_batch = X_train.iloc[batch_idx]
        y_batch = y_train.iloc[batch_idx]

        # partial_update
        online_learner.partial_update(X_batch, y_batch)

        # Valid & train accuracy ölç
        try:
            # valid
            y_val_pred = online_learner.model.predict(X_valid[online_learner.feature_columns])
            val_acc = accuracy_score(y_valid, y_val_pred)

            # train batch
            y_tr_pred = online_learner.model.predict(X_batch[online_learner.feature_columns])
            tr_acc = accuracy_score(y_batch, y_tr_pred)
        except Exception as e:
            system_logger.exception(f"[OFFLINE] Evaluation error at iter={it}: {e}")
            val_acc = np.nan
            tr_acc = np.nan

        # Log'a yaz
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{it},{n_iterations},{len(batch_idx)},"
                f"\"{batch_idx.tolist()}\",{val_acc:.6f},{tr_acc:.6f}\n"
            )

        if it % 100 == 0 or it == n_iterations:
            system_logger.info(
                f"[OFFLINE] Iter={it}/{n_iterations} | "
                f"valid_acc={val_acc:.4f} | train_acc={tr_acc:.4f}"
            )

    system_logger.info(
        f"[OFFLINE] Finished {n_iterations} iterations. "
        f"Model saved as models/online_model.joblib and "
        f"logs written to {log_path}"
    )


# ───────────────────── main ─────────────────────

def main():
    # Logger
    setup_logger()
    system_logger.info("📦 [OFFLINE] Starting 6-month offline pretraining with 2000 iters")

    # .env yükle
    load_dotenv(dotenv_path=".env")

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_API_KEY / BINANCE_API_SECRET bulunamadı. "
            "Lütfen .env dosyanı kontrol et."
        )

    symbol = os.getenv("SYMBOL", "BTCUSDT")
    interval = os.getenv("INTERVAL", "1m")

    client = Client(api_key, api_secret)

    run_offline_training(
        client=client,
        symbol=symbol,
        interval=interval,
        days=180,
        n_iterations=2000,
        batch_size=256,
    )

    system_logger.info("✅ [OFFLINE] Pretraining completed successfully.")


if __name__ == "__main__":
    main()
