import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ------------------------------------------------------------
# Optional: load .env so ENABLE_PG_POS_LOG, MTF_INTERVALS, etc. are visible
# ------------------------------------------------------------
def _try_load_env() -> None:
    try:
        from config.load_env import load_environment_variables
        load_environment_variables()
    except Exception:
        pass


_try_load_env()


MODELS_DIR = os.getenv("MODELS_DIR", "models")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DATA_DIR = os.getenv("OFFLINE_CACHE_DIR", "data/offline_cache")

# Loader ile uyumlu default isimler (senin mevcut models/ içeriğin de bu şekilde)
MODEL_LONG_TEMPLATE = os.getenv("LSTM_MODEL_LONG_TEMPLATE", "lstm_long_{itv}.h5")
MODEL_SHORT_TEMPLATE = os.getenv("LSTM_MODEL_SHORT_TEMPLATE", "lstm_short_{itv}.h5")
SCALER_TEMPLATE_JOBLIB = os.getenv("LSTM_SCALER_TEMPLATE", "lstm_scaler_{itv}.joblib")


# ------------------------------------------------------------
# Robust CSV reader + schema normalize + numeric coercion
# ------------------------------------------------------------
_KLINE_COLS_12 = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def read_offline_csv(symbol: str, itv: str, limit: int = 50000) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, f"{symbol}_{itv}_6m.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    df = pd.read_csv(p, header=None, low_memory=False)

    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    # force 12-column schema
    if df.shape[1] >= 12:
        df = df.iloc[:, :12].copy()
        df.columns = _KLINE_COLS_12
    else:
        # rare: incomplete -> pad
        for i in range(df.shape[1], 12):
            df[i] = 0
        df = df.iloc[:, :12].copy()
        df.columns = _KLINE_COLS_12

    # numeric coercion
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # make ints safe
    for c in int_cols:
        try:
            df[c] = df[c].astype(int)
        except Exception:
            df[c] = df[c].fillna(0).astype(int)

    # cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    return df


# ------------------------------------------------------------
# Stable feature builder (main.py ile uyumlu, string riskini kaldırır)
# ------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # guarantee columns
    for c in _KLINE_COLS_12:
        if c not in df.columns:
            df[c] = 0

    # open_time/close_time -> float seconds
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            try:
                # ms epoch -> seconds float
                dt = pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
                # if parsing fails, fallback numeric
                if dt.isna().all():
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
                else:
                    df[col] = (dt.astype("int64") / 1e9).astype(float)
            except Exception:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # numeric coercion (critical fix for 'str - str')
    for c in ["open", "high", "low", "close", "volume",
              "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    for c in ["number_of_trades", "ignore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    close = df["close"].astype(float)

    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)

    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()

    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0.0)
    return df


# ------------------------------------------------------------
# Meta / schema helpers
# ------------------------------------------------------------
def load_meta(itv: str) -> Dict:
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def save_meta(itv: str, meta: Dict) -> None:
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_schema_and_seq(itv: str, meta: Dict) -> Tuple[List[str], int, int, float]:
    # LSTM schema varsa onu kullan, yoksa feature_schema
    schema = meta.get("lstm_feature_schema") or meta.get("feature_schema")
    if not (isinstance(schema, list) and schema and all(isinstance(x, str) for x in schema)):
        raise RuntimeError(f"Missing lstm_feature_schema/feature_schema in model_meta_{itv}.json")

    # seq_len override
    env_key = f"LSTM_SEQ_LEN_{itv}"
    seq_len = int(os.getenv(env_key, str(meta.get("seq_len", 50) or 50)))

    horizon = int(meta.get("label_horizon", 3) or 3)
    thr = float(meta.get("label_thr", 0.0005) or 0.0005)

    return schema, seq_len, horizon, thr


# ------------------------------------------------------------
# Labeling & sequencing
# ------------------------------------------------------------
def make_labels(close: np.ndarray, horizon: int, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    fut = np.roll(close, -horizon)
    ret = (fut - close) / np.maximum(np.abs(close), 1e-9)

    y_long = (ret > thr).astype(np.int32)
    y_short = (ret < -thr).astype(np.int32)

    # last horizon invalid
    y_long[-horizon:] = 0
    y_short[-horizon:] = 0
    return y_long, y_short


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n <= seq_len + 5:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32)

    xs = np.zeros((n - seq_len, seq_len, X.shape[1]), dtype=np.float32)
    ys = np.zeros((n - seq_len,), dtype=np.int32)

    for i in range(seq_len, n):
        xs[i - seq_len] = X[i - seq_len:i]
        ys[i - seq_len] = y[i]
    return xs, ys


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_lstm(input_dim: int, seq_len: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, input_dim))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)

    lr = float(os.getenv("LSTM_LR", "0.001"))
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return m


def _train_val_split(X: np.ndarray, y: np.ndarray, val_split: float) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    n_val = int(n * val_split)
    n_tr = max(1, n - n_val)
    return (X[:n_tr], y[:n_tr]), (X[n_tr:], y[n_tr:])


def train_one_interval(itv: str) -> None:
    meta = load_meta(itv)
    schema, seq_len, horizon, thr = load_schema_and_seq(itv, meta)

    raw = read_offline_csv(SYMBOL, itv)
    feat = build_features(raw)

    # ensure schema cols
    for c in schema:
        if c not in feat.columns:
            feat[c] = 0.0
    feat = feat[schema].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    close = pd.to_numeric(raw["close"], errors="coerce").fillna(0.0).astype(float).values
    y_long, y_short = make_labels(close, horizon=horizon, thr=thr)

    X = feat.values.astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    X_long, Y_long = make_sequences(Xs, y_long, seq_len=seq_len)
    X_short, Y_short = make_sequences(Xs, y_short, seq_len=seq_len)

    if len(X_long) < 50 or len(X_short) < 50:
        print(f"[{itv}] WARNING: Too few sequences (long={len(X_long)}, short={len(X_short)}). Skipping.")
        return

    val_split = float(os.getenv("LSTM_VAL_SPLIT", "0.2"))
    (XtrL, YtrL), (XvaL, YvaL) = _train_val_split(X_long, Y_long, val_split)
    (XtrS, YtrS), (XvaS, YvaS) = _train_val_split(X_short, Y_short, val_split)

    epochs = int(os.getenv("LSTM_EPOCHS", "30"))
    batch = int(os.getenv("LSTM_BATCH", "64"))
    patience = int(os.getenv("LSTM_PATIENCE", "5"))

    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=True
        )
    ]

    m_long = build_lstm(input_dim=X.shape[1], seq_len=seq_len)
    m_short = build_lstm(input_dim=X.shape[1], seq_len=seq_len)

    print(f"[{itv}] train LONG: tr={XtrL.shape} val={XvaL.shape} seq_len={seq_len}")
    histL = m_long.fit(
        XtrL, YtrL,
        validation_data=(XvaL, YvaL),
        epochs=epochs,
        batch_size=batch,
        callbacks=cb,
        verbose=2
    )

    print(f"[{itv}] train SHORT: tr={XtrS.shape} val={XvaS.shape} seq_len={seq_len}")
    histS = m_short.fit(
        XtrS, YtrS,
        validation_data=(XvaS, YvaS),
        epochs=epochs,
        batch_size=batch,
        callbacks=cb,
        verbose=2
    )

    # Best val auc
    best_auc_L = float(np.max(histL.history.get("val_auc", [0.0]) or [0.0]))
    best_auc_S = float(np.max(histS.history.get("val_auc", [0.0]) or [0.0]))

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, SCALER_TEMPLATE_JOBLIB.format(itv=itv))
    joblib.dump(scaler, scaler_path)

    # Save models
    long_path = os.path.join(MODELS_DIR, MODEL_LONG_TEMPLATE.format(itv=itv))
    short_path = os.path.join(MODELS_DIR, MODEL_SHORT_TEMPLATE.format(itv=itv))
    m_long.save(long_path)
    m_short.save(short_path)

    # Update meta (so loader uses them cleanly)
    meta["use_lstm_hybrid"] = True
    meta["seq_len"] = int(seq_len)
    meta["lstm_feature_schema"] = list(schema)
    meta["lstm_long_auc"] = best_auc_L
    meta["lstm_short_auc"] = best_auc_S

    # Optional: helpful trace
    meta["lstm_trained_at"] = pd.Timestamp.utcnow().isoformat()

    save_meta(itv, meta)

    print(f"[{itv}] saved scaler={scaler_path}")
    print(f"[{itv}] saved long={long_path} (best_val_auc={best_auc_L:.4f})")
    print(f"[{itv}] saved short={short_path} (best_val_auc={best_auc_S:.4f})")
    print(f"[{itv}] meta updated: models/model_meta_{itv}.json")


def parse_intervals() -> List[str]:
    # .env: MTF_INTERVALS=1m,3m,5m,15m,30m,1h
    v = os.getenv("MTF_INTERVALS", "1m,3m,5m,15m,30m,1h")
    itvs = [x.strip() for x in v.split(",") if x.strip()]
    return itvs


def main() -> None:
    # Tensorflow log noise azalt
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", os.getenv("TF_CPP_MIN_LOG_LEVEL", "2"))

    itvs = parse_intervals()
    print("[LSTM-ALL] intervals:", itvs)

    for itv in itvs:
        try:
            train_one_interval(itv)
        except Exception as e:
            print(f"[{itv}] ERROR: {e}")


if __name__ == "__main__":
    main()
