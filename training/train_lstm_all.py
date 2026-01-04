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

# Projedeki mevcut feature builder'ı kullanmak en iyisi:
# Eğer sende training/offline_train_hybrid.py gibi bir yer varsa onu import et.
# Yoksa main.py içindeki build_features'ı import etmeyi deneyebiliriz.
try:
    from training.offline_train_hybrid import build_features  # varsa
except Exception:
    from main import build_features  # fallback (main.py import döngüsü yapmıyorsa)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DATA_DIR = os.getenv("OFFLINE_CACHE_DIR", "data/offline_cache")

# === BURAYI LOADERA GÖRE AYARLAYACAĞIZ ===
# Örnekler:
#  - models/lstm_model_5m_long.h5
#  - models/lstm_long_5m.keras
# Hangisi sende varsa onu seç.
MODEL_LONG_TEMPLATE = os.getenv("LSTM_MODEL_LONG_TEMPLATE", "lstm_model_{itv}_long.h5")
MODEL_SHORT_TEMPLATE = os.getenv("LSTM_MODEL_SHORT_TEMPLATE", "lstm_model_{itv}_short.h5")

SCALER_TEMPLATE_JOBLIB = "lstm_scaler_{itv}.joblib"

def read_offline_csv(symbol: str, itv: str, limit: int = 50000) -> pd.DataFrame:
    p = os.path.join(DATA_DIR, f"{symbol}_{itv}_6m.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, header=None, low_memory=False)
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    # main.py normalize mantığı: 12 kolon
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_volume","taker_buy_quote_volume","ignore",
    ]
    if df.shape[1] >= 12:
        df = df.iloc[:, :12].copy()
        df.columns = cols
    return df

def load_meta_schema(itv: str) -> Tuple[List[str], List[str], int]:
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f) or {}

    # LSTM schema varsa onu kullan
    lstm_schema = meta.get("lstm_feature_schema") or meta.get("feature_schema")
    if not (isinstance(lstm_schema, list) and lstm_schema):
        raise RuntimeError(f"Missing lstm_feature_schema/feature_schema in {meta_path}")

    # seq_len env override
    env_key = f"LSTM_SEQ_LEN_{itv}"
    seq_len = int(os.getenv(env_key, str(meta.get("seq_len", 50) or 50)))

    # label params (meta ile tutarlı)
    horizon = int(meta.get("label_horizon", 3) or 3)
    thr = float(meta.get("label_thr", 0.0005) or 0.0005)

    return lstm_schema, [str(x) for x in lstm_schema], seq_len

def make_labels(close: np.ndarray, horizon: int, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    # forward return
    fut = np.roll(close, -horizon)
    ret = (fut - close) / np.maximum(np.abs(close), 1e-9)
    # long label: ret > thr
    y_long = (ret > thr).astype(np.int32)
    # short label: ret < -thr
    y_short = (ret < -thr).astype(np.int32)
    # son horizon kadar geçersiz
    y_long[-horizon:] = 0
    y_short[-horizon:] = 0
    return y_long, y_short

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32)

def build_lstm(input_dim: int, seq_len: int) -> keras.Model:
    inp = keras.Input(shape=(seq_len, input_dim))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(os.getenv("LSTM_LR", "0.001"))),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return m

def train_one_interval(itv: str) -> None:
    raw = read_offline_csv(SYMBOL, itv)
    feat = build_features(raw)

    # schema
    schema, schema2, seq_len = load_meta_schema(itv)

    # sadece schema kolonlarını al, eksikse 0 bas
    for c in schema2:
        if c not in feat.columns:
            feat[c] = 0.0
    feat = feat[schema2].copy()

    feat = feat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    close = pd.to_numeric(raw["close"], errors="coerce").fillna(0).astype(float).values
    # meta ile tutarlı label
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    horizon = int(meta.get("label_horizon", 3) or 3)
    thr = float(meta.get("label_thr", 0.0005) or 0.0005)

    y_long, y_short = make_labels(close, horizon=horizon, thr=thr)

    X = feat.values.astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_long, Y_long = make_sequences(Xs, y_long, seq_len=seq_len)
    X_short, Y_short = make_sequences(Xs, y_short, seq_len=seq_len)

    # train/val split
    val_split = float(os.getenv("LSTM_VAL_SPLIT", "0.2"))
    n = len(X_long)
    n_val = int(n * val_split)
    n_tr = n - n_val

    def split(Xa, Ya):
        return (Xa[:n_tr], Ya[:n_tr]), (Xa[n_tr:], Ya[n_tr:])

    (XtrL, YtrL), (XvaL, YvaL) = split(X_long, Y_long)
    (XtrS, YtrS), (XvaS, YvaS) = split(X_short, Y_short)

    epochs = int(os.getenv("LSTM_EPOCHS", "30"))
    batch = int(os.getenv("LSTM_BATCH", "64"))
    patience = int(os.getenv("LSTM_PATIENCE", "5"))

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience, restore_best_weights=True),
    ]

    m_long = build_lstm(input_dim=X.shape[1], seq_len=seq_len)
    m_short = build_lstm(input_dim=X.shape[1], seq_len=seq_len)

    print(f"[{itv}] train LONG: X={XtrL.shape} val={XvaL.shape} seq_len={seq_len}")
    m_long.fit(XtrL, YtrL, validation_data=(XvaL, YvaL), epochs=epochs, batch_size=batch, callbacks=cb, verbose=2)

    print(f"[{itv}] train SHORT: X={XtrS.shape} val={XvaS.shape} seq_len={seq_len}")
    m_short.fit(XtrS, YtrS, validation_data=(XvaS, YvaS), epochs=epochs, batch_size=batch, callbacks=cb, verbose=2)

    # save scaler
    scaler_path = os.path.join(MODELS_DIR, SCALER_TEMPLATE_JOBLIB.format(itv=itv))
    joblib.dump(scaler, scaler_path)

    # save models (template)
    long_path = os.path.join(MODELS_DIR, MODEL_LONG_TEMPLATE.format(itv=itv))
    short_path = os.path.join(MODELS_DIR, MODEL_SHORT_TEMPLATE.format(itv=itv))

    m_long.save(long_path)
    m_short.save(short_path)

    print(f"[{itv}] saved scaler={scaler_path}")
    print(f"[{itv}] saved long={long_path}")
    print(f"[{itv}] saved short={short_path}")

def parse_intervals() -> List[str]:
    v = os.getenv("MTF_INTERVALS", "1m,5m,15m,30m,1h")
    itvs = [x.strip() for x in v.split(",") if x.strip()]
    return itvs

def main():
    itvs = parse_intervals()
    print("[LSTM-ALL] intervals:", itvs)
    for itv in itvs:
        train_one_interval(itv)

if __name__ == "__main__":
    main()
