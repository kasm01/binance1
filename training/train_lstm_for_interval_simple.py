#!/usr/bin/env python
import os

from app_paths import MODELS_DIR
import json
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("lstm_train")

# --- LSTM feature contract (MUST match inference) ---
# LSTM uses NO time columns; keep it stable across intervals.
LSTM_FEATURE_SCHEMA = [
    "open","high","low","close","volume",
    "quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore",
    "hl_range","oc_change","return_1","return_3","return_5",
    "ma_5","ma_10","ma_20","vol_10","dummy_extra",
]

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip()=="" else str(v).strip()

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip()=="":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip()=="":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

SYMBOL     = _env_str("SYMBOL", "BTCUSDT")
INTERVAL   = _env_str("INTERVAL", "3m")
DATA_DIR   = _env_str("DATA_DIR", "data/offline_cache")

MAX_BARS   = _env_int("OFFLINE_MAX_BARS", 50000)
HORIZON    = _env_int("LABEL_HORIZON", 1)
LABEL_THR  = _env_float("LABEL_THR", 0.0)

SEQ_LEN    = _env_int("LSTM_SEQ_LEN", _env_int("LSTM_SEQ_LEN_DEFAULT", 50))
EPOCHS     = _env_int("LSTM_EPOCHS", 3)
BATCH_SIZE = _env_int("LSTM_BATCH_SIZE", 256)
UNITS      = _env_int("LSTM_UNITS", 64)
DROPOUT    = _env_float("LSTM_DROPOUT", 0.2)
PATIENCE   = _env_int("LSTM_PATIENCE", 2)

def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # temel feature set (offline_train_hybrid ile aynı)
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["ma_5"] = df["close"].rolling(5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(20, min_periods=1).mean()
    df["vol_10"] = df["volume"].rolling(10, min_periods=1).std()
    df["dummy_extra"] = 0.0
    df = df.ffill().bfill().fillna(0.0)
    return df

def make_xy(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    feat = build_features(df_raw)

    # --- enforce LSTM schema (order + missing fill) ---
    for c in LSTM_FEATURE_SCHEMA:
        if c not in feat.columns:
            feat[c] = 0.0
    feat = feat[LSTM_FEATURE_SCHEMA].copy()

    future_close = df_raw["close"].shift(-HORIZON)  # label raw close üzerinden
    ret = (future_close / df_raw["close"]) - 1.0
    y = (ret > float(LABEL_THR)).astype(int)

    m = y.notna()
    X = feat.loc[m].to_numpy(dtype=float)
    y = y.loc[m].astype(int).to_numpy()
    cols = list(feat.columns)
    return X, y, cols

def build_seqs(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        raise ValueError(f"Not enough rows for seq_len={seq_len} (n={len(X)})")
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i-seq_len:i, :])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

def make_model(n_features: int) -> Sequential:
    m = Sequential()
    m.add(LSTM(UNITS, input_shape=(SEQ_LEN, n_features)))
    m.add(Dropout(DROPOUT))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy")
    return m

def main():
    path = os.path.join(DATA_DIR, f"{SYMBOL}_{INTERVAL}_6m.csv")
    if not os.path.exists(path):
        raise SystemExit(f"CSV yok: {path}")

    df = pd.read_csv(path)
    if MAX_BARS > 0 and len(df) > MAX_BARS:
        df = df.tail(MAX_BARS).reset_index(drop=True)

    X, y, cols = make_xy(df)

    # Align LSTM features with offline meta schema (prevents scaler mismatch)
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{INTERVAL}.json")
    try:
        import json
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                _m = json.load(f) or {}
            sch = _m.get("feature_schema")
            if isinstance(sch, list) and sch:
                # LSTM uses only numeric features; enforce same order
                cols = [c for c in sch if c in df.columns]
                df = df[cols].copy()
                X, y, cols = make_xy(df)
    except Exception as _e:
        log.warning(f"Meta schema align skipped interval={INTERVAL}: {_e}")


    if len(X) < (SEQ_LEN + 1000):
        log.warning(f"Not enough data for LSTM interval={INTERVAL}: n={len(X)} seq_len={SEQ_LEN}. Skipping.")
        return


    # time split
    n = len(X)
    split = int(n * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    X_tr_seq, y_tr_seq = build_seqs(X_tr_s, y_tr, SEQ_LEN)
    X_va_seq, y_va_seq = build_seqs(X_va_s, y_va, SEQ_LEN)

    log.info("Train LSTM %s | X_tr_seq=%s X_va_seq=%s seq_len=%d", INTERVAL, X_tr_seq.shape, X_va_seq.shape, SEQ_LEN)

    m_long = make_model(n_features=X_tr_seq.shape[2])
    m_short = make_model(n_features=X_tr_seq.shape[2])

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    m_long.fit(X_tr_seq, y_tr_seq, validation_data=(X_va_seq, y_va_seq), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)
    m_short.fit(X_tr_seq, y_tr_seq, validation_data=(X_va_seq, y_va_seq), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

    # AUC
    p = m_long.predict(X_va_seq, verbose=0).reshape(-1)
    auc = 0.5
    try:
        if len(np.unique(y_va_seq)) > 1:
            auc = float(roc_auc_score(y_va_seq, p))
    except Exception:
        auc = 0.5

    os.makedirs(MODELS_DIR, exist_ok=True)
    long_path  = os.path.join(MODELS_DIR, f"lstm_long_{INTERVAL}.h5")
    short_path = os.path.join(MODELS_DIR, f"lstm_short_{INTERVAL}.h5")
    scaler_path= os.path.join(MODELS_DIR, f"lstm_scaler_{INTERVAL}.joblib")

    m_long.save(long_path)
    m_short.save(short_path)
    joblib.dump(scaler, scaler_path)

    # meta update (varsa)
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{INTERVAL}.json")
    meta: Dict[str, Any] = {}

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception as e:
            log.warning("Meta load failed, will recreate: %s", e)
            meta = {}

    # LSTM sözleşmesini META'ya KİLİTLE
    meta.update(
        {
            "use_lstm_hybrid": True,
            "seq_len": int(SEQ_LEN),
            "lstm_long_auc": float(auc),
            "lstm_short_auc": float(auc),

            # 🔒 KRİTİK: inference ile birebir aynı
            "lstm_feature_schema": list(LSTM_FEATURE_SCHEMA),
        }
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(
        "Saved LSTM: %s %s | scaler=%s | auc=%.4f | n_features=%d",
        long_path,
        short_path,
        scaler_path,
        auc,
        scaler.n_features_in_,
    )

if __name__ == "__main__":
    main()
