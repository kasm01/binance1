cat > models/train_lstm_for_interval.py << 'EOF'
import os
import json
import argparse
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ----------------------------------------------------------------------
# Basit logger
# ----------------------------------------------------------------------
logger = logging.getLogger("lstm_train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[LSTM-TRAIN] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)


# ----------------------------------------------------------------------
# main.py'deki ile aynı feature builder
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hem Binance canlı verisi (ms epoch) hem de offline CSV (ISO datetime string)
    ile çalışacak şekilde feature üretir.
    """
    df = raw_df.copy()

    # 1) Zaman kolonlarını normalize et
    for col in ["open_time", "close_time"]:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            dt = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            dt = pd.to_datetime(df[col], utc=True)

        df[col] = dt.astype("int64") / 1e9  # saniye bazlı epoch float

    # 2) Numeric cast
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    int_cols = ["number_of_trades"]

    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    if "ignore" not in df.columns:
        df["ignore"] = 0.0

    # 3) Teknik feature'lar
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    df["vol_10"] = df["volume"].rolling(10).std()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    df = df.ffill().bfill().fillna(0.0)
    return df


def build_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Basit label: horizon bar sonra close > current close ise 1 (up), yoksa 0 (down)
    """
    close = df["close"].astype(float)
    future = close.shift(-horizon)
    labels = (future > close).astype(float)
    return labels


# ----------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------
def load_meta(model_dir: str, interval: str) -> Dict[str, Any]:
    meta_path = os.path.join(model_dir, f"model_meta_{interval}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta dosyası yok: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f) or {}

    return meta


def save_meta(model_dir: str, interval: str, meta: Dict[str, Any]) -> None:
    meta_path = os.path.join(model_dir, f"model_meta_{interval}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_raw_klines(symbol: str, interval: str) -> pd.DataFrame:
    csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Offline kline CSV bulunamadı: {csv_path}. "
            "Önce data/offline_cache altında bu dosyayı oluşturmalısın."
        )
    df = pd.read_csv(csv_path)
    logger.info("Kline CSV yüklendi: %s, shape=%s", csv_path, df.shape)
    return df


def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X) <= seq_len:
        raise ValueError(
            f"Yeterli veri yok: len(X)={len(X)}, seq_len={seq_len}"
        )

    X_seqs = []
    y_seqs = []
    for i in range(seq_len, len(X)):
        X_seqs.append(X[i - seq_len : i, :])
        y_seqs.append(y[i])

    X_seqs = np.asarray(X_seqs, dtype=np.float32)
    y_seqs = np.asarray(y_seqs, dtype=np.float32)
    return X_seqs, y_seqs


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------------------------------------------------------------------
# Ana eğitim fonksiyonu
# ----------------------------------------------------------------------
def train_for_interval(interval: str, symbol: str = "BTCUSDT") -> None:
    model_dir = "models"

    logger.info("Interval=%s için eğitim başlıyor...", interval)

    # 1) Meta yükle (feature_schema + seq_len + label_horizon)
    meta = load_meta(model_dir, interval)
    feature_cols: List[str] = meta.get("feature_schema", [])
    if not feature_cols:
        raise ValueError(
            f"model_meta_{interval}.json içinde feature_schema boş. "
            "Önce online model eğitiminde feature_schema kaydedilmiş olmalı."
        )

    seq_len = int(meta.get("seq_len", 32))
    horizon = int(meta.get("label_horizon", 1))

    logger.info(
        "feature_cols=%d, seq_len=%d, horizon=%d",
        len(feature_cols),
        seq_len,
        horizon,
    )

    # 2) Ham kline CSV yükle ve feature üret
    raw_df = load_raw_klines(symbol=symbol, interval=interval)
    feat_df = build_features(raw_df)

    # 3) Feature kolon kontrolü
    missing_cols = [c for c in feature_cols if c not in feat_df.columns]
    if missing_cols:
        raise ValueError(
            f"Feature DF içinde eksik kolonlar var: {missing_cols}"
        )

    X_all = feat_df[feature_cols].astype(float).values

    # 4) Label üret
    y_series = build_labels(feat_df, horizon=horizon)
    mask = ~y_series.isna()
    X_all = X_all[mask.values]
    y_all = y_series[mask].values

    logger.info("Toplam örnek sayısı (mask sonrası): %d", len(X_all))

    # 5) Train / validation split (80/20)
    n = len(X_all)
    if n <= seq_len + 10:
        raise ValueError(
            f"Veri çok az (n={n}), seq_len={seq_len}. Eğitim yapılamıyor."
        )

    split_idx = int(n * 0.8)
    X_train_raw = X_all[:split_idx]
    y_train_raw = y_all[:split_idx]
    X_val_raw = X_all[split_idx:]
    y_val_raw = y_all[split_idx:]

    # 6) Scaler (train set üzerinde)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    # 7) Sequence oluştur
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_raw, seq_len)
    X_val_seq, y_val_seq = make_sequences(X_val_scaled, y_val_raw, seq_len)

    logger.info(
        "Train seq shape=%s, Val seq shape=%s",
        X_train_seq.shape,
        X_val_seq.shape,
    )

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    # 8) Callback'ler
    es = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-5
    )

    # 9) Long model (y=1 olasılığı)
    logger.info("LSTM long modeli eğitiliyor...")
    long_model = build_lstm_model(input_shape)
    long_model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=64,
        callbacks=[es, rlrop],
        verbose=2,
    )

    y_val_pred_long = long_model.predict(X_val_seq, verbose=0).reshape(-1)
    try:
        auc_long = roc_auc_score(y_val_seq, y_val_pred_long)
    except Exception:
        auc_long = 0.5

    logger.info("Long model val AUC=%.4f", auc_long)

    # 10) Short model (basitçe aynı mimariyi yeniden eğitelim)
    logger.info("LSTM short modeli eğitiliyor...")
    short_model = build_lstm_model(input_shape)
    # Short modeli, hedefi (1 - y) ile de eğitebilirdik; ama
    # hibritte ortalama alınacağı için aynı y ile eğitmek de kabul edilebilir.
    short_model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=64,
        callbacks=[es, rlrop],
        verbose=2,
    )

    y_val_pred_short = short_model.predict(X_val_seq, verbose=0).reshape(-1)
    try:
        auc_short = roc_auc_score(y_val_seq, y_val_pred_short)
    except Exception:
        auc_short = 0.5

    logger.info("Short model val AUC=%.4f", auc_short)

    # 11) Modelleri ve scaler'ı kaydet
    long_path = os.path.join(model_dir, f"lstm_long_{interval}.h5")
    short_path = os.path.join(model_dir, f"lstm_short_{interval}.h5")
    scaler_path = os.path.join(model_dir, f"lstm_scaler_{interval}.joblib")

    long_model.save(long_path)
    short_model.save(short_path)
    joblib.dump(scaler, scaler_path)

    logger.info("Modeller kaydedildi: %s, %s", long_path, short_path)
    logger.info("Scaler kaydedildi: %s", scaler_path)

    # 12) Meta güncelle
    meta["use_lstm_hybrid"] = True
    meta["seq_len"] = int(seq_len)
    meta["lstm_long_auc"] = float(auc_long)
    meta["lstm_short_auc"] = float(auc_short)

    save_meta(model_dir, interval, meta)
    logger.info("Meta güncellendi: model_meta_%s.json", interval)
    logger.info("Interval=%s için LSTM eğitim tamamlandı.", interval)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, required=True, help="örn: 1m,5m,15m,1h")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    train_for_interval(interval=args.interval, symbol=args.symbol)


if __name__ == "__main__":
    main()
EOF
