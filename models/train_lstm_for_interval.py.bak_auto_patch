import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from joblib import dump

# ----------------------------------------------------------------------
# Feature şeması (main.py ile birebir aynı)
# ----------------------------------------------------------------------
FEATURE_SCHEMA = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
    "hl_range",
    "oc_change",
    "return_1",
    "return_3",
    "return_5",
    "ma_5",
    "ma_10",
    "ma_20",
    "vol_10",
    "dummy_extra",
]

SEQ_LEN = 32
LABEL_HORIZON = 1


# ----------------------------------------------------------------------
# main.py ile aynı feature engineering
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hem Binance canlı verisi (ms epoch) hem de offline CSV (ISO datetime string)
    ile çalışacak şekilde feature üretir.
    """
    df = raw_df.copy()

    # 1) Zaman kolonlarını normalize et (open_time / close_time)
    for col in ["open_time", "close_time"]:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            dt = pd.to_datetime(df[col], unit="ms", utc=True)
        else:
            dt = pd.to_datetime(df[col], utc=True)

        df[col] = dt.astype("int64") / 1e9  # ns -> s

    # 2) Temel numeric cast
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


# ----------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ----------------------------------------------------------------------
def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def load_offline_csv(symbol: str, interval: str):
    csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik feature kolonları: {missing}")
    return df[FEATURE_SCHEMA]


def make_labels(df: pd.DataFrame, horizon=1):
    close = df["close"].values.astype(float)
    y = (np.roll(close, -horizon) > close).astype(int)
    y[-horizon:] = 0
    return y


# ----------------------------------------------------------------------
# Ana eğitim fonksiyonu
# ----------------------------------------------------------------------
def train_for_interval(interval="5m", symbol="BTCUSDT"):

    print(f"[LSTM-TRAIN] Eğitim başlıyor -> {symbol} {interval}")

    # 1) Ham CSV'yi oku
    raw_df = load_offline_csv(symbol, interval)

    # 2) main.py ile aynı feature engineering
    feat_df = build_features(raw_df)

    # 3) LSTM için kullanılacak kolonları seç
    feat_df = ensure_features(feat_df)

    # 4) Label üret
    y = make_labels(feat_df, horizon=LABEL_HORIZON)

    # 5) Numpy array'e çevir
    X = feat_df.values.astype(float)

    # 6) StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7) Sekans (SEQ_LEN, n_features)
    X_seq, y_seq = build_sequences(X_scaled, y, SEQ_LEN)

    print("[LSTM-TRAIN] Veri şekli:", X_seq.shape, y_seq.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=True
    )

    # 8) Model mimarisi
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    es = EarlyStopping(
        patience=10,
        restore_best_weights=True,
        monitor="val_loss",
    )

    print("[LSTM-TRAIN] Eğitim başlıyor (fit)...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[es],
        verbose=1,
    )

    print("[LSTM-TRAIN] Eğitim bitti. Modeller kaydediliyor...")

    os.makedirs("models", exist_ok=True)

    long_path = f"models/lstm_long_{interval}.h5"
    short_path = f"models/lstm_short_{interval}.h5"
    scaler_path = f"models/lstm_scaler_{interval}.joblib"

    model.save(long_path)
    model.save(short_path)  # Şimdilik aynı modeli 2 kez kaydediyoruz
    dump(scaler, scaler_path)

    print(f"[LSTM-TRAIN] Modeller kaydedildi:")
    print(f"  - {long_path}")
    print(f"  - {short_path}")
    print(f"  - {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    train_for_interval(args.interval, args.symbol)
