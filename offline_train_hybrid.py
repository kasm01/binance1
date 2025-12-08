#!/usr/bin/env python
import os
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger("offline_train")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DATA_DIR = "data/offline_cache"
MODELS_DIR = "models"

# Hangi timeframe'leri eğitelim?
# İstersen buraya "5m" de ekledim, hepsi birlikte eğitilecek.
INTERVALS: List[str] = ["1m", "5m", "15m", "1h"]

# Ne kadar bar kullanalım? Çok büyük dosyalarda kırpabiliriz.
MAX_BARS = int(os.getenv("OFFLINE_MAX_BARS", "50000"))
HORIZON = int(os.getenv("LABEL_HORIZON", "1"))  # future bar sayısı


# ------------------------------------------------------------
# Feature & Label helpers
# ------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim ve inference için kullandığımız feature şeması:

      ['open_time', 'open', 'high', 'low', 'close', 'volume',
       'close_time', 'quote_asset_volume', 'number_of_trades',
       'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore',
       'hl_range', 'oc_change', 'return_1', 'return_3', 'return_5',
       'ma_5', 'ma_10', 'ma_20', 'vol_10', 'dummy_extra']
    """
    df = raw_df.copy()

    # Zaman kolonlarını saniye cinsinden float'a çevir (tutarlı olsun)
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            # ms ise unit="ms" ile çevir
            dt = pd.to_datetime(df[col], unit="ms", utc=True, errors="coerce")
            # gelecekteki pandas uyarısından kaçınmak için astype kullan
            df[col] = dt.view("int64") / 1e9

    # Price / volume feature'ları
    df["hl_range"] = df["high"] - df["low"]
    df["oc_change"] = df["close"] - df["open"]

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    df["ma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df["close"].rolling(window=10, min_periods=1).mean()
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()

    df["vol_10"] = df["volume"].rolling(window=10, min_periods=1).std()

    # Eğitime uyum için ekstra dummy kolon
    df["dummy_extra"] = 0.0

    # NA temizliği (forward/backfill + 0)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    return df


def build_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Eğitimde kullanılan label mantığı:

        future_close = df["close"].shift(-horizon)
        ret = future_close / df["close"] - 1.0
        y = (ret > 0.0).astype(int)
    """
    future_close = close.shift(-horizon)
    ret = future_close / close - 1.0
    y = (ret > 0.0).astype(int)
    return y


def make_xy(df_raw: pd.DataFrame, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """
    raw_df'den feature DF ve label serisi üretir ve hizalar.
    Son horizon bar, geleceği bilinmediği için atılır.
    """
    feat_df = build_features(df_raw)

    y_all = build_labels(feat_df["close"], horizon=horizon)
    # Son horizon bar için future_close bilinmiyor → NA.
    mask = y_all.notna()
    feat_df_aligned = feat_df[mask].copy()
    y_aligned = y_all[mask].astype(int)

    return feat_df_aligned, y_aligned


# ------------------------------------------------------------
# Data loader
# ------------------------------------------------------------
def load_offline_klines(symbol: str, interval: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_{interval}_6m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Offline cache yok: {path}")

    df = pd.read_csv(path)
    if MAX_BARS > 0 and len(df) > MAX_BARS:
        df = df.tail(MAX_BARS).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    logger.info(
        "[%s] Loaded offline klines: shape=%s | path=%s",
        interval,
        df.shape,
        path,
    )
    return df


# ------------------------------------------------------------
# Training logic per interval
# ------------------------------------------------------------
def train_interval(interval: str) -> None:
    logger.info("========== TRAIN interval=%s ==========", interval)

    # 1) Veri yükle
    raw_df = load_offline_klines(SYMBOL, interval)

    # 2) X, y üret
    X_df, y = make_xy(raw_df, horizon=HORIZON)

    # Yeterli veri var mı?
    if len(X_df) < 1000:
        logger.warning("[%s] Çok az sample: n=%d, skip.", interval, len(X_df))
        return

    # 3) Numeric feature matrix
    X_num = X_df.select_dtypes(include=[np.number])
    feature_cols = list(X_num.columns)
    X = X_num.to_numpy(dtype=float)

    logger.info(
        "[%s] X shape=%s | y len=%d | feature_cols=%s",
        interval,
        X.shape,
        len(y),
        feature_cols,
    )

    # 4) Train/validation split (time-series, shuffle=False)
    n = X.shape[0]
    split_idx = int(n * 0.8)
    X_train, y_train = X[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X[split_idx:], y.iloc[split_idx:]

    # 5) SGDClassifier pipeline
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "sgd",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    max_iter=50,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logger.info(
        "[%s] Training SGDClassifier... (train_n=%d, val_n=%d)",
        interval,
        X_train.shape[0],
        X_val.shape[0],
    )

    clf.fit(X_train, y_train)

    # 6) Validation AUC
    try:
        proba = clf.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc = float(roc_auc_score(y_val, proba))
        else:
            auc = 0.5
            logger.warning(
                "[%s] y_val tek sınıf! AUC gerçek anlamlı değil, 0.5 set edildi.",
                interval,
            )
    except Exception as e:
        logger.warning("[%s] AUC hesaplanırken hata: %s", interval, e)
        auc = 0.5

    logger.info(
        "[%s] Validation AUC=%.4f | label_mean=%.4f",
        interval,
        auc,
        float(y.mean()),
    )

    # 7) Model & meta kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"online_model_{interval}_best.joblib")
    joblib.dump(clf, model_path)

    # best_side kaba mantık: long/short hangi taraf daha anlamlı gibi.
    best_side = "long" if auc >= 0.5 else "short"

    meta_path = os.path.join(MODELS_DIR, f"model_meta_{interval}.json")
    meta: Dict[str, Any] = {
        "interval": interval,
        "symbol": SYMBOL,
        "feature_schema": feature_cols,
        "feature_source": "fixed_schema",
        "n_samples": int(len(y)),
        "best_auc": float(auc),
        "best_side": best_side,
        # 5m için LSTM zaten var; diğerleri için şu an sadece SGD kullanıyoruz.
        "use_lstm_hybrid": True if interval == "5m" else False,
        "seq_len": 32,
        "meta_version": 1,
        "label_horizon": HORIZON,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "[%s] Model & meta kaydedildi: model=%s | meta=%s",
        interval,
        model_path,
        meta_path,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    logger.info(
        "Offline hybrid SGD training started | symbol=%s | intervals=%s",
        SYMBOL,
        INTERVALS,
    )

    for itv in INTERVALS:
        try:
            train_interval(itv)
        except FileNotFoundError as e:
            logger.warning(
                "[%s] CSV bulunamadı, bu interval atlandı: %s",
                itv,
                e,
            )
        except Exception as e:
            logger.exception(
                "[%s] Interval train sırasında hata: %s",
                itv,
                e,
            )

    logger.info("Offline hybrid SGD training finished.")


if __name__ == "__main__":
    main()
