from __future__ import annotations
import os, json
from pathlib import Path
from features.pipeline import make_matrix_sgd

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


def _get_feature_builder():
    """
    Projede features/pipeline.py içindeki feature builder fonksiyonunu otomatik bul.
    """
    import importlib
    mod = importlib.import_module("features.pipeline")

    candidates = [
        "build_features_df",
        "build_features",
        "make_features_df",
        "make_features",
        "create_features_df",
        "create_features",
        "make_feature_df",
        "build_feature_df",
    ]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn, name

    raise ImportError(
        "features.pipeline içinde feature builder bulunamadı. "
        "Lütfen features/pipeline.py içindeki fonksiyon adını kontrol et."
    )


def make_labels(df: pd.DataFrame, horizon: int, thr: float) -> np.ndarray:
    close = df["close"].astype(float).values
    fut = np.roll(close, -horizon)
    ret = (fut - close) / np.maximum(close, 1e-12)
    y = (ret > thr).astype(int)
    y[-horizon:] = 0
    return y


def main():
    train_csv = os.getenv("SGD_TRAIN_CSV", "data/live_cache/BTCUSDT_5m_live_30d.csv")
    out_dir = Path(os.getenv("SGD_OUT_DIR", "models"))
    interval = os.getenv("SGD_INTERVAL", "5m")

    horizon = int(os.getenv("SGD_LABEL_H", "3"))
    thr = float(os.getenv("SGD_LABEL_THR", "0.0005"))

    max_iter = int(os.getenv("SGD_MAX_ITER", "80000"))
    alpha = float(os.getenv("SGD_ALPHA", "1e-5"))
    tol = float(os.getenv("SGD_TOL", "1e-6"))

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)

    build_fn, build_name = (make_matrix_sgd, 'make_matrix_sgd')
# Bazı projelerde signature farklı olabilir: interval=..., symbol=..., vs.
    # Bu yüzden önce interval ile deneriz, patlarsa interval olmadan deneriz.
    try:
        feat = build_fn(df)
    except TypeError:
        feat = build_fn(df)

    print(f"[FE] builder={build_name} | feat_shape={feat.shape}")

    # label
    y = make_labels(df, horizon=horizon, thr=thr)


    # --- align X (feat) and y due to horizon ---
    # make_labels returns length n-horizon. We must trim feat accordingly.
    if hasattr(feat, "shape") and len(getattr(feat, "shape", ())) == 2:
        n = feat.shape[0]
        if len(y) != n:
            # common case: len(y)=n-horizon
            feat = feat[:len(y), :]
    
    # X seçimi (feat DataFrame veya numpy olabilir)
    if hasattr(feat, "select_dtypes"):
        # pandas DataFrame path
        Xdf = feat.select_dtypes(include=[np.number]).copy()

        # zaman kolonları scaler için sorun çıkarabiliyor; istersen tamamen at
        for c in ["open_time", "close_time"]:
            if c in Xdf.columns:
                Xdf.drop(columns=[c], inplace=True)

        Xdf = Xdf.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[:len(Xdf)]
        X = Xdf.to_numpy(dtype=float)
    else:
        # numpy array path
        X = np.asarray(feat, dtype=float)

        # NaN/inf temizliği
        X = np.where(np.isfinite(X), X, np.nan)
        # satır bazında NaN içerenleri at
        row_ok = ~np.isnan(X).any(axis=1)
        X = X[row_ok]
        y = np.asarray(y, dtype=int)[:X.shape[0]]

    y = np.asarray(y, dtype=int)
    u, cnt = np.unique(y, return_counts=True)
    print("[YDIST]", dict(zip(u.tolist(), cnt.tolist())))
    if u.size < 2:
        raise RuntimeError(f"Tek sınıf label çıktı: {dict(zip(u.tolist(), cnt.tolist()))}. thr/horizon ayarını gözden geçir.")

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            learning_rate="optimal",
            early_stopping=False,
            random_state=42
        ))
    ])

    pipe.fit(X, y)

    model_path = out_dir / f"online_model_{interval}_best.joblib"
    meta_path = out_dir / f"model_meta_{interval}.json"

    joblib.dump(pipe, model_path)

    meta = {
        "interval": interval,
        "symbol": "BTCUSDT",
        "feature_schema": (
                list(getattr(df, "columns", []))
                if hasattr(df, "columns") and ("close" in df.columns)
                else [f"f{i}" for i in range(int(getattr(X, "shape", [0,0])[1]))]
            ),
        "feature_source": f"train_sgd_online_strict::{build_name}",
        "n_samples": int(len(Xdf)),
        "label_horizon": horizon,
        "label_thr": thr,
        "meta_version": 2
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # sanity
    X0 = pd.DataFrame([np.zeros(len(Xdf.columns))], columns=Xdf.columns)
    p1 = pipe.predict_proba(X0)[0, 1]
    print("[OK] saved:", model_path)
    print("[OK] meta:", meta_path)
    print("[OK] p1@zeros:", p1)


if __name__ == "__main__":
    main()
