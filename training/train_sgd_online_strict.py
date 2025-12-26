from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from features.pipeline import make_matrix_sgd


def _get_feature_builder():
    """
    Projede features/pipeline.py içindeki feature builder fonksiyonunu otomatik bul.
    (Şu an default make_matrix_sgd kullanıyoruz; gerekirse bunu aktive edersin.)
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
        "make_matrix_sgd",
        "make_matrix",
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

    # ---- feature builder ----
    # Default: sende var olan fonksiyon
    build_fn, build_name = (make_matrix_sgd, "make_matrix_sgd")
    # Alternatif istersen:
    # build_fn, build_name = _get_feature_builder()

    try:
        feat = build_fn(df)
    except TypeError:
        feat = build_fn(df)

    print(f"[FE] builder={build_name} | feat_shape={getattr(feat, 'shape', None)}")

    # ---- labels ----
    y = make_labels(df, horizon=horizon, thr=thr)

    # ---- align feat and y (horizon farkı) ----
    if hasattr(feat, "shape") and len(getattr(feat, "shape", ())) == 2:
        if len(y) != feat.shape[0]:
            feat = feat[: len(y), :]

    # ---- X extraction (DataFrame veya numpy) ----
    Xdf = None
    feature_names: list[str]

    if hasattr(feat, "select_dtypes"):
        # pandas path
        Xdf = feat.select_dtypes(include=[np.number]).copy()

        for c in ["open_time", "close_time"]:
            if c in Xdf.columns:
                Xdf.drop(columns=[c], inplace=True)

        Xdf = Xdf.replace([np.inf, -np.inf], np.nan).dropna()

        # y'yi Xdf ile hizala
        y = y[: len(Xdf)]
        X = Xdf.to_numpy(dtype=float)

        feature_names = list(Xdf.columns)
    else:
        # numpy path
        X = np.asarray(feat, dtype=float)

        # NaN/inf temizliği
        X = np.where(np.isfinite(X), X, np.nan)
        row_ok = ~np.isnan(X).any(axis=1)
        X = X[row_ok]

        y = np.asarray(y, dtype=int)[: X.shape[0]]

        feature_names = [f"f{i}" for i in range(int(X.shape[1]))]

    # ---- y dist ----
    y = np.asarray(y, dtype=int)
    u, cnt = np.unique(y, return_counts=True)
    print("[YDIST]", dict(zip(u.tolist(), cnt.tolist())))
    if u.size < 2:
        raise RuntimeError(
            f"Tek sınıf label çıktı: {dict(zip(u.tolist(), cnt.tolist()))}. "
            f"thr/horizon ayarını gözden geçir."
        )

    # ---- train ----
    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=alpha,
                    max_iter=max_iter,
                    tol=tol,
                    learning_rate="optimal",
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe.fit(X, y)

    model_path = out_dir / f"online_model_{interval}_best.joblib"
    meta_path = out_dir / f"model_meta_{interval}.json"

    joblib.dump(pipe, model_path)

    meta = {
        "interval": interval,
        "symbol": "BTCUSDT",
        "feature_schema": feature_names,
        "feature_source": f"train_sgd_online_strict::{build_name}",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_horizon": horizon,
        "label_thr": thr,
        "meta_version": 3,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- sanity ----
    try:
        if Xdf is not None:
            X0 = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
            p1 = float(pipe.predict_proba(X0)[0, 1])
        else:
            X0 = np.zeros((1, len(feature_names)), dtype=float)
            p1 = float(pipe.predict_proba(X0)[0, 1])
    except Exception as e:
        p1 = None
        print("[WARN] sanity predict_proba failed:", repr(e))

    print("[OK] saved:", model_path)
    print("[OK] meta:", meta_path)
    print("[OK] n_samples:", X.shape[0], "n_features:", X.shape[1])
    print("[OK] p1@zeros:", p1)


if __name__ == "__main__":
    main()
