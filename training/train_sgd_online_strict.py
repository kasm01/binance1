from __future__ import annotations

import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from features.pipeline import make_matrix, FEATURE_SCHEMA_22


def make_labels(df: pd.DataFrame, horizon: int, thr: float) -> np.ndarray:
    close = df["close"].astype(float).values
    fut = np.roll(close, -horizon)
    ret = (fut - close) / np.maximum(close, 1e-12)
    y = (ret > thr).astype(int)
    y[-horizon:] = 0
    return y


def walk_forward_eval(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    close: np.ndarray,
    n_folds: int = 5,
    buy_thr: float = 0.60,
    sell_thr: float = 0.40,
):
    """
    X,y aynı uzunlukta olmalı.
    close: aynı indeks hizasında kapanış fiyatları (PnL için).
    """
    n = len(y)
    fold_size = n // (n_folds + 1)
    results = []

    import copy
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = min(fold_size * (k + 1), n)

        Xtr, ytr = X[:train_end], y[:train_end]
        Xte, yte = X[train_end:test_end], y[train_end:test_end]
        close_te = close[train_end:test_end]

        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            results.append({"fold": k, "auc": None, "pnl": None, "note": "single-class fold"})
            continue

        p = copy.deepcopy(pipe)
        p.fit(Xtr, ytr)

        proba = p.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba))

        if len(close_te) >= 2:
            ret = (close_te[1:] - close_te[:-1]) / np.maximum(close_te[:-1], 1e-12)
            sig = np.zeros(len(ret), dtype=float)
            sig[proba[:-1] > buy_thr] = 1.0
            sig[proba[:-1] < sell_thr] = -1.0
            pnl = float(np.sum(sig * ret))
        else:
            pnl = 0.0

        results.append({"fold": k, "auc": auc, "pnl": pnl, "n_test": int(len(yte))})

    return results


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

    # --- feature schema (isimli) ---
    feature_names = [c for c in FEATURE_SCHEMA_22 if c not in ("open_time", "close_time")]

    # --- features (numpy matrix) ---
    schema_used = feature_names
    feat = make_matrix(df, schema=schema_used)
    feat = np.asarray(feat, dtype=float)
    print(f"[FE] builder=make_matrix(schema_used) | feat_shape={feat.shape}")

    # --- labels ---
    y = make_labels(df, horizon=horizon, thr=thr)

    # align (horizon kaynaklı)
    if feat.ndim == 2 and len(y) != feat.shape[0]:
        feat = feat[: len(y), :]

    # numeric temizlik
    X = np.where(np.isfinite(feat), feat, np.nan)
    row_ok = ~np.isnan(X).any(axis=1)
    X = X[row_ok]
    y = np.asarray(y, dtype=int)[: X.shape[0]]

    u, cnt = np.unique(y, return_counts=True)
    freq = dict(zip(u.tolist(), cnt.tolist()))
    print("[YDIST]", freq)
    if len(freq) < 2:
        raise RuntimeError(f"Tek sınıf label çıktı: {freq}. thr/horizon ayarını gözden geçir.")

    # ---- label imbalance auto-reweight (sample_weight) ----
    w0 = 0.5 / max(freq.get(0, 1), 1)
    w1 = 0.5 / max(freq.get(1, 1), 1)
    sw = np.where(y == 1, w1, w0).astype(float)

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

    pipe.fit(X, y, clf__sample_weight=sw)

    # ---- walk-forward validation ----
    close = df["close"].astype(float).values
    close = close[: len(row_ok)][row_ok]  # row_ok hizası
    close = close[: len(y)]               # safety
    wf = walk_forward_eval(pipe, X, y, close, n_folds=5)
    aucs = [r["auc"] for r in wf if r.get("auc") is not None]
    pnls = [r["pnl"] for r in wf if r.get("pnl") is not None]

    # ---- save model + meta ----
    model_path = out_dir / f"online_model_{interval}_best.joblib"
    meta_path = out_dir / f"model_meta_{interval}.json"

    joblib.dump(pipe, model_path)

    meta = {
        "interval": interval,
        "symbol": "BTCUSDT",
        "feature_schema": feature_names,
        "feature_source": "train_sgd_online_strict::make_matrix(FEATURE_SCHEMA_22)",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_horizon": horizon,
        "label_thr": thr,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "meta_version": 4,
        "walk_forward": wf,
        "wf_auc_mean": float(np.mean(aucs)) if aucs else None,
        "wf_pnl_sum": float(np.sum(pnls)) if pnls else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- sanity (p1 clamp) ----
    try:
        X0 = np.zeros((1, len(feature_names)), dtype=float)
        p1 = float(pipe.predict_proba(X0)[0, 1])
    except Exception as e:
        p1 = None
        print("[WARN] sanity predict_proba failed:", repr(e))

    if p1 is not None:
        p1c = min(0.95, max(0.05, p1))
        print("[OK] p1@zeros:", p1, "clamped:", p1c)

    # ---- metrics log append ----
    metrics_path = Path("logs/training/metrics_log.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        "interval": interval,
        "train_csv": str(train_csv),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_h": horizon,
        "label_thr": thr,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "p1_zeros": float(p1) if p1 is not None else None,
        "p1_zeros_clamped": float(min(0.95, max(0.05, p1))) if p1 is not None else None,
        "wf_auc_mean": float(np.mean(aucs)) if aucs else None,
        "wf_pnl_sum": float(np.sum(pnls)) if pnls else None,
    }

    dfrow = pd.DataFrame([row])
    if metrics_path.exists():
        dfrow.to_csv(metrics_path, mode="a", header=False, index=False)
    else:
        dfrow.to_csv(metrics_path, index=False)

    print("[OK] saved:", model_path)
    print("[OK] meta :", meta_path)
    print("[LOG] metrics appended:", metrics_path)
    print(f"[OK] n_samples: {X.shape[0]} n_features: {X.shape[1]}")


if __name__ == "__main__":
    main()
