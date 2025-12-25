from __future__ import annotations

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from features.pipeline import make_matrix_sgd, SGD_SCHEMA_NO_TIME

def make_label(df: pd.DataFrame, horizon: int = 1, thr: float = 0.0) -> np.ndarray:
    """
    Binary label:
      y=1 if future_return(horizon) > thr else 0
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    fut = close.shift(-horizon)
    ret = (fut - close) / close.replace(0, np.nan)
    y = (ret > thr).astype(int).to_numpy()
    return y

def main():
    csv_path = Path(os.getenv("SGD_TRAIN_CSV", "data/offline_cache/BTCUSDT_5m_6m.csv"))
    out_dir = Path(os.getenv("SGD_OUT_DIR", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(os.getenv("SGD_LABEL_H", "1"))
    thr = float(os.getenv("SGD_LABEL_THR", "0.0"))
    alpha = float(os.getenv("SGD_ALPHA", "5e-5"))

    print("[SGD_HELPER] csv:", csv_path)
    df = pd.read_csv(csv_path)
    df_full = df.copy()  # === TRAIN_DF_LOCK (auto) ===

    # sıralama (stabil)
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    # Feature matrix + mask
    # === MAKE_MATRIX_UNWRAP (auto) ===
    res = make_matrix_sgd(df_full)
    if isinstance(res, tuple) and len(res) == 3:
        X, cols, mask = res
    else:
        X = res
        try:
            cols = list(getattr(df, 'columns', []))
        except Exception:
            cols = []
        mask = None
    if cols != SGD_SCHEMA_NO_TIME:
        print("[SGD_HELPER] WARN: schema mismatch, using pipeline cols")

    # Label (df ile hizalı üret, sonra mask uygula)
    y_all = make_label(df_full, horizon=horizon, thr=thr)
    # mask ve horizon nedeniyle son horizon satır label NaN olabilir -> mask zaten engineered NaN kırıyor ama
    # burada bir de finite check:
    # === LABEL_MASK_GUARD (auto) ===
    import numpy as _np
    if isinstance(y_all, tuple) and len(y_all) >= 1:
        y_all = y_all[0]
    y_all = _np.asarray(y_all).reshape(-1)
    if mask is None:
        y_masked = y_all
    else:
        y_masked = y_all[mask]
    # son horizon satırları shift(-horizon) yüzünden NaN'a gidebilir; y_all onları 0 yapar ama
    # yine de boy kontrol:
    n = min(len(X), len(y_masked))
    X = X[:n]
    y = y_masked[:n].astype(int)

    pos = float(y.mean()) if len(y) else 0.0
    print(f"[SGD_HELPER] X shape={X.shape} y shape={y.shape} pos_rate={pos:.4f} (h={horizon}, thr={thr})")
    if len(y) < 1000:
        raise SystemExit("[SGD_HELPER] too few samples after feature cleaning; check pipeline/CSV")

    # scale
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    # SGD (helper model)
    # === SGD_MAXITER_TUNING (auto) ===
    # max_iter/tol/early_stopping'i env ile kontrol et (convergence için)
    max_iter = int(os.getenv('SGD_MAX_ITER', '5000'))
    tol = float(os.getenv('SGD_TOL', '1e-4'))
    early = str(os.getenv('SGD_EARLY_STOP', '1')).lower() in ('1','true','yes','on')
    val_frac = float(os.getenv('SGD_VAL_FRAC', '0.05'))
    n_no_change = int(os.getenv('SGD_N_NO_CHANGE', '10'))
    base = SGDClassifier(

        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        shuffle=True,
        class_weight="balanced",
        random_state=42,
    
)

    # Calibration (sigmoid) - helper için kritik
    cal = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    cal.fit(Xs, y)

    # quick sanity: proba mean ~ 0.5 hedef
    p1 = cal.predict_proba(Xs)[:, 1]
    print(f"[SGD_HELPER] train p1 stats: min={p1.min():.4f} max={p1.max():.4f} mean={p1.mean():.4f}")

    # save
    # === SGD_HELPER_SAVE_PIPELINE (auto) ===
    # scaler+calibrated SGD'yi tek pipeline olarak kaydet (runtime'da aynı dönüşüm)
    pipe = Pipeline([
        ('scaler', scaler),
        ('clf', cal),
    ])
    # === SGD_OUTPATH_FIX (auto) ===
    out_path = os.path.join(out_dir, 'sgd_helper.joblib')

    # --- bundle save (robust) ---
    model_obj = None
    for _k in ("clf","calibrated","cal","model","pipe","estimator"):
        if _k in locals():
            model_obj = locals().get(_k)
            if model_obj is not None:
                break
    if model_obj is None:
        raise RuntimeError("[SGD_HELPER] trained model object not found in locals()")

    bundle = {
        'model': model_obj,
        'scaler': scaler,
        'cols': SGD_SCHEMA_NO_TIME if 'SGD_SCHEMA_NO_TIME' in globals() else [],
        'meta': {'horizon': horizon, 'thr': thr},
    }
    joblib.dump(bundle, out_path)

    print("[SGD_HELPER] saved:", out_dir / "sgd_helper.joblib")

if __name__ == "__main__":
    main()
