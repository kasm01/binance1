# training/offline_pretrain_six_months.py
"""
BTCUSDT için 6 aylık geçmiş veriyi kullanarak offline pre-train:

- Binance public API'den 6 aylık kline çekme (her interval için)
- Feature engineering (returns, MA'ler, volatility vb.)
- Anomali filtresi (IsolationForest) (opsiyonel, hata olursa atlanır)
- Long & Short için ayrı SGDClassifier tuning (shallow/full/deep modları)
- En iyi modeli & yönü seçip:
    models/online_model_<interval>_long.joblib
    models/online_model_<interval>_short.joblib
    models/online_model_<interval>_best.joblib
  olarak kaydeder.
- CLI:
    --mode shallow | full | deep
    --intervals "1m,5m,15m,1h"
    --use-lstm-hybrid (şimdilik sadece log yazar, LSTM eğitimi yok)
"""

from __future__ import annotations

import argparse
import time
import json
from datetime import datetime, timedelta, timezone
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os
import config as app_config
from data.lstm_hybrid import train_lstm_hybrid

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import copy

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from pathlib import Path
import pandas as pd
def normalize_kline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binance kline kolon isimlerini normalize eder.
    Hem 'taker_buy_base_asset_volume' hem 'taker_buy_base_volume' gibi
    farklı isimlendirmeleri tolere etmek için kullanıyoruz.
    """
    # base volume
    if "taker_buy_base_asset_volume" not in df.columns and "taker_buy_base_volume" in df.columns:
        df["taker_buy_base_asset_volume"] = df["taker_buy_base_volume"]

    if "taker_buy_base_volume" not in df.columns and "taker_buy_base_asset_volume" in df.columns:
        df["taker_buy_base_volume"] = df["taker_buy_base_asset_volume"]

    # quote volume
    if "taker_buy_quote_asset_volume" not in df.columns and "taker_buy_quote_volume" in df.columns:
        df["taker_buy_quote_asset_volume"] = df["taker_buy_quote_volume"]

    if "taker_buy_quote_volume" not in df.columns and "taker_buy_quote_asset_volume" in df.columns:
        df["taker_buy_quote_volume"] = df["taker_buy_quote_asset_volume"]

    return df

# ... diğer importlar ...

OFFLINE_CACHE_DIR = Path("data/offline_cache")
OFFLINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _offline_cache_path(symbol: str, interval: str, months: int) -> Path:
    """
    Offline klines cache dosya yolu.
    Örn: data/offline_cache/BTCUSDT_5m_6m.csv
    """
    fname = f"{symbol}_{interval}_{months}m.csv"
    return OFFLINE_CACHE_DIR / fname


# Binance public client (API key gerekmiyor, public endpoint)
try:
    from binance.client import Client
except ImportError:  # tip: offline test için
    Client = None  # type: ignore

# Ortam değişkenleri için (uygunsa)
try:
    from config.load_env import load_environment_variables  # type: ignore
except Exception:  # pragma: no cover
    def load_environment_variables(*args, **kwargs):
        print(
            "[load_env] INFO: config.load_env.load_environment_variables bulunamadı, atlanıyor.")


# -------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -------------------------------------------------------------------------


def fetch_klines_offline(
    symbol: str,
    interval: str,
    months: int = 6,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Offline eğitim için 6 aylık (veya verilen months değeri kadar) klines verisini getirir.

    - use_cache=True ise önce data/offline_cache klasöründen CSV arar.
    - force_refresh=True ise cache'i yok sayar, Binance'ten yeniden çeker.
    """
    cache_path = _offline_cache_path(symbol, interval, months)

    if use_cache and not force_refresh and cache_path.exists():
        print(
            f"[OFFLINE] Loading cached klines from {cache_path} "
            f"(symbol={symbol}, interval={interval}, months={months})"
        )
        df = pd.read_csv(cache_path, parse_dates=["open_time", "close_time"])
        print(f"[OFFLINE] cached_df.shape={df.shape}")
        return df

    # ---- CACHE YOKSA / FORCE REFRESH ----
    from binance.client import Client
    from config.load_env import load_environment_variables  # type: ignore

    # Ortam değişkenlerini yükle (en azından BASE_URL vb. varsa)
    load_environment_variables()

    # Basit client (senin projende create_binance_client varsa onu da
    # kullanabilirsin)
    try:
        from core.binance_client import create_binance_client  # type: ignore
        client = create_binance_client(app_config)
    except Exception:
        api_key = ""
        api_secret = ""
        use_testnet = getattr(app_config, "USE_TESTNET", False)
        client = Client(api_key, api_secret, testnet=use_testnet)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * months)

    print(
        f"[OFFLINE] fetch_klines_offline: symbol={symbol}, interval={interval}, "
        f"start={start}, end={end}, months={months}")

    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=int(start.timestamp() * 1000),
        end_str=int(end.timestamp() * 1000),
    )
    if not klines:
        raise RuntimeError(
            f"[OFFLINE] get_historical_klines returned empty list "
            f"for symbol={symbol}, interval={interval}"
        )

    cols = [
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
    ]
    df = pd.DataFrame(klines, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    print(f"[OFFLINE][{interval}] raw_df.shape={df.shape}")

    # CACHE'e yaz
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[OFFLINE] cached klines saved to {cache_path}")

    return df


def build_features_offline(raw_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Offline eğitim için feature engineering.
    Amaç: online tarafta kullanılan feature set'e mümkün olduğunca benzemek.

    Üretilen kolonlar:
    [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        "return_1", "return_5", "return_15",
        "volatility_10", "volatility_30", "buy_ratio",
        "ma_close_10", "ma_close_20", "ma_close_50",
        "price_diff_1", "price_diff_5",
        "volume_change_1", "volume_ma_20",
    ]
    """
    df = raw_df.copy()

    # --- Sıralama ---
    if "open_time" in df.columns:
        df = df.sort_values("open_time").reset_index(drop=True)

    # --- Numerik kolonları float'a çevir (1h cache'de str geliyordu) ---
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Taker buy kolon isimlerini normalize et ---
    # Cache'de: taker_buy_base_volume, taker_buy_quote_volume
    # Online FE'de: taker_buy_base_asset_volume, taker_buy_quote_asset_volume
    if "taker_buy_base_asset_volume" not in df.columns and "taker_buy_base_volume" in df.columns:
        df["taker_buy_base_asset_volume"] = df["taker_buy_base_volume"]

    if "taker_buy_quote_asset_volume" not in df.columns and "taker_buy_quote_volume" in df.columns:
        df["taker_buy_quote_asset_volume"] = df["taker_buy_quote_volume"]

    # --- Basit price / range feature'ları ---
    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)

    # --- Returns ---
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_15"] = df["close"].pct_change(15)

    # --- Volatility (rolling std) ---
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_30"] = df["return_1"].rolling(30).std()

    # --- Buy ratio ---
    # volume = 0 ise bölme hatasına karşı koruma
    vol = df["volume"].replace(0, pd.NA)
    df["buy_ratio"] = df["taker_buy_base_asset_volume"] / vol

    # --- Moving averages ---
    df["ma_close_10"] = df["close"].rolling(10).mean()
    df["ma_close_20"] = df["close"].rolling(20).mean()
    df["ma_close_50"] = df["close"].rolling(50).mean()

    # --- Price diff ---
    df["price_diff_1"] = df["close"] - df["close"].shift(1)
    df["price_diff_5"] = df["close"] - df["close"].shift(5)

    # --- Volume features ---
    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_ma_20"] = df["volume"].rolling(20).mean()

    # --- NaN temizliği ---
    df = df.dropna().reset_index(drop=True)

    # --- Kolonları sabitle ---
    expected_cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
        "return_1",
        "return_5",
        "return_15",
        "volatility_10",
        "volatility_30",
        "buy_ratio",
        "ma_close_10",
        "ma_close_20",
        "ma_close_50",
        "price_diff_1",
        "price_diff_5",
        "volume_change_1",
        "volume_ma_20",
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[OFFLINE][{interval}] Beklenen kolonlar eksik: {missing}. "
            f"Mevcut kolonlar: {df.columns.tolist()}"
        )

    print(f"[OFFLINE][{interval}] features_df.shape={df[expected_cols].shape}")
    return df[expected_cols]



def apply_anomaly_filter(
    df: pd.DataFrame,
    interval: str,
        contamination: float = 0.02) -> pd.DataFrame:
    """
    IsolationForest ile basit anomali filtresi.
    Hata olursa DF'i aynen döner (ve warn log yazar).
    """
    try:
        # Sayısal kolonları seç
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            print(
                f"[OFFLINE] Uyarı: AnomalyDetector için numeric kolon bulunamadı, filtre atlanıyor.")
            return df

        iso = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        y_pred = iso.fit_predict(df[num_cols])
        mask = y_pred == 1
        clean_df = df.loc[mask].reset_index(drop=True)
        print(f"[OFFLINE][{interval}] clean_df.shape={clean_df.shape}")
        return clean_df
    except Exception as e:
        print(
            f"[OFFLINE] Uyarı: AnomalyDetector uygun metod bulamadı veya hata verdi "
            f"({e!r}), anomali filtresi uygulanmadan devam ediliyor.")
        return df


def evaluate_model(clf: SGDClassifier, X_val: pd.DataFrame,
                   y_val: pd.Series) -> Dict[str, float]:
    """Accuracy + ROC-AUC hesapla (ikisi de döner, log'a yazarız)."""
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    metrics: Dict[str, float] = {"accuracy": float(acc)}

    # ROC-AUC için pozitif olasılıkları kullan
    try:
        if hasattr(clf, "predict_proba"):  # bazı SGD varyantlarında yok
            y_proba = clf.predict_proba(X_val)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_scores = clf.decision_function(X_val)
            # decision_function çıktısını 0-1 aralığına "sıkıştırma" (kabaca
            # sigmoid)
            with np.errstate(over="ignore"):
                y_proba = 1 / (1 + np.exp(-y_scores))
        else:
            y_proba = None

        if y_proba is not None:
            auc = roc_auc_score(y_val, y_proba)
            metrics["roc_auc"] = float(auc)
    except Exception:
        # Herhangi bir sebeple ROC-AUC hesaplanamazsa boş geç
        pass

    return metrics


def sample_sgd_params(rng: np.random.Generator, mode: str) -> Dict:
    """Mode'a göre SGDClassifier hyperparam sample eder."""
    if mode == "shallow":
        alpha_range = (1e-4, 1e-1)
        max_iter_range = (500, 1500)
    elif mode == "full":
        alpha_range = (1e-5, 5e-1)
        max_iter_range = (800, 2000)
    else:  # deep
        alpha_range = (1e-6, 1.0)
        max_iter_range = (1200, 3000)

    loss = rng.choice(["log_loss", "modified_huber"])
    penalty = rng.choice(["l1", "l2", "elasticnet"])
    alpha = float(
        10 ** rng.uniform(np.log10(alpha_range[0]), np.log10(alpha_range[1])))
    l1_ratio = float(rng.uniform(0.0, 1.0)) if penalty == "elasticnet" else 0.0
    max_iter = int(rng.integers(max_iter_range[0], max_iter_range[1] + 1))
    tol = float(10 ** rng.uniform(-4, -2))  # 1e-4 ile 1e-2 arası

    params = {
        "loss": loss,
        "penalty": penalty,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "tol": tol,
        "random_state": int(rng.integers(1, 1_000_000)),
    }
    return params


# -------------------------------------------------------------------------
# Ana training fonksiyonu
# -------------------------------------------------------------------------


def offline_train_for_interval(
    symbol: str,
    interval: str,
    mode: str,
    model_dir: str = "models",
    n_rounds: int = 8,
    n_iter: int = 600,
    months: int = 6,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> None:
    """
    Tek bir interval için offline pre-train:
    - Veriyi çek
    - Feature engineering + anomaly filter
    - Long & short için SGDClassifier tuning (mode'a göre daha az/çok tur)
    - En iyi modeli kaydet (long/short/best)
    """
    # Mode -> N_ROUNDS, N_ITER (partial_fit step sayısı)
    if mode == "shallow":
        N_ROUNDS, N_ITER = 3, 200
    elif mode == "full":
        N_ROUNDS, N_ITER = 5, 400
    else:  # deep
        N_ROUNDS, N_ITER = 8, 600

    print(f"[OFFLINE][{interval}] N_ROUNDS={N_ROUNDS} | N_ITER={N_ITER}")

    # Ortam (sadece log için)
    try:
        load_environment_variables(silent=True)  # type: ignore[arg-type]
    except TypeError:
        # Eski versiyonlarda silent parametresi olmayabilir
        try:
            load_environment_variables()
        except Exception:
            pass

    # 1) Veri çek
    try:
        raw_df = fetch_klines_offline(
            symbol=symbol,
            interval=interval,
            months=months,          # months argümanını birazdan CLI'dan alacağız
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    except Exception as e:
        print(f"[OFFLINE][{interval}] Kline hatası: {e!r}")
        return

    if len(raw_df) < 1000:
        print(f"[OFFLINE][{interval}] Uyarı: Çok az kline var: {len(raw_df)}")
        return

    # 2) Feature engineering
    try:
        features_df = build_features_offline(raw_df, interval=interval)
    except Exception as e:
        print(
            f"[OFFLINE][{interval}] feature engineering sırasında hata: {e!r}")
        return

    # 3) Anomali filtresi
    clean_df = apply_anomaly_filter(features_df, interval=interval)

    # 4) Target (y_long, y_short) oluşturma
    target_col = "return_5"
    if target_col not in clean_df.columns:
        raise RuntimeError(
            f"[OFFLINE][{interval}] Hedef kolonu {target_col} bulunamadı.")

    fwd_ret = clean_df[target_col].astype(float)

    # Çok küçük hareketleri (ör. mutlak getiri < 0.1%) "gürültü" sayıp
    # atıyoruz.
    thr = 0.001  # 0.1%
    mask = fwd_ret.abs() > thr

    if mask.sum() < 1000:
        print(
            f"[OFFLINE][{interval}] Uyarı: threshold sonrası yeterli örnek kalmadı: {mask.sum()}")

    clean_df = clean_df.loc[mask].reset_index(drop=True)
    fwd_ret = fwd_ret.loc[mask].reset_index(drop=True)

    # Long sinyali: getiri > +thr  -> 1, aksi 0
    # Short sinyali: getiri < -thr -> 1, aksi 0
    y_long = (fwd_ret > thr).astype(int)
    y_short = (fwd_ret < -thr).astype(int)

    # X: bütün özellik kolonları (hepsi numeric olsun)
    feature_cols = [
        c for c in clean_df.columns
        if c not in ["open_time", "close_time", "ignore"]
    ]
    X = clean_df[feature_cols].astype(float)

    print(
        f"[OFFLINE][{interval}] X.shape={X.shape}, "
        f"y_long pozitif oran={y_long.mean():.3f}, "
        f"y_short pozitif oran={y_short.mean():.3f}"
    )

    # Train / Val split (aynı X split'i hem long hem short için kullan)
    X_train, X_val, y_long_train, y_long_val = train_test_split(
        X, y_long, test_size=0.2, shuffle=False
    )
    _, _, y_short_train, y_short_val = train_test_split(
        X, y_short, test_size=0.2, shuffle=False
    )

    print(
        f"[OFFLINE][{interval}] Train={len(X_train)}, Val={len(X_val)} "
        f"(long & short aynı X split'i kullanıyor)"
    )

    rng = np.random.default_rng(seed=42)

    # Globaller
    best_score: float = 0.5
    best_side: str = "long"
    best_params: Dict = {}
    best_model: Optional[SGDClassifier] = None

    long_model: Optional[SGDClassifier] = None
    short_model: Optional[SGDClassifier] = None

    # -------------------------------------------------------------------------
    # Long / Short için ayrı training
    # -------------------------------------------------------------------------
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib
    import time
    import os
    import config as app_config

    # X ve y'leri hazırlama
    X = clean_df[feature_cols].astype(float).values
    y_long_arr = y_long.astype(int).values
    y_short_arr = y_short.astype(int).values

    # Aynı X split'ini hem long hem short için kullan
    X_train, X_val, y_long_train, y_long_val, y_short_train, y_short_val = train_test_split(
        X,
        y_long_arr,
        y_short_arr,
        test_size=0.20,
        shuffle=False,  # zaman serisi bozulmasın
    )

    n_train = len(X_train)
    n_val = len(X_val)

    print(
        f"[OFFLINE][{interval}] X.shape={X.shape}, "
        f"y_long pozitif oran={y_long_arr.mean():.3f}, "
        f"y_short pozitif oran={y_short_arr.mean():.3f}"
    )
    print(
        f"[OFFLINE][{interval}] Train={n_train}, Val={n_val} "
        f"(long & short aynı X split'i kullanıyor)"
    )

    # MODE'a göre N_ROUNDS / N_ITER (üstte de ayarlıyor olabilirsin,
    # burada tekrar etmek istemezsen bu kısmı silebilirsin)
    if mode == "deep":
        N_ROUNDS = 8
        N_ITER = 600
    elif mode == "full":
        N_ROUNDS = 5
        N_ITER = 400
    else:  # shallow
        N_ROUNDS = 3
        N_ITER = 200

    print(f"[OFFLINE][{interval}] N_ROUNDS={N_ROUNDS} | N_ITER={N_ITER}")

    # Küçük helper: SGD hiperparam sampling (sende zaten varsa onu kullan)
    rng = np.random.RandomState(42)

    def sample_sgd_params(mode: str) -> dict:
        losses = ["log_loss", "modified_huber"]
        penalties = ["l2", "l1", "elasticnet"]

        loss = rng.choice(losses)
        penalty = rng.choice(penalties)

        # mode'a göre biraz farklı aralıklar
        if mode == "deep":
            alpha = float(10 ** rng.uniform(-4, -1))  # 1e-4 .. 1e-1
            max_iter = int(rng.randint(800, 2500))
        elif mode == "full":
            alpha = float(10 ** rng.uniform(-4.5, -1.5))
            max_iter = int(rng.randint(600, 2000))
        else:  # shallow
            alpha = float(10 ** rng.uniform(-5, -2))
            max_iter = int(rng.randint(400, 1500))

        l1_ratio = float(rng.uniform(0.1, 0.9))
        tol = float(10 ** rng.uniform(-4, -2))  # 1e-4 .. 1e-2

        return {
            "loss": loss,
            "penalty": penalty,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": int(rng.randint(0, 1_000_000)),
        }

    # Tüm side'lar için global en iyi model
    global_best_score = 0.5
    best_side = "long"
    best_params = {}
    long_model_path: str = ""
    short_model_path: str = ""
    best_model_path: str = ""

    t_interval_start = time.perf_counter()

    for side in ["long", "short"]:
        if side == "long":
            y_train_side = y_long_train
            y_val_side = y_long_val
        else:
            y_train_side = y_short_train
            y_val_side = y_short_val

        print(f"[OFFLINE][{interval}][{side}] ---- ROUND'lar başlıyor ----")

        best_score_side = 0.5
        best_model_side = None
        best_params_side = None

        # ROUND döngüsü
        for round_idx in range(1, N_ROUNDS + 1):
            params = sample_sgd_params(mode=mode)

            model = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=0.001,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
            )
            # Paramları overwrite et
            model.set_params(**params)

            # İlk partial_fit için class listesi
            classes = np.array([0, 1], dtype=int)

            # Train setini N_ITER kadar küçük batch'e böl
            indices = np.arange(n_train)
            rng.shuffle(indices)
            batches = np.array_split(indices, N_ITER)

            eval_every = max(1, N_ITER // 10)  # her ~%10 adımda bir eval

            for iter_idx, batch_idx in enumerate(batches, start=1):
                if batch_idx.size == 0:
                    continue

                X_batch = X_train[batch_idx]
                y_batch = y_train_side[batch_idx]

                if iter_idx == 1:
                    # İlk adımda sınıfları ver
                    model.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    model.partial_fit(X_batch, y_batch)

                # Belirli aralıklarda validation skoru hesapla
                if iter_idx % eval_every == 0 or iter_idx == N_ITER:
                    y_val_pred = model.predict(X_val)
                    acc = accuracy_score(y_val_side, y_val_pred)

                    # proba / decision_function
                    if hasattr(model, "predict_proba"):
                        y_val_proba = model.predict_proba(X_val)[:, 1]
                    else:
                        y_val_proba = model.decision_function(X_val)
                        # decision'ı 0-1 aralığına squash edelim
                        y_val_proba = 1 / (1 + np.exp(-y_val_proba))

                    try:
                        auc = roc_auc_score(y_val_side, y_val_proba)
                    except ValueError:
                        # Tüm label'lar tek sınıf ise AUC hesaplanamıyor
                        auc = 0.5

                    score = auc  # ana metrik AUC

                    print(
                        f"[OFFLINE][{interval}][{side}] "
                        f"Round {round_idx}/{N_ROUNDS} | Iter {iter_idx}/{N_ITER} | "
                        f"score={score:.4f} (acc={acc:.4f}, auc={auc:.4f})")

                    # side için en iyi model güncelle
                    if score > best_score_side:
                        best_score_side = score
                        best_model_side = copy.deepcopy(model)  # derin kopya
                        best_params_side = params

            # ROUND bitti logu
            print(
                f"[OFFLINE][{interval}][{side}] ROUND {round_idx} bitti. "
                f"Şu anki side en iyi skor={best_score_side:.4f}"
            )

        # Side için model path ve global en iyi side güncelle
        if best_model_side is None:
            # Fail-safe: hiç iyileşme olmadıysa son modeli kullan
            best_model_side = model
            best_params_side = params

        side_model_path = os.path.join(
            model_dir, f"online_model_{interval}_{side}.joblib")
        joblib.dump(best_model_side, side_model_path)

        if side == "long":
            long_model_path = side_model_path
        else:
            short_model_path = side_model_path

        print(
            f"[OFFLINE][{interval}][{side}] Side training bitti. "
            f"En iyi skor={best_score_side:.4f}, model={side_model_path}"
        )

        # Global en iyi side seçimi (AUC bazlı)
        if best_score_side > global_best_score:
            global_best_score = best_score_side
            best_side = side
            best_params = best_params_side if best_params_side is not None else {}

    # Global en iyi side'ı ayrı best model olarak kaydet
    if best_side == "long":
        best_model_src = long_model_path
    else:
        best_model_src = short_model_path

    best_model_path = os.path.join(
        model_dir, f"online_model_{interval}_best.joblib")
    # Kopyala
    best_model_obj = joblib.load(best_model_src)
    joblib.dump(best_model_obj, best_model_path)

    t_interval_end = time.perf_counter()
    elapsed = t_interval_end - t_interval_start

    print(
        f"[OFFLINE][{interval}] TRAINING TAMAMLANDI. "
        f"En iyi skor={global_best_score:.4f}, seçilen yön={best_side}, en iyi paramlar={best_params}"
    )
    print(f"[OFFLINE][{interval}] Long model kaydedildi:   {long_model_path}")
    print(f"[OFFLINE][{interval}] Short model kaydedildi:  {short_model_path}")
    print(f"[OFFLINE][{interval}] Best  model kaydedildi:  {best_model_path}")
    print(
        f"[OFFLINE][{interval}] Interval training bitti. "
        f"Süre: {elapsed:.1f} sn (~{elapsed/60:.1f} dk)"
    )

    # ==============================================================
    #  LSTM HYBRID EĞİTİM (opsiyonel)
    # ==============================================================
    use_lstm_hybrid = False
    lstm_meta = None

    # Sadece HYBRID_MODE env=true ise LSTM dene
    hybrid_env = os.environ.get("HYBRID_MODE", "false").lower()
    if hybrid_env in ("1", "true", "yes", "on"):
        print(f"[OFFLINE][{interval}] HYBRID_MODE=true -> LSTM hibrit eğitimi denenecek.")
        try:
            from data.lstm_hybrid import train_lstm_hybrid

            lstm_meta = train_lstm_hybrid(
                features_df=clean_df,
                y_long=y_long,
                y_short=y_short,
                interval=interval,
                model_dir=str(model_dir),
            )
            use_lstm_hybrid = True
            print(f"[OFFLINE][{interval}] LSTM training OK: {lstm_meta}")
        except Exception as e:
            print(f"[OFFLINE][{interval}] LSTM training FAILED: {e!r}")
            use_lstm_hybrid = False
    else:
        print(f"[OFFLINE][{interval}] HYBRID_MODE=false -> LSTM eğitimi atlanıyor.")

    # ==============================================================
    #  META DOSYASINI YAZ
    # ==============================================================
    meta_path = model_dir / f"model_meta_{interval}.json"
    meta = {
        "best_auc": float(best_score),
        "best_side": best_side,
        "use_lstm_hybrid": bool(use_lstm_hybrid),
    }
    if lstm_meta is not None:
        meta.update(lstm_meta)

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OFFLINE][{interval}] Meta kaydedildi: {meta_path}")

    # ------------------------------------------------------------------
    # Meta JSON kaydı (best_auc, best_side, use_lstm_hybrid)
    # ------------------------------------------------------------------
    meta_path = Path(model_dir) / f"model_meta_{interval}.json"
    meta_data = {
        "best_auc": float(best_score),
        "best_side": best_side,
        "use_lstm_hybrid": False,  # şimdilik sadece SGD hibrit; LSTM aktif olunca True yaparız
    }
    meta_path.write_text(json.dumps(meta_data, indent=2))
    print(f"[OFFLINE][{interval}] Meta kaydedildi: {meta_path}")

# -------------------------------------------------------------------------
# CLI / main
# -------------------------------------------------------------------------


def main():
    print("[MAIN] offline_pretrain_six_months başlıyor.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="deep", help="Eğitim modu")
    parser.add_argument("--interval", type=str, help="1m, 5m, 15m vs")
    parser.add_argument("--months", type=int, default=6,
                        help="Kaç aylık veri çekileceği (default: 6)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Cache kullanma, Binance'ten direkt çek")
    parser.add_argument("--force-refresh-cache", action="store_true",
                        help="Cache dosyasını yok sayıp yeniden çek")
    args = parser.parse_args()

    # ---- ARGUMENTS ----
    mode = args.mode
    interval_env = os.environ.get("INTERVAL")
    months = args.months
    use_cache = not args.no_cache
    force_refresh = args.force_refresh_cache

    # ---- INTERVAL SET ----
    if interval_env:
        intervals = [interval_env]
        print(f"[MAIN] INTERVAL env bulundu → Sadece {intervals} eğitilecek.")
    elif args.interval:
        intervals = [args.interval]
        print(f"[MAIN] Arg interval bulundu → {intervals}")
    else:
        intervals = ["1m", "5m"]
        print(f"[MAIN] INTERVAL verilmedi → default {intervals}")

    # ---- MODEL DIR ----
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    print(f"Offline pre-train başlıyor | mode={mode} | intervals={intervals}")

    for interval in intervals:
        print(f"\n========== INTERVAL: {interval} | MODE: {mode} ==========\n")

        offline_train_for_interval(
            symbol="BTCUSDT",
            interval=interval,
            mode=mode,
            model_dir=model_dir,
            n_rounds=8,
            n_iter=600,
            months=months,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    print("Offline pre-train tamamlandı.")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# Backwards compatibility: eski kod build_features_offline() çağırıyor.
# Yeni FE fonksiyonumuz build_features_offline()'ı sarmalıyoruz.
# ----------------------------------------------------------------------
def build_features_offline(raw_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    return build_features_offline(raw_df, interval)
