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
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

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
        print("[load_env] INFO: config.load_env.load_environment_variables bulunamadı, atlanıyor.")


# -------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -------------------------------------------------------------------------


def fetch_klines_offline(
    symbol: str,
    interval: str,
    months: int = 6,
) -> pd.DataFrame:
    """
    Binance public API kullanarak son `months` aylık veriyi çeker.
    Kimlik doğrulama gerektirmez (public endpoint).

    Dönen DF kolonları:
    [open_time, open, high, low, close, volume,
     close_time, quote_asset_volume, number_of_trades,
     taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
    """
    if Client is None:
        raise RuntimeError("python-binance bulunamadı. `pip install python-binance` gerekli.")

    client = Client(api_key="", api_secret="")  # public-only

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=30 * months)

    print(
        f"[OFFLINE] fetch_klines_offline: symbol={symbol}, interval={interval}, "
        f"start={start_dt}, end={end_dt}"
    )

    klines: List[List] = client.get_historical_klines(
        symbol,
        interval,
        start_str=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_str=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if not klines:
        raise RuntimeError(f"[OFFLINE] Hiç kline verisi çekilemedi. symbol={symbol}, interval={interval}")

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
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    # Tip dönüşümleri
    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_asset_volume", "taker_buy_base_asset_volume",
                  "taker_buy_quote_asset_volume"]
    int_cols = ["open_time", "close_time", "number_of_trades"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["ignore"] = pd.to_numeric(df["ignore"], errors="coerce")

    print(f"[OFFLINE][{interval}] raw_df.shape={df.shape}")
    return df


def build_features_from_raw(raw_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Runtime feature_engineering'e benzer bir set üretir.

    Çıktı kolonları:
    ['open_time','open','high','low','close','volume','close_time',
     'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
     'taker_buy_quote_asset_volume','ignore',
     'return_1','return_5','return_15',
     'volatility_10','volatility_30','buy_ratio',
     'ma_close_10','ma_close_20','ma_close_50',
     'price_diff_1','price_diff_5','volume_change_1','volume_ma_20']
    """
    df = raw_df.copy()

    # Basit getiriler
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_15"] = df["close"].pct_change(15)

    # Volatilite (return_1 üzerinden)
    df["volatility_10"] = df["return_1"].rolling(window=10).std()
    df["volatility_30"] = df["return_1"].rolling(window=30).std()

    # Buy ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        df["buy_ratio"] = df["taker_buy_base_asset_volume"] / df["volume"]
    df["buy_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # MA'ler
    df["ma_close_10"] = df["close"].rolling(window=10).mean()
    df["ma_close_20"] = df["close"].rolling(window=20).mean()
    df["ma_close_50"] = df["close"].rolling(window=50).mean()

    # Fiyat farkları
    df["price_diff_1"] = df["close"].diff(1)
    df["price_diff_5"] = df["close"].diff(5)

    # Hacim değişimi
    df["volume_change_1"] = df["volume"].pct_change(1)

    # Hacim MA
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()

    # NaN'li satırları at
    df = df.dropna().reset_index(drop=True)

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

    df = df[expected_cols]
    print(f"[OFFLINE][{interval}] features_df.shape={df.shape}")
    return df


def apply_anomaly_filter(df: pd.DataFrame, interval: str, contamination: float = 0.02) -> pd.DataFrame:
    """
    IsolationForest ile basit anomali filtresi.
    Hata olursa DF'i aynen döner (ve warn log yazar).
    """
    try:
        # Sayısal kolonları seç
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            print(f"[OFFLINE] Uyarı: AnomalyDetector için numeric kolon bulunamadı, filtre atlanıyor.")
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
            f"({e!r}), anomali filtresi uygulanmadan devam ediliyor."
        )
        return df


def evaluate_model(clf: SGDClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
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
            # decision_function çıktısını 0-1 aralığına "sıkıştırma" (kabaca sigmoid)
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
    alpha = float(10 ** rng.uniform(np.log10(alpha_range[0]), np.log10(alpha_range[1])))
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
    interval: str,
    mode: str = "shallow",
    symbol: str = "BTCUSDT",
    use_lstm_hybrid: bool = False,
    model_dir: str = "models",
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
        raw_df = fetch_klines_offline(symbol=symbol, interval=interval, months=6)
    except Exception as e:
        print(f"[OFFLINE][{interval}] Kline hatası: {e!r}")
        return

    if len(raw_df) < 1000:
        print(f"[OFFLINE][{interval}] Uyarı: Çok az kline var: {len(raw_df)}")
        return

    # 2) Feature engineering
    try:
        features_df = build_features_from_raw(raw_df, interval=interval)
    except Exception as e:
        print(f"[OFFLINE][{interval}] feature engineering sırasında hata: {e!r}")
        return

    # 3) Anomali filtresi
    clean_df = apply_anomaly_filter(features_df, interval=interval)

    # 4) Target (y_long, y_short) oluşturma
    target_col = "return_5"
    if target_col not in clean_df.columns:
        raise RuntimeError(f"[OFFLINE][{interval}] Hedef kolonu {target_col} bulunamadı.")

    fwd_ret = clean_df[target_col].astype(float)

    # Çok küçük hareketleri (ör. mutlak getiri < 0.1%) "gürültü" sayıp atıyoruz.
    thr = 0.001  # 0.1%
    mask = fwd_ret.abs() > thr

    if mask.sum() < 1000:
        print(f"[OFFLINE][{interval}] Uyarı: threshold sonrası yeterli örnek kalmadı: {mask.sum()}")

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

    # --------------------------------------------------
    # Long / Short için ayrı training
    # --------------------------------------------------
    for side, (y_train, y_val) in {
        "long": (y_long_train, y_long_val),
        "short": (y_short_train, y_short_val),
    }.items():
        print(f"[OFFLINE][{interval}][{side}] ---- ROUND'lar başlıyor ----")

        side_best_score = 0.5
        side_best_params: Dict = {}
        side_best_model: Optional[SGDClassifier] = None

        for round_idx in range(1, N_ROUNDS + 1):
            params = sample_sgd_params(rng, mode=mode)
            clf = SGDClassifier(**params)

            # partial_fit ile iteratif eğitim (roc-auc trendini takip için)
            classes = np.array([0, 1], dtype=int)
            last_score_for_round = 0.5

            for iter_idx in range(1, N_ITER + 1):
                clf.partial_fit(X_train, y_train, classes=classes)

                if iter_idx % 20 == 0 or iter_idx == N_ITER:
                    metrics = evaluate_model(clf, X_val, y_val)
                    score = metrics.get("roc_auc", metrics.get("accuracy", 0.0))
                    last_score_for_round = score

                    print(
                        f"[OFFLINE][{interval}][{side}] "
                        f"Round {round_idx}/{N_ROUNDS} | Iter {iter_idx}/{N_ITER} | "
                        f"score={score:.4f} "
                        f"(acc={metrics.get('accuracy', float('nan')):.4f}, "
                        f"auc={metrics.get('roc_auc', float('nan')):.4f})"
                    )

            # Round sonu -> side bazlı en iyiyi güncelle
            if last_score_for_round > side_best_score:
                side_best_score = last_score_for_round
                side_best_params = params
                side_best_model = clf

            print(
                f"[OFFLINE][{interval}][{side}] ROUND {round_idx} bitti. "
                f"Şu anki side en iyi skor={side_best_score:.4f}"
            )

        # Side training tamam -> global objelere yaz
        if side == "long":
            long_model = side_best_model
        else:
            short_model = side_best_model

        # Global en iyi yönü güncelle
        if side_best_model is not None and side_best_score >= best_score:
            best_score = side_best_score
            best_side = side
            best_params = side_best_params
            best_model = side_best_model

    # Eğer hiçbir model oluşmadıysa (herhangi bir sebeple) çık
    if best_model is None or long_model is None or short_model is None:
        print(f"[OFFLINE][{interval}] Uyarı: Hiç model oluşturulamadı, kaydetme atlanıyor.")
        return

    print(
        f"[OFFLINE][{interval}] TRAINING TAMAMLANDI. "
        f"En iyi skor={best_score:.4f}, seçilen yön={best_side}, "
        f"en iyi paramlar={best_params}"
    )

    # LSTM hibrit flag'i (şimdilik sadece log)
    if use_lstm_hybrid:
        print(
            f"[OFFLINE][{interval}] use_lstm_hybrid=True - "
            f"LSTM hibrit entegrasyonu için iskelet hazır, şu an sadece SGD modelleri kaydediliyor."
        )

    # Long & short & best modelleri ayrı kaydet
    base_name = f"online_model_{interval}"
    long_path = f"{model_dir}/{base_name}_long.joblib"
    short_path = f"{model_dir}/{base_name}_short.joblib"
    best_path = f"{model_dir}/{base_name}_best.joblib"

    joblib.dump(long_model, long_path)
    joblib.dump(short_model, short_path)
    joblib.dump(best_model, best_path)

    print(f"[OFFLINE][{interval}] Long model kaydedildi:   {long_path}")
    print(f"[OFFLINE][{interval}] Short model kaydedildi:  {short_path}")
    print(f"[OFFLINE][{interval}] Best  model kaydedildi:  {best_path}")


# -------------------------------------------------------------------------
# CLI / main
# -------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BTCUSDT offline pre-train (6 ay x multi-interval, long/short ayrı)."
    )

    parser.add_argument(
        "--mode",
        choices=["shallow", "full", "deep"],
        default="shallow",
        help="Eğitim modu: shallow (hızlı), full (daha detaylı), deep (agresif).",
    )

    parser.add_argument(
        "--intervals",
        type=str,
        default="5m",
        help="Virgülle ayrılmış interval listesi, örn: '1m,5m,15m,1h'",
    )

    parser.add_argument(
        "--use-lstm-hybrid",
        action="store_true",
        help="LSTM + SGD hibrit offline pretrain'i aktifleştir (şimdilik sadece SGD eğitimi, log'ta belirtilir).",
    )

    args = parser.parse_args()
    mode = args.mode
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]
    use_lstm_hybrid = bool(getattr(args, "use_lstm_hybrid", False))

    project_root = None
    try:
        import os

        project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
    except Exception:
        pass

    print(f"Offline pre-train başlıyor | mode={mode} | intervals={intervals}")
    if project_root:
        print(f"Çalışma klasörü: {project_root}")

    start_ts = time.time()

    for interval in intervals:
        print(f"\n========== INTERVAL: {interval} | MODE: {mode} ==========")
        t0 = time.time()
        offline_train_for_interval(
            interval=interval,
            mode=mode,
            symbol="BTCUSDT",
            use_lstm_hybrid=use_lstm_hybrid,
            model_dir="models",
        )
        t1 = time.time()
        elapsed = t1 - t0
        print(
            f"[OFFLINE][{interval}] Interval training bitti. Süre: {elapsed:.1f} sn "
            f"(~{elapsed/60:.1f} dk)"
        )

    total_elapsed = time.time() - start_ts
    print(f"\nOffline pre-train tamamlandı. Toplam süre: {total_elapsed:.1f} sn (~{total_elapsed/60:.1f} dk)")


if __name__ == "__main__":
    main()

