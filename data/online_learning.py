"""
Online incremental learning (SGD tabanlı) modülü.

- Offline pretrain sonrası üretilen:
    models/online_model_{interval}_long.joblib
    models/online_model_{interval}_short.joblib
    models/online_model_{interval}_best.joblib

  dosyalarından, **online taraf için** öncelikle
  `online_model_{interval}_best.joblib` yüklenir.

- Eğer interval'e özel best model yoksa:
    1) models/online_model_best.joblib
    2) models/online_model.joblib
  sırasıyla denenir.

- Online güncellemeler her partial_update sonrasında
  klasik path'e:
    models/online_model.joblib
  olarak kaydedilir (Cloud Run mevcut akışı bozulmasın diye).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

# Sistem log formatına uyması için "system" logger'ını kullanıyoruz
logger = logging.getLogger("system")


ArrayLike = Union[np.ndarray, Sequence[float]]


class OnlineLearner:
    """
    SGDClassifier tabanlı online learner.

    Özellikler:
    - interval'e göre offline pretrain best modeli yükler.
    - partial_fit ile streaming (online) güncelleme.
    - predict_proba ile p_buy / p_sell hesaplama.
    """

    def __init__(
        self,
        model_dir: str = "models",
        base_model_name: str = "online_model",
        interval: str = "1m",
        n_classes: int = 2,
    ) -> None:
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.interval = interval
        self.n_classes = n_classes

        # sklearn modeli
        self.model: SGDClassifier = self._load_or_create_model()

        # Özellik isimleri (DataFrame ile uyum için)
        self.feature_columns: Optional[List[str]] = None

        # İlk partial_fit çağrısında classes parametresi verilecek mi?
        self._is_fitted: bool = hasattr(self.model, "classes_")

        # Offline meta (AUC, best_side vs.)
        self.meta: Dict[str, Any] = self._load_meta()

        logger.info(
            "[ONLINE] OnlineLearner initialized with model_dir=%s, "
            "base_model_name=%s, interval=%s, n_classes=%d",
            self.model_dir,
            self.base_model_name,
            self.interval,
            self.n_classes,
        )

        if self.meta:
            logger.info(
                "[ONLINE] Loaded offline meta for interval=%s: %s",
                self.interval,
                json.dumps(self.meta, ensure_ascii=False),
            )

    # ------------------------------------------------------------------
    # Model & meta yükleme
    # ------------------------------------------------------------------
    def _candidate_model_paths(self) -> List[str]:
        """
        Online başlangıç modeli için denenecek path sırası.
        Öncelik:
            1) online_model_{interval}_best.joblib
            2) online_model_best.joblib
            3) online_model.joblib
        """
        paths = [
            os.path.join(self.model_dir, f"{self.base_model_name}_{self.interval}_best.joblib"),
            os.path.join(self.model_dir, f"{self.base_model_name}_best.joblib"),
            os.path.join(self.model_dir, f"{self.base_model_name}.joblib"),
        ]
        return paths

    def _load_or_create_model(self) -> SGDClassifier:
        """
        Diskten uygun bir modeli yüklemeye çalışır, bulamazsa
        sıfırdan SGDClassifier oluşturur.
        """
        for path in self._candidate_model_paths():
            if os.path.exists(path):
                try:
                    model = joblib.load(path)
                    logger.info("[ONLINE] Loaded existing online model from %s", path)
                    return model
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[ONLINE] Failed to load model from %s: %r. "
                        "Trying next candidate...",
                        path,
                        exc,
                    )

        # Hiç model bulunamadı, yeni oluştur
        logger.info("[ONLINE] No existing model found. Creating new SGDClassifier.")
        model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )
        return model

    def _meta_path(self) -> str:
        """
        Offline pretrain sırasında yazılan meta file path'i.
        Örneğin:
            models/model_meta_5m.json
        """
        return os.path.join(self.model_dir, f"model_meta_{self.interval}.json")

    def _load_meta(self) -> Dict[str, Any]:
        """
        Offline pretraining sırasında yazılmış meta verisini okur.
        Örn:
        {
            "interval": "5m",
            "best_auc": 0.71,
            "best_side": "short",
            "mode": "deep",
            "n_samples": 27868
        }
        """
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return {}

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[ONLINE] Failed to load meta file %s: %r", meta_path, exc
            )
            return {}

    # ------------------------------------------------------------------
    # Feature kolonları
    # ------------------------------------------------------------------
    def set_feature_columns(self, columns: Sequence[str]) -> None:
        """
        Online tarafta DataFrame ile predict/partial_update yapılırken
        hangi kolonların kullanılacağını belirler.
        """
        self.feature_columns = list(columns)
        logger.info(
            "[ONLINE] feature_columns set with %d columns.", len(self.feature_columns)
        )

    # ------------------------------------------------------------------
    # Veri sanitizasyonu
    # ------------------------------------------------------------------
    def _sanitize_X_y(
        self,
        X: Union[np.ndarray, "Any"],
        y: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        - Pandas DataFrame ise feature_columns sırasına göre np.array'e çevirir.
        - float'a cast eder.
        - inf / -inf değerleri eler.
        - NaN içeren satırları düşer.

        y verilmişse, X ile aynı satır maskesini uygular.
        """
        # DataFrame ise -> ilgili kolonları sırala
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover
            pd = None

        if pd is not None and isinstance(X, pd.DataFrame):
            if self.feature_columns is not None:
                missing = [c for c in self.feature_columns if c not in X.columns]
                if missing:
                    logger.warning(
                        "[ONLINE] Some feature columns are missing in incoming X: %s",
                        missing,
                    )
                cols = [c for c in self.feature_columns if c in X.columns]
                X_arr = X[cols].to_numpy(dtype=float)
            else:
                X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples_before = X_arr.shape[0]

        # inf'leri NaN yap
        X_arr[~np.isfinite(X_arr)] = np.nan

        # satır bazlı NaN maskesi
        row_mask = ~np.isnan(X_arr).any(axis=1)
        X_clean = X_arr[row_mask]

        y_clean: Optional[np.ndarray] = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != n_samples_before:
                raise ValueError(
                    f"[ONLINE] X and y have different lengths: {n_samples_before} vs {y_arr.shape[0]}"
                )
            y_clean = y_arr[row_mask]

        removed = n_samples_before - X_clean.shape[0]
        if removed > 0:
            logger.warning(
                "[ONLINE] Dropped %d rows due to NaN/inf in online data. Remaining: %d",
                removed,
                X_clean.shape[0],
            )

        return X_clean, y_clean

    # ------------------------------------------------------------------
    # Eğitim / güncelleme
    # ------------------------------------------------------------------
    def initial_fit(
        self,
        X: Union[np.ndarray, "Any"],
        y: ArrayLike,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Offline pretrain sonrasında modeli ilk defa X, y ile fit etmek için
        kullanılabilir. (Şu an offline pretrain kendi SGD loop'unu
        kullandığı için çok kritik değil ama geriye dönük uyumluluk için
        bıraktım.)
        """
        X_clean, y_clean = self._sanitize_X_y(X, y)
        if X_clean.shape[0] == 0:
            logger.warning("[ONLINE] initial_fit skipped: no valid samples.")
            return

        if classes is None:
            classes = list(range(self.n_classes))

        logger.info(
            "[ONLINE] initial_fit called with %d samples, %d features.",
            X_clean.shape[0],
            X_clean.shape[1],
        )

        self.model.partial_fit(X_clean, y_clean, classes=np.asarray(classes, dtype=int))
        self._is_fitted = True

    def partial_update(
        self,
        X: Union[np.ndarray, "Any"],
        y: ArrayLike,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Streaming online öğrenme adımı.

        - X, y sanitize edilir (NaN/inf temizlenir).
        - İsteğe bağlı batch_size ile küçük batch'ler halinde partial_fit yapılır.
        - Her çağrı sonunda model models/online_model.joblib olarak kaydedilir.
        """
        X_clean, y_clean = self._sanitize_X_y(X, y)
        if X_clean.shape[0] == 0:
            logger.warning("[ONLINE] partial_update skipped: no valid samples.")
            return

        n_samples, n_features = X_clean.shape
        logger.info(
            "[ONLINE] partial_update with %d samples, %d features.",
            n_samples,
            n_features,
        )

        if batch_size is None or batch_size <= 0:
            batch_size = n_samples

        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        batches = np.array_split(indices, max(1, n_samples // batch_size))

        classes = np.arange(self.n_classes, dtype=int)

        for i, batch_idx in enumerate(batches, start=1):
            if batch_idx.size == 0:
                continue
            X_batch = X_clean[batch_idx]
            y_batch = y_clean[batch_idx]

            if not self._is_fitted:
                self.model.partial_fit(X_batch, y_batch, classes=classes)
                self._is_fitted = True
            else:
                self.model.partial_fit(X_batch, y_batch)

        logger.info(
            "[ONLINE] partial_update completed successfully."
        )
        self.save_model()

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: Union[np.ndarray, "Any"],
    ) -> np.ndarray:
        """
        X için sınıf olasılıklarını döner.
        - SGDClassifier.predict_proba varsa direkt kullanır.
        - Yoksa decision_function + sigmoid ile 2 sınıflı proba üretir.
        """
        if not self._is_fitted:
            raise RuntimeError("[ONLINE] predict_proba called but model is not fitted yet.")

        X_clean, _ = self._sanitize_X_y(X)
        if X_clean.shape[0] == 0:
            raise ValueError("[ONLINE] predict_proba: no valid samples after sanitization.")

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_clean)
            # Eğer sınıf sırası [0,1] değilse, kolonları yeniden sırala
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                if classes != [0, 1]:
                    # 2 sınıf varsayımı
                    idx0 = classes.index(0)
                    idx1 = classes.index(1)
                    proba = proba[:, [idx0, idx1]]
            return proba

        # predict_proba yoksa decision_function kullan
        scores = self.model.decision_function(X_clean)
        # binary senaryoda scores shape (n_samples,) olabilir
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        # sigmoid
        p1 = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        p0 = 1.0 - p1
        proba = np.vstack([p0, p1]).T
        return proba

    # ------------------------------------------------------------------
    # Kaydet / yükle
    # ------------------------------------------------------------------
    @property
    def save_path(self) -> str:
        """
        Online update sonrası kaydedilecek ana model path'i.
        (Cloud Run'da kullanılan klasik path)
        """
        return os.path.join(self.model_dir, f"{self.base_model_name}.joblib")

    def save_model(self) -> None:
        """
        Mevcut modeli disk'e kaydeder.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.save_path)
        logger.info("[ONLINE] Online model saved to %s", self.save_path)

    def reload_from_disk(self) -> None:
        """
        Mevcut save_path'ten modeli tekrar yükler (isteğe bağlı).
        """
        if not os.path.exists(self.save_path):
            logger.warning(
                "[ONLINE] reload_from_disk called but %s does not exist.",
                self.save_path,
            )
            return
        self.model = joblib.load(self.save_path)
        self._is_fitted = hasattr(self.model, "classes_")
        logger.info("[ONLINE] Online model reloaded from %s", self.save_path)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def get_offline_quality(self) -> Optional[float]:
        """
        Offline pretrain sırasında hesaplanan en iyi AUC değerini döner
        (risk manager daha agresif / defansif davranmak için kullanabilir).

        meta dosyasında 'best_auc' yoksa None döner.
        """
        if not self.meta:
            return None
        return float(self.meta.get("best_auc")) if "best_auc" in self.meta else None

    def get_offline_best_side(self) -> Optional[str]:
        """
        Offline pretrain'de long / short tarafında hangi modelin
        daha iyi olduğuna dair bilgi (best_side) döner.
        """
        if not self.meta:
            return None
        side = self.meta.get("best_side")
        return str(side) if side is not None else None

