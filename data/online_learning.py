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


def safe_p_buy(model: Any, X: Any) -> float:
    """
    Modelden güvenli şekilde p_buy üretir.
    Sıra:
      1) predict_proba
      2) decision_function -> sigmoid
      3) predict -> {0,1}
      4) fallback 0.5
    """
    # 1) predict_proba varsa
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)[0, 1]
            return float(p)
        except Exception:
            pass

    # 2) decision_function varsa -> sigmoid
    if hasattr(model, "decision_function"):
        try:
            z = float(model.decision_function(X)[0])
            return float(1.0 / (1.0 + np.exp(-z)))
        except Exception:
            pass

    # 3) predict fallback
    try:
        pred = model.predict(X)[0]
        return 1.0 if int(pred) == 1 else 0.0
    except Exception:
        return 0.5


class OnlineLearner:
    """
    SGDClassifier tabanlı online learner.

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
        load_existing: bool = True,
    ) -> None:
        self.model_dir = model_dir
        self.base_model_name = base_model_name
        self.interval = interval
        self.n_classes = int(n_classes)
        self.load_existing = bool(load_existing)

        # sklearn modeli
        self.model: SGDClassifier = self._load_or_create_model()

        # Özellik isimleri (DataFrame ile uyum için)
        self.feature_columns: Optional[List[str]] = None

        # İlk partial_fit çağrısında classes parametresi verilecek mi?
        self._is_fitted: bool = hasattr(self.model, "classes_")

        # Offline meta (AUC, best_side vs.)
        self.meta: Dict[str, Any] = self._load_meta()

        logger.info(
            "[ONLINE] OnlineLearner initialized model_dir=%s base_model_name=%s interval=%s n_classes=%d load_existing=%s",
            self.model_dir,
            self.base_model_name,
            self.interval,
            self.n_classes,
            self.load_existing,
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
        return [
            os.path.join(self.model_dir, f"{self.base_model_name}_{self.interval}_best.joblib"),
            os.path.join(self.model_dir, f"{self.base_model_name}_best.joblib"),
            os.path.join(self.model_dir, f"{self.base_model_name}.joblib"),
        ]

    def _load_or_create_model(self) -> SGDClassifier:
        """
        Diskten uygun bir modeli yüklemeye çalışır, bulamazsa sıfırdan SGDClassifier oluşturur.
        load_existing=False ise her zaman yeni model oluşturur.
        """
        if self.load_existing:
            for path in self._candidate_model_paths():
                if os.path.exists(path):
                    try:
                        model = joblib.load(path)
                        logger.info("[ONLINE] Loaded existing online model from %s", path)
                        return model
                    except Exception as exc:  # defensive
                        logger.warning(
                            "[ONLINE] Failed to load model from %s: %r. Trying next candidate...",
                            path,
                            exc,
                        )

        # Yeni oluştur
        logger.info("[ONLINE] No existing model found (or load_existing=False). Creating new SGDClassifier.")
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        )

    def _meta_path(self) -> str:
        return os.path.join(self.model_dir, f"model_meta_{self.interval}.json")

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            return {}

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception as exc:  # defensive
            logger.warning("[ONLINE] Failed to load meta file %s: %r", meta_path, exc)
            return {}

    # ------------------------------------------------------------------
    # Feature kolonları
    # ------------------------------------------------------------------
    def set_feature_columns(self, columns: Sequence[str]) -> None:
        self.feature_columns = list(columns)
        logger.info("[ONLINE] feature_columns set with %d columns.", len(self.feature_columns))

    # ------------------------------------------------------------------
    # Veri sanitizasyonu
    # ------------------------------------------------------------------
    def _sanitize_X_y(
        self,
        X: Any,
        y: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        - Pandas DataFrame ise feature_columns sırasına göre np.array'e çevirir.
        - float'a cast eder.
        - inf / -inf -> NaN
        - NaN içeren satırları düşer.
        """
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            pd = None

        if pd is not None and isinstance(X, pd.DataFrame):
            if self.feature_columns is not None:
                cols = [c for c in self.feature_columns if c in X.columns]
                missing = [c for c in self.feature_columns if c not in X.columns]
                if missing:
                    logger.warning("[ONLINE] Missing columns in X (ignored): %s", missing)
                X_arr = X[cols].to_numpy(dtype=float)
            else:
                X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_before = X_arr.shape[0]

        X_arr[~np.isfinite(X_arr)] = np.nan
        row_mask = ~np.isnan(X_arr).any(axis=1)
        X_clean = X_arr[row_mask]

        y_clean: Optional[np.ndarray] = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != n_before:
                raise ValueError(f"[ONLINE] X and y lengths mismatch: {n_before} vs {y_arr.shape[0]}")
            y_clean = y_arr[row_mask]

        removed = n_before - X_clean.shape[0]
        if removed > 0:
            logger.warning("[ONLINE] Dropped %d rows due to NaN/inf. Remaining=%d", removed, X_clean.shape[0])

        return X_clean, y_clean

    # ------------------------------------------------------------------
    # Eğitim / güncelleme
    # ------------------------------------------------------------------
    def initial_fit(
        self,
        X: Any,
        y: ArrayLike,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        X_clean, y_clean = self._sanitize_X_y(X, y)
        if X_clean.shape[0] == 0:
            logger.warning("[ONLINE] initial_fit skipped: no valid samples.")
            return

        if classes is None:
            classes = list(range(self.n_classes))

        y_clean = np.asarray(y_clean, dtype=int)

        u = np.unique(y_clean)
        if u.size < 2:
            logger.warning("[ONLINE] initial_fit skipped: single-class batch label=%s n=%d", int(u[0]), len(y_clean))
            return

        logger.info("[ONLINE] initial_fit with %d samples, %d features.", X_clean.shape[0], X_clean.shape[1])
        self.model.partial_fit(X_clean, y_clean, classes=np.asarray(classes, dtype=int))
        self._is_fitted = True

    def partial_update(
        self,
        X: Any,
        y: ArrayLike,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Streaming online öğrenme adımı.
        - X, y sanitize edilir
        - batch_size ile partial_fit
        - her çağrı sonunda save_model()
        """
        X_clean, y_clean = self._sanitize_X_y(X, y)
        if X_clean.shape[0] == 0 or y_clean is None or len(y_clean) == 0:
            logger.warning("[ONLINE] partial_update skipped: no valid samples.")
            return

        n_samples, n_features = X_clean.shape
        logger.info("[ONLINE] partial_update with %d samples, %d features.", n_samples, n_features)

        # Label distribution debug (ENV)
        dbg = str(os.getenv("ONLINE_LABEL_DBG", "0")).strip().lower() in ("1", "true", "yes", "on")
        if dbg:
            try:
                from collections import Counter
                c = Counter(list(y_clean))
                logger.info("[ONLINE][LABEL_DBG] dist=%s n=%d", dict(c), len(y_clean))
            except Exception:
                pass

        if batch_size is None or batch_size <= 0:
            batch_size = n_samples

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        # n_samples // batch_size -> 0 olmasın
        n_splits = max(1, int(np.ceil(n_samples / float(batch_size))))
        batches = np.array_split(indices, n_splits)

        classes = np.arange(self.n_classes, dtype=int)

        for batch_idx in batches:
            if batch_idx.size == 0:
                continue

            X_batch = X_clean[batch_idx]
            y_batch = np.asarray(y_clean[batch_idx], dtype=int)

            u = np.unique(y_batch)
            if u.size < 2:
                logger.warning("[ONLINE] skip partial_fit: single-class batch label=%s n=%d", int(u[0]), len(y_batch))
                continue

            if not self._is_fitted:
                self.model.partial_fit(X_batch, y_batch, classes=classes)
                self._is_fitted = True
            else:
                self.model.partial_fit(X_batch, y_batch)

        logger.info("[ONLINE] partial_update completed successfully.")
        self.save_model()

    # ------------------------------------------------------------------
    # Tahmin
    # ------------------------------------------------------------------
    def predict_proba(self, X: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("[ONLINE] predict_proba called but model is not fitted yet.")

        X_clean, _ = self._sanitize_X_y(X)
        if X_clean.shape[0] == 0:
            raise ValueError("[ONLINE] predict_proba: no valid samples after sanitization.")

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_clean)

            # classes_ sırası [0,1] değilse reorder
            if hasattr(self.model, "classes_"):
                classes = list(self.model.classes_)
                if classes != [0, 1] and 0 in classes and 1 in classes:
                    idx0 = classes.index(0)
                    idx1 = classes.index(1)
                    proba = proba[:, [idx0, idx1]]

            return proba

        # predict_proba yoksa decision_function + sigmoid
        scores = self.model.decision_function(X_clean)
        if np.ndim(scores) == 1:
            scores = scores.reshape(-1, 1)

        p1 = 1.0 / (1.0 + np.exp(-scores[:, 0]))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    # ------------------------------------------------------------------
    # Kaydet / yükle
    # ------------------------------------------------------------------
    @property
    def save_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.base_model_name}.joblib")

    def save_model(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.save_path)
        logger.info("[ONLINE] Online model saved to %s", self.save_path)

    def reload_from_disk(self) -> None:
        if not os.path.exists(self.save_path):
            logger.warning("[ONLINE] reload_from_disk: %s does not exist.", self.save_path)
            return
        self.model = joblib.load(self.save_path)
        self._is_fitted = hasattr(self.model, "classes_")
        logger.info("[ONLINE] Online model reloaded from %s", self.save_path)

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------
    def get_offline_quality(self) -> Optional[float]:
        if not self.meta:
            return None
        return float(self.meta.get("best_auc")) if "best_auc" in self.meta else None

    def get_offline_best_side(self) -> Optional[str]:
        if not self.meta:
            return None
        side = self.meta.get("best_side")
        return str(side) if side is not None else None
