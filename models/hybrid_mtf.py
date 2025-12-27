# models/hybrid_mtf.py

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

system_logger = logging.getLogger("system")


class MultiTimeframeHybridEnsemble:
    """
    Çoklu timeframe (1m, 5m, 15m, 1h) HybridModel ensemble sınıfı.

    - Her interval için ayrı bir model bekler (model.predict_proba(X) çağrılabilir olmalı).
    - predict_mtf:
        * Her intervalde model.predict_proba(X_itv) çağırır
        * Son barın ([-1]) olasılığını alır
        * Interval weight'i dışarıdan (HybridMTF) verilebilir
        * Ağırlıklı ortalama ile ensemble p üretir
        * Detayları mtf_debug içinde döner
    """

    def __init__(
        self,
        models_by_interval: Dict[str, Any],
        logger_: logging.Logger | None = None,
    ) -> None:
        self.models_by_interval = models_by_interval or {}
        self.logger = logger_ or system_logger

    @staticmethod
    def _extract_last_prob(proba_arr: Any) -> float:
        """
        predict_proba çıktısından son bar olasılığını float olarak çıkarır.
        """
        arr = np.asarray(proba_arr)

        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            arr = arr[:, -1]

        if arr.size == 0:
            return 0.5

        return float(arr[-1])

    @staticmethod
    def _sanitize_weight(w: Any) -> float:
        """
        Weight'i güvenli float'a çevirir:
        - NaN/inf/negatif -> 0.0
        """
        try:
            wf = float(w)
        except Exception:
            return 0.0

        if not np.isfinite(wf) or wf < 0.0:
            return 0.0

        return wf

    def predict_mtf(
                self,
        X_by_interval: Dict[str, Any],
        weight_by_interval: Dict[str, float] | None = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Dönüş:
            (ensemble_p, mtf_debug)
        """
        per_interval: Dict[str, Any] = {}
        probs: list[float] = []
        weights: list[float] = []
        intervals_used: list[str] = []

        weight_by_interval = weight_by_interval or {}

        for itv, model in self.models_by_interval.items():
            X_itv = X_by_interval.get(itv)

            # None / empty kontrolü (DataFrame için)
            if X_itv is None:
                continue
            if isinstance(X_itv, pd.DataFrame) and X_itv.empty:
                continue

            try:
                proba_arr, dbg = model.predict_proba(X_itv)
                p_last = self._extract_last_prob(proba_arr)

                weight_raw = weight_by_interval.get(itv, 1.0)
                weight = self._sanitize_weight(weight_raw)

                per_interval[itv] = {
                    "p_last": float(p_last),
                    "weight": float(weight),
                    "debug": dbg,
                }

                probs.append(float(p_last))
                weights.append(float(weight))
                intervals_used.append(itv)

            except Exception as e:
                self.logger.warning(
                    "[HYBRID-MTF] Interval=%s için predict_proba hata: %s",
                    itv,
                    e,
                )
                continue

        if not probs:
            self.logger.warning(
                "[HYBRID-MTF] Hiç interval için geçerli prob üretilemedi, ensemble fallback."
            )
            return 0.5, {
                "per_interval": per_interval,
                "intervals_used": [],
                "weights_raw": [],
                "weights_norm": [],
                "n_used": 0,
                "ensemble_p": 0.5,
            }

        probs_arr = np.asarray(probs, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)

        weights_sum = float(weights_arr.sum())
        if weights_sum <= 1e-12:
            weights_norm = np.ones_like(weights_arr) / len(weights_arr)
        else:
            weights_norm = weights_arr / weights_sum

        ensemble_p = float((probs_arr * weights_norm).sum())

        mtf_debug = {
            "per_interval": per_interval,
            "intervals_used": intervals_used,
            "weights_raw": weights,
            "weights_sum_raw": weights_sum,
            "weights_norm": weights_norm.tolist(),
            "n_used": len(intervals_used),
            "ensemble_p": ensemble_p,
        }

        return ensemble_p, mtf_debug


class HybridMTF:
    """
    Stabil MTF ağırlıklandırma (tek kaynak):

    Tek eşik: auc_floor
    - auc <= auc_floor  -> weight=0.0 (skip)
    - auc_floor..auc_max_used lineer map:
        weight = (auc - auc_floor) / (auc_max_used - auc_floor)
    - auc >= auc_max_used -> weight=1.0

    Otomatik kalibrasyon (opsiyonel):
    - auc_max_used sabit yerine, son N gün AUC dağılımından percentile alınır
    - auc_max_used bounds içinde kırpılır (örn. 0.56..0.70)
    - AUC geçmişi yoksa fallback: self.auc_max

    AUC standardizasyon:
    - meta içinde hangi AUC alanı kullanılıyorsa auc_key_priority ile seçilir
    - opsiyonel olarak seçilen AUC meta[standardize_auc_key] alanına yazılabilir
    """

    def __init__(
        self,
        models_by_interval: Dict[str, Any] | None = None,
        auc_floor: float = 0.50,
        auc_max: float = 0.60,
        weight_floor: float = 1e-6,
        logger: logging.Logger | None = None,
        # --- auto calibration ---
        auto_calibrate_auc_max: bool = True,
        calib_days: int = 14,
        auc_max_percentile: float = 0.80,
        auc_max_bounds: Tuple[float, float] = (0.56, 0.70),
        min_history_points: int = 30,
        # --- AUC key standardization ---
        auc_key_priority: Tuple[str, ...] = ("auc_used", "wf_auc_mean", "val_auc", "best_auc", "auc", "oof_auc"),
    ) -> None:
        self.models_by_interval = models_by_interval or {}

        self.auc_floor = float(auc_floor)
        self.auc_max = float(auc_max)
        self.weight_floor = float(weight_floor)

        self.logger = logger or system_logger

        self.auto_calibrate_auc_max = bool(auto_calibrate_auc_max)
        self.calib_days = int(calib_days)
        self.auc_max_percentile = float(auc_max_percentile)
        self.auc_max_bounds = (float(auc_max_bounds[0]), float(auc_max_bounds[1]))
        self.min_history_points = int(min_history_points)

        self.auc_key_priority = tuple(str(x) for x in auc_key_priority) if auc_key_priority else ("best_auc",)

        # Log spam kontrolü (interval bazlı 1 kez)
        self._warned_skip_auc: set[str] = set()
        self._warned_low_w: set[str] = set()
        self._warned_calib_missing: set[str] = set()

        self._ensemble = MultiTimeframeHybridEnsemble(
            models_by_interval=self.models_by_interval,
            logger_=self.logger,
        )

    # --------------------------------------------------------------
    # AUC seçim + standardizasyon
    # --------------------------------------------------------------
    def _pick_auc_from_meta(self, meta: Dict[str, Any]) -> Tuple[float, str]:
        """
        meta içinde AUC alanını öncelik sırasına göre seçer.

        Returns:
            (auc_value, auc_key_used)

        Not:
            - Uygun/parse edilemeyen değerler atlanır.
            - Hiçbiri yoksa 0.5 döner ve key="fallback".
        """
        if not meta:
            return 0.5, "fallback"

        for key in self.auc_key_priority:
            if key not in meta:
                continue

            raw = meta.get(key, None)
            if raw is None:
                continue

            try:
                v = float(raw)
            except Exception:
                continue

            if not np.isfinite(v):
                continue

            return v, key

        return 0.5, "fallback"

    def _maybe_standardize_auc(
        self,
        model: Any,
        auc_value: float,
        target_key: str | None,
        overwrite: bool,
    ) -> None:
        """
        Seçilen AUC'yi model.meta[target_key] alanına yazmak için yardımcı.
        """
        if not target_key:
            return

        meta = getattr(model, "meta", None)
        if meta is None or not isinstance(meta, dict):
            meta = {}
            setattr(model, "meta", meta)

        if (target_key in meta) and (not overwrite):
            return

        meta[target_key] = float(auc_value)

    # -----------------------------
    # AUC history extraction helpers
    # -----------------------------
    @staticmethod
    def _coerce_auc_series(items: Any) -> pd.Series:
        """
        AUC geçmişini esnek formatlardan pandas Series'e çevirir.
        """
        if items is None:
            return pd.Series(dtype=float)

        if isinstance(items, pd.Series):
            s = items.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                try:
                    s.index = pd.to_datetime(s.index, errors="coerce", utc=True)
                except Exception:
                    pass
            s = pd.to_numeric(s, errors="coerce")
            return s.dropna()

        if isinstance(items, dict):
            try:
                idx = pd.to_datetime(list(items.keys()), errors="coerce", utc=True)
                vals = pd.to_numeric(list(items.values()), errors="coerce")
                s = pd.Series(vals, index=idx).dropna()
                return s[s.index.notna()]
            except Exception:
                return pd.Series(dtype=float)

        if isinstance(items, (list, tuple, np.ndarray)):
            if len(items) == 0:
                return pd.Series(dtype=float)

            first = items[0]
            if isinstance(first, dict):
                rows: List[Tuple[pd.Timestamp | None, float]] = []
                for d in items:
                    if not isinstance(d, dict):
                        continue
                    auc_val = d.get("auc", d.get("value", d.get("v", None)))
                    ts_val = d.get("ts", d.get("t", d.get("time", d.get("date", None))))
                    try:
                        auc_f = float(auc_val)
                    except Exception:
                        continue
                    ts = pd.to_datetime(ts_val, errors="coerce", utc=True) if ts_val is not None else None
                    rows.append((ts if pd.notna(ts) else None, auc_f))

                if not rows:
                    return pd.Series(dtype=float)

                ts_list = [r[0] for r in rows]
                auc_list = [r[1] for r in rows]

                if any(t is not None for t in ts_list):
                    idx = pd.to_datetime([t for t in ts_list], errors="coerce", utc=True)
                    s = pd.Series(pd.to_numeric(auc_list, errors="coerce"), index=idx).dropna()
                    return s[s.index.notna()]

                s = pd.Series(pd.to_numeric(auc_list, errors="coerce")).dropna()
                return s

            try:
                s = pd.Series(pd.to_numeric(items, errors="coerce")).dropna()
                return s
            except Exception:
                return pd.Series(dtype=float)

        try:
            return pd.Series([float(items)])
        except Exception:
            return pd.Series(dtype=float)

    def _extract_recent_auc_history(self, meta: Dict[str, Any]) -> pd.Series:
        """
        model.meta içinden AUC geçmişini çıkarır ve son calib_days gün ile sınırlar.
        """
        if not meta:
            return pd.Series(dtype=float)

        history = meta.get("auc_history", meta.get("auc_hist", meta.get("auc_series", None)))
        s = self._coerce_auc_series(history)
        if s.empty:
            return s

        if isinstance(s.index, pd.DatetimeIndex) and s.index.notna().any():
            now = pd.Timestamp.utcnow()
            start = now - pd.Timedelta(days=max(self.calib_days, 1))
            s2 = s[(s.index >= start) & (s.index <= now)]
            return s2.dropna()

        return s.dropna()

    # ----------------------------------------
    # Auto-calibrated auc_max per interval
    # ----------------------------------------
    def _calibrate_auc_max(self, interval: str, meta: Dict[str, Any]) -> float:
        """
        Son N gün AUC dağılımından auc_max_used üretir.
        """
        if not self.auto_calibrate_auc_max:
            return float(self.auc_max)

        s = self._extract_recent_auc_history(meta)
        if s.empty or len(s) < self.min_history_points:
            if interval not in self._warned_calib_missing:
                self._warned_calib_missing.add(interval)
                self.logger.info(
                    "[HYBRID-MTF] Interval=%s için yeterli AUC history yok (n=%d). auc_max=%.3f kullanılacak.",
                    interval,
                    int(len(s)),
                    float(self.auc_max),
                )
            return float(self.auc_max)

        p = float(np.clip(self.auc_max_percentile, 0.50, 0.99))
        auc_p = float(np.nanpercentile(s.values.astype(float), p * 100.0))

        lo, hi = self.auc_max_bounds
        auc_max_used = float(np.clip(auc_p, lo, hi))

        if auc_max_used <= self.auc_floor + 1e-9:
            auc_max_used = float(max(self.auc_floor + 0.01, self.auc_max))

        return auc_max_used

    # --------------------------------------------------------------
    # AUC -> weight (lineer map)
    # --------------------------------------------------------------
    def _auc_to_weight_linear(self, auc: float, auc_max_used: float) -> float:
        auc = float(auc)
        auc_max_used = float(auc_max_used)

        if auc <= self.auc_floor:
            return 0.0

        if auc >= auc_max_used:
            return 1.0

        denom = auc_max_used - self.auc_floor
        if denom <= 0:
            return 1.0

        return (auc - self.auc_floor) / denom

    # --------------------------------------------------------------
    # Weight hesaplama: skip + stabil map + floor + auto-calib auc_max
    # --------------------------------------------------------------
    def _get_weight_for_interval(
        self,
        interval: str,
        auc_used: float,
        auc_max_used: float,
    ) -> float:
        auc_used = float(auc_used)

        if auc_used <= self.auc_floor:
            if interval not in self._warned_skip_auc:
                self._warned_skip_auc.add(interval)
                self.logger.warning(
                    "[HYBRID-MTF] Interval=%s weight=0 (auc<=%.2f), ensemble'da skip.",
                    interval,
                    self.auc_floor,
                )
            return 0.0

        weight = float(self._auc_to_weight_linear(auc_used, auc_max_used))

        if self.weight_floor > 0.0:
            weight = max(weight, self.weight_floor)

        if weight < 0.30 and interval not in self._warned_low_w:
            self._warned_low_w.add(interval)
            self.logger.info(
                "[HYBRID-MTF] Interval=%s düşük weight=%.6f (auc=%.4f, auc_max_used=%.4f) map+floor ile.",
                interval,
                weight,
                auc_used,
                float(auc_max_used),
            )

        return weight

    # --------------------------------------------------------------
    # Dış API: MTF tahmin
    # --------------------------------------------------------------
    def predict_mtf(
        self,
        X_by_interval: Dict[str, Any],
        standardize_auc_key: str | None = "auc_used",
        standardize_overwrite: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        self.logger.info(
            "[HYBRID-MTF] predict_mtf called | intervals=%s",
            list(self.models_by_interval.keys()),
        )

        weight_by_interval: Dict[str, float] = {}
        meta_by_interval: Dict[str, Any] = {}

        for itv, model in self.models_by_interval.items():
            meta = getattr(model, "meta", {}) or {}

            # AUC history yoksa ve wf_auc_mean varsa history bootstrap et
            if (not meta.get("auc_history")) and (meta.get("wf_auc_mean") is not None):
                try:
                    self.update_auc_history(interval=itv, auc_value=float(meta["wf_auc_mean"]))
                except Exception:
                    pass

            # Eğer auc_history yoksa ama wf_auc_mean varsa, otomatik history başlat
            try:
                if (not meta.get("auc_history")) and (meta.get("wf_auc_mean") is not None):
                    self.update_auc_history(interval=itv, auc_value=float(meta["wf_auc_mean"]))
            except Exception:
                pass

            auc_used, auc_key_used = self._pick_auc_from_meta(meta)

            self._maybe_standardize_auc(
                model=model,
                auc_value=auc_used,
                target_key=standardize_auc_key,
                overwrite=standardize_overwrite,
            )

            # Standardize yazdıysak, kalibrasyonda güncel meta'yı kullanmak için meta'yı tekrar alalım
            meta2 = getattr(model, "meta", {}) or {}

            auc_max_used = self._calibrate_auc_max(itv, meta2)

            weight = self._get_weight_for_interval(
                interval=itv,
                auc_used=auc_used,
                auc_max_used=auc_max_used,
            )

            weight_by_interval[itv] = float(weight)

            meta_by_interval[itv] = {
                "auc_used": float(auc_used),
                "auc_key_used": str(auc_key_used),
                "auc_max_used": float(auc_max_used),
                "weight": float(weight),
                "best_side": meta2.get("best_side", "long"),
                "has_auc_history": bool(meta2.get("auc_history") is not None),
            }

        ensemble_p, mtf_debug = self._ensemble.predict_mtf(
            X_by_interval=X_by_interval,
            weight_by_interval=weight_by_interval,
        )

        mtf_debug["meta_by_interval"] = meta_by_interval

        mtf_debug["auc_floor"] = self.auc_floor
        mtf_debug["auc_max_default"] = self.auc_max
        mtf_debug["weight_floor"] = self.weight_floor

        mtf_debug["auto_calibrate_auc_max"] = self.auto_calibrate_auc_max
        mtf_debug["calib_days"] = self.calib_days
        mtf_debug["auc_max_percentile"] = self.auc_max_percentile
        mtf_debug["auc_max_bounds"] = list(self.auc_max_bounds)
        mtf_debug["min_history_points"] = self.min_history_points

        mtf_debug["auc_key_priority"] = list(self.auc_key_priority)
        mtf_debug["standardize_auc_key"] = standardize_auc_key
        mtf_debug["standardize_overwrite"] = bool(standardize_overwrite)

        return ensemble_p, mtf_debug

    # ----------------------------------------
    # AUC history update (auto append)
    # ----------------------------------------
    def update_auc_history(
        self,
        interval: str,
        auc_value: float,
        ts: str | pd.Timestamp | None = None,
        max_points: int = 400,
        replace_same_day: bool = True,
    ) -> None:
        model = self.models_by_interval.get(interval)
        if model is None:
            return

        try:
            auc_f = float(auc_value)
        except Exception:
            return

        if not np.isfinite(auc_f):
            return

        meta = getattr(model, "meta", None)
        if meta is None or not isinstance(meta, dict):
            meta = {}
            setattr(model, "meta", meta)

        if ts is None:
            ts_dt = pd.Timestamp.utcnow()
        else:
            ts_dt = pd.to_datetime(ts, errors="coerce", utc=True)
            if ts_dt is pd.NaT:
                ts_dt = pd.Timestamp.utcnow()

        rec = {"ts": ts_dt.isoformat(), "auc": auc_f}

        hist = meta.get("auc_history", [])
        if hist is None:
            hist = []

        if isinstance(hist, dict):
            try:
                hist = [{"ts": str(k), "auc": float(v)} for k, v in hist.items()]
            except Exception:
                hist = []

        if not isinstance(hist, list):
            hist = []

        if replace_same_day and hist:
            day = ts_dt.date()
            replaced = False
            for i in range(len(hist) - 1, -1, -1):
                h = hist[i]
                if not isinstance(h, dict):
                    continue
                hts = pd.to_datetime(h.get("ts"), errors="coerce", utc=True)
                if hts is pd.NaT:
                    continue
                if hts.date() == day:
                    hist[i] = rec
                    replaced = True
                    break
            if not replaced:
                hist.append(rec)
        else:
            hist.append(rec)

        try:
            hist.sort(key=lambda x: pd.to_datetime(x.get("ts"), errors="coerce", utc=True))
        except Exception:
            pass

        if max_points > 0 and len(hist) > max_points:
            hist = hist[-max_points:]

        meta["auc_history"] = hist

    def update_auc_history_from_models_meta(
        self,
        ts: str | pd.Timestamp | None = None,
        max_points: int = 400,
        replace_same_day: bool = True,
        auc_key: str = "best_auc",
    ) -> None:
        for itv, model in self.models_by_interval.items():
            meta = getattr(model, "meta", {}) or {}
            auc_val = meta.get(auc_key, None)
            if auc_val is None:
                continue

            self.update_auc_history(
                interval=itv,
                auc_value=float(auc_val),
                ts=ts,
                max_points=max_points,
                replace_same_day=replace_same_day,
            )
