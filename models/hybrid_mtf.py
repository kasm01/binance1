from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

system_logger = logging.getLogger("system")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


class MultiTimeframeHybridEnsemble:
    """
    Çoklu timeframe HybridModel ensemble.

    model.predict_proba(X) -> (proba_arr, debug)
    """

    def __init__(self, models_by_interval: Dict[str, Any], logger_: logging.Logger | None = None) -> None:
        self.models_by_interval = models_by_interval or {}
        self.logger = logger_ or system_logger

    @staticmethod
    def _extract_last_prob(proba_arr: Any) -> float:
        arr = np.asarray(proba_arr)
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)
            arr = arr[:, -1]
        if arr.size == 0:
            return 0.5
        return float(arr[-1])

    @staticmethod
    def _sanitize_weight(w: Any) -> float:
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
        per_interval: Dict[str, Any] = {}
        probs: list[float] = []
        weights: list[float] = []
        intervals_used: list[str] = []

        weight_by_interval = weight_by_interval or {}

        for itv, model in self.models_by_interval.items():
            X_itv = X_by_interval.get(itv)
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
                self.logger.warning("[HYBRID-MTF] Interval=%s predict_proba hata: %s", itv, e)
                continue

        if not probs:
            self.logger.warning("[HYBRID-MTF] Hiç interval için geçerli prob üretilemedi, ensemble fallback.")
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
    Stabil MTF ağırlıklandırma (tek kaynak)

    AUC -> weight (linear map):
      - auc <= auc_floor  -> weight=0.0 (skip)
      - auc_floor..auc_max_used -> lineer map
      - auc >= auc_max_used -> weight=1.0

    Auto-calibrate auc_max_used:
      - auc_history içinden percentile ile
      - history yoksa auc_max default

    AUC history persistence:
      - models/auc_history_{interval}.jsonl
      - init'te load edip model.meta["auc_history"] içine koyar

    IMPORTANT:
      - history yazımı "hour bucket" (1 saatte 1 kayıt) olacak şekilde yapılır (default: hour)
      - aynı bucket içinde yeni kayıt gelirse replace edilir
      - AUC değişimi küçükse (|Δ| < AUC_HISTORY_MIN_DELTA) yazılmaz
      - disk yazımı append değil, rewrite (duplicate bucket birikmesin)
    """

    def __init__(
        self,
        models_by_interval: Dict[str, Any] | None = None,
        auc_floor: float = 0.50,
        auc_max: float = 0.60,
        weight_floor: float = 1e-6,
        logger: logging.Logger | None = None,
        auto_calibrate_auc_max: bool = True,
        calib_days: int = 14,
        auc_max_percentile: float = 0.80,
        auc_max_bounds: Tuple[float, float] = (0.56, 0.70),
        min_history_points: int = 30,
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

        # predict_mtf log level control
        self.predict_log_level = str(os.getenv("HYBRID_MTF_LOG_LEVEL", "INFO")).upper()

        # history persistence dir
        self.auc_history_dir = os.getenv("AUC_HISTORY_DIR", "models")

        # bucket policy: "hour" (default) or "day"
        self.auc_history_bucket = str(os.getenv("AUC_HISTORY_BUCKET", "hour")).strip().lower()
        if self.auc_history_bucket not in ("hour", "day"):
            self.auc_history_bucket = "hour"

        # min delta to write history (same bucket)
        self.auc_history_min_delta = float(_env_float("AUC_HISTORY_MIN_DELTA", 0.001))

        # OPTIONAL: predict sırasında history güncellemek istersen (default kapalı)
        self.update_history_on_predict = bool(_env_bool("AUC_HISTORY_UPDATE_ON_PREDICT", False))

        self._warned_skip_auc: set[str] = set()
        self._warned_low_w: set[str] = set()
        self._warned_calib_missing: set[str] = set()

        self._ensemble = MultiTimeframeHybridEnsemble(models_by_interval=self.models_by_interval, logger_=self.logger)

        # init: load persisted histories into model.meta (fix n=0 inconsistency)
        self._bootstrap_auc_histories_from_disk()

    # --------------------------------------------------------------
    # Disk persistence
    # --------------------------------------------------------------
    def _history_path(self, interval: str) -> str:
        return os.path.join(self.auc_history_dir, f"auc_history_{interval}.jsonl")

    def _read_history_jsonl(self, path: str, max_points: int = 400) -> List[Dict[str, Any]]:
        if not path or (not os.path.exists(path)):
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and ("auc" in rec) and ("ts" in rec):
                            out.append(rec)
                    except Exception:
                        continue
        except Exception:
            return []
        if max_points > 0 and len(out) > max_points:
            out = out[-max_points:]
        return out

    def _write_history_jsonl(self, path: str, rows: List[Dict[str, Any]]) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _bootstrap_auc_histories_from_disk(self) -> None:
        """
        Disk'ten auc_history_{itv}.jsonl yükle.
        Yoksa best_auc ile seed et (bucketed write).
        """
        for itv, model in self.models_by_interval.items():
            meta = getattr(model, "meta", None)
            if meta is None or not isinstance(meta, dict):
                meta = {}
                setattr(model, "meta", meta)

            # already has history -> keep
            if isinstance(meta.get("auc_history", None), list) and len(meta["auc_history"]) > 0:
                continue

            path = self._history_path(itv)
            hist = self._read_history_jsonl(path, max_points=400)

            if hist:
                meta["auc_history"] = hist
                self.logger.info(
                    "[AUC-HIST] Loaded auc_history from disk interval=%s n=%d path=%s",
                    itv,
                    len(hist),
                    path,
                )
                continue

            # seed from best_auc once
            try:
                best_auc = float(meta.get("best_auc", 0.0) or 0.0)
            except Exception:
                best_auc = 0.0

            if best_auc > 0.0:
                self.update_auc_history(
                    interval=itv,
                    auc_value=best_auc,
                    max_points=400,
                    replace_same_bucket=True,
                )
                self.logger.info("[AUC-HIST] Seeded auc_history interval=%s auc=%.6f", itv, best_auc)

    # --------------------------------------------------------------
    # AUC pick/standardize
    # --------------------------------------------------------------
    def _pick_auc_from_meta(self, meta: Dict[str, Any]) -> Tuple[float, str]:
        if not meta:
            return 0.5, "fallback"
        for key in self.auc_key_priority:
            raw = meta.get(key, None)
            if raw is None:
                continue
            try:
                v = float(raw)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            return float(v), str(key)
        return 0.5, "fallback"

    def _maybe_standardize_auc(self, model: Any, auc_value: float, target_key: str | None, overwrite: bool) -> None:
        if not target_key:
            return
        meta = getattr(model, "meta", None)
        if meta is None or not isinstance(meta, dict):
            meta = {}
            setattr(model, "meta", meta)
        if (target_key in meta) and (not overwrite):
            return
        meta[target_key] = float(auc_value)

    # --------------------------------------------------------------
    # AUC history extraction / calibration
    # --------------------------------------------------------------
    @staticmethod
    def _coerce_auc_series(items: Any) -> pd.Series:
        if items is None:
            return pd.Series(dtype=float)

        if isinstance(items, list):
            rows = []
            for d in items:
                if not isinstance(d, dict):
                    continue
                auc_val = d.get("auc", d.get("value", d.get("v", None)))
                ts_val = d.get("ts", d.get("t", d.get("time", d.get("date", None))))
                try:
                    auc_f = float(auc_val)
                except Exception:
                    continue
                ts = pd.to_datetime(ts_val, errors="coerce", utc=True) if ts_val is not None else pd.NaT
                rows.append((ts, auc_f))
            if not rows:
                return pd.Series(dtype=float)
            idx = pd.to_datetime([r[0] for r in rows], errors="coerce", utc=True)
            vals = pd.to_numeric([r[1] for r in rows], errors="coerce")
            s = pd.Series(vals, index=idx).dropna()
            return s[s.index.notna()]

        # fallback
        try:
            return pd.Series([float(items)])
        except Exception:
            return pd.Series(dtype=float)

    def _extract_recent_auc_history(self, meta: Dict[str, Any]) -> pd.Series:
        if not meta:
            return pd.Series(dtype=float)
        history = meta.get("auc_history", None)
        s = self._coerce_auc_series(history)
        if s.empty:
            return s

        if isinstance(s.index, pd.DatetimeIndex) and s.index.notna().any():
            now = pd.Timestamp.utcnow()
            start = now - pd.Timedelta(days=max(self.calib_days, 1))
            s2 = s[(s.index >= start) & (s.index <= now)]
            return s2.dropna()

        return s.dropna()

    def _calibrate_auc_max(self, interval: str, meta: Dict[str, Any]) -> float:
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
    # AUC -> weight
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

    def _get_weight_for_interval(self, interval: str, auc_used: float, auc_max_used: float) -> float:
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
                "[HYBRID-MTF] Interval=%s düşük weight=%.6f (auc=%.4f, auc_max_used=%.4f)",
                interval,
                weight,
                auc_used,
                float(auc_max_used),
            )

        return weight

    # --------------------------------------------------------------
    # Dış API: MTF tahmin (TEK)
    # --------------------------------------------------------------
    def predict_mtf(
        self,
        X_by_interval: Dict[str, Any],
        standardize_auc_key: str | None = "auc_used",
        standardize_overwrite: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        lvl = getattr(logging, self.predict_log_level, logging.INFO)
        self.logger.log(
            lvl,
            "[HYBRID-MTF] predict_mtf called | intervals=%s",
            list(self.models_by_interval.keys()),
        )

        weight_by_interval: Dict[str, float] = {}
        meta_by_interval: Dict[str, Any] = {}

        for itv, model in self.models_by_interval.items():
            meta = getattr(model, "meta", {}) or {}

            # AUC pick
            auc_used, auc_key_used = self._pick_auc_from_meta(meta)

            # standardize (runtime)
            self._maybe_standardize_auc(model, auc_used, standardize_auc_key, standardize_overwrite)

            # ensure history loaded (if someone created new model instance)
            if not (isinstance(meta.get("auc_history", None), list) and len(meta.get("auc_history", [])) > 0):
                hist = self._read_history_jsonl(self._history_path(itv), max_points=400)
                if hist:
                    meta["auc_history"] = hist

            # OPTIONAL: predict sırasında history update (normalde retrain sonrası yazmak daha iyi)
            if self.update_history_on_predict and (auc_key_used != "fallback"):
                try:
                    self.update_auc_history(
                        interval=itv,
                        auc_value=float(auc_used),
                        ts=None,
                        max_points=400,
                        replace_same_bucket=True,
                    )
                except Exception:
                    pass

            auc_max_used = self._calibrate_auc_max(itv, meta)
            weight = self._get_weight_for_interval(itv, auc_used, auc_max_used)

            weight_by_interval[itv] = float(weight)
            meta_by_interval[itv] = {
                "auc_used": float(auc_used),
                "auc_key_used": str(auc_key_used),
                "auc_max_used": float(auc_max_used),
                "weight": float(weight),
                "best_side": meta.get("best_side", "long"),
                "auc_hist_n": int(len(meta.get("auc_history", []) or [])),
                "auc_history_bucket": self.auc_history_bucket,
                "auc_history_min_delta": float(self.auc_history_min_delta),
            }

        ensemble_p, mtf_debug = self._ensemble.predict_mtf(
            X_by_interval=X_by_interval,
            weight_by_interval=weight_by_interval,
        )

        # Inject AUC info into per_interval
        per_interval = mtf_debug.get("per_interval")
        if isinstance(per_interval, dict):
            for itv, info in per_interval.items():
                if not isinstance(info, dict):
                    continue
                mi = meta_by_interval.get(itv) or {}
                info["auc_used"] = mi.get("auc_used")
                info["auc_key_used"] = mi.get("auc_key_used")
                info["auc_max_used"] = mi.get("auc_max_used")
                info["auc_hist_n"] = mi.get("auc_hist_n")

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
    # AUC history update (bucketed: hour/day)
    # ----------------------------------------
    def _bucket_ts(self, ts_dt: pd.Timestamp) -> pd.Timestamp:
        # UTC-aware
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.tz_localize("UTC")
        else:
            ts_dt = ts_dt.tz_convert("UTC")

        if self.auc_history_bucket == "day":
            return ts_dt.floor("D")
        # default hour
        return ts_dt.floor("H")

    def update_auc_history(
        self,
        interval: str,
        auc_value: float,
        ts: str | pd.Timestamp | None = None,
        max_points: int = 400,
        replace_same_bucket: bool = True,
    ) -> None:
        """
        Saatlik (veya günlük) bucket ile history günceller.
        - aynı bucket varsa: min-delta küçükse yazmaz; değilse replace eder
        - disk'e rewrite eder (bucket duplicate olmasın)
        """
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

        # timestamp
        if ts is None:
            ts_dt = pd.Timestamp.utcnow().tz_localize("UTC")
        else:
            ts_dt = pd.to_datetime(ts, errors="coerce", utc=True)
            if ts_dt is pd.NaT:
                ts_dt = pd.Timestamp.utcnow().tz_localize("UTC")

        bucket = self._bucket_ts(ts_dt)
        rec = {"ts": bucket.isoformat(), "auc": float(auc_f), "bucket": self.auc_history_bucket}

        hist = meta.get("auc_history", [])
        if hist is None:
            hist = []
        if not isinstance(hist, list):
            hist = []

        # find existing record in same bucket
        idx_same = None
        old_auc = None
        for i in range(len(hist) - 1, -1, -1):
            h = hist[i]
            if not isinstance(h, dict):
                continue
            hts = pd.to_datetime(h.get("ts"), errors="coerce", utc=True)
            if hts is pd.NaT:
                continue
            if self._bucket_ts(hts) == bucket:
                idx_same = i
                try:
                    old_auc = float(h.get("auc"))
                except Exception:
                    old_auc = None
                break

        # min-delta gate (aynı bucket içinde küçük farksa hiç yazma)
        if idx_same is not None and old_auc is not None:
            if abs(float(auc_f) - float(old_auc)) < float(self.auc_history_min_delta):
                return

        # apply update
        if idx_same is not None and replace_same_bucket:
            hist[idx_same] = rec
        else:
            hist.append(rec)

        # sort & trim
        try:
            hist.sort(key=lambda x: pd.to_datetime(x.get("ts"), errors="coerce", utc=True))
        except Exception:
            pass

        if max_points > 0 and len(hist) > max_points:
            hist = hist[-max_points:]

        meta["auc_history"] = hist

        # persist to disk (rewrite to avoid duplicates within bucket)
        path = self._history_path(interval)
        self._write_history_jsonl(path, hist)
