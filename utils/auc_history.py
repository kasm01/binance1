# utils/auc_history.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import pandas as pd

from app_paths import MODELS_DIR


def _safe_float(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _load_meta(meta_path: str) -> Dict[str, Any]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _write_meta(meta_path: str, meta: Dict[str, Any], logger=None) -> bool:
    try:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return True
    except Exception as e:
        if logger:
            logger.warning("[AUC-HIST] meta write failed path=%s err=%s", meta_path, e)
        return False


def _meta_path_for_interval(interval: str) -> str:
    return os.path.join(MODELS_DIR, f"model_meta_{interval}.json")


def seed_auc_history_if_missing(intervals: List[str], logger=None) -> None:
    """
    auc_history yoksa (veya boşsa) tek kayıtla seed eder.
    Seed değeri: auc_used -> best_auc -> 0.5
    """
    for itv in intervals:
        meta_path = _meta_path_for_interval(itv)
        if not os.path.exists(meta_path):
            continue

        meta = _load_meta(meta_path)

        hist = meta.get("auc_history")
        if isinstance(hist, list) and len(hist) > 0:
            continue

        auc_val = _safe_float(meta.get("auc_used", meta.get("best_auc", 0.5)), 0.5)
        ts = pd.Timestamp.utcnow()
        rec = {"ts": ts.isoformat(), "auc": float(auc_val)}

        meta["auc_history"] = [rec]
        meta["auc_history_source"] = "boot_seed_from_auc_used_or_best_auc"
        meta["auc_history_last_day"] = ts.strftime("%Y-%m-%d")

        ok = _write_meta(meta_path, meta, logger=logger)
        if ok and logger:
            logger.info("[AUC-HIST] Seeded auc_history interval=%s auc=%.6f", itv, float(auc_val))


def append_auc_used_once_per_day(intervals: List[str], logger=None) -> None:
    """
    Her interval için günde 1 kez:
      - meta['auc_used'] (yoksa best_auc yoksa 0.5) değerini auc_history'ye ekler.
      - Aynı gün tekrar çağrılsa bile ikinci kez yazmaz.
    """
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    now_ts = pd.Timestamp.utcnow()

    for itv in intervals:
        meta_path = _meta_path_for_interval(itv)
        if not os.path.exists(meta_path):
            continue

        meta = _load_meta(meta_path)

        last_day = meta.get("auc_history_last_day")
        if isinstance(last_day, str) and last_day.strip() == today:
            continue

        hist = meta.get("auc_history")
        if not isinstance(hist, list):
            hist = []

        auc_val = _safe_float(meta.get("auc_used", meta.get("best_auc", 0.5)), 0.5)
        rec = {"ts": now_ts.isoformat(), "auc": float(auc_val)}
        hist.append(rec)

        max_points = int(meta.get("auc_history_max_points", 400) or 400)
        if max_points > 0 and len(hist) > max_points:
            hist = hist[-max_points:]

        meta["auc_history"] = hist
        meta["auc_history_last_day"] = today
        meta["auc_history_daily_source"] = "append_auc_used_once_per_day::auc_used"

        ok = _write_meta(meta_path, meta, logger=logger)
        if ok and logger:
            logger.info("[AUC-HIST] Daily append interval=%s day=%s auc=%.6f", itv, today, float(auc_val))
