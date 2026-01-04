# utils/auc_history.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from app_paths import MODELS_DIR


def seed_auc_history_if_missing(intervals: List[str], logger=None) -> None:
    for itv in intervals:
        meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            continue

        hist = meta.get("auc_history")
        if isinstance(hist, list) and len(hist) > 0:
            continue

        auc_val = meta.get("auc_used", meta.get("best_auc", 0.5))
        try:
            auc_val = float(auc_val)
        except Exception:
            auc_val = 0.5

        rec = {"ts": pd.Timestamp.utcnow().isoformat(), "auc": float(auc_val)}
        meta["auc_history"] = [rec]
        meta["auc_history_source"] = "boot_seed_from_auc_used_or_best_auc"

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            if logger:
                logger.info("[AUC-HIST] Seeded auc_history interval=%s auc=%.6f", itv, float(auc_val))
        except Exception as e:
            if logger:
                logger.warning("[AUC-HIST] Failed to write meta interval=%s: %s", itv, e)


def append_auc_used_once_per_day(
    intervals: List[str],
    logger=None,
    today_utc: Optional[pd.Timestamp] = None,
    max_points: int = 400,
) -> None:
    """
    Her interval için günde 1 kez:
      auc_used (yoksa best_auc) -> auc_history ekler.

    Kalıcılık:
      - model_meta_{itv}.json içine yazar
      - meta["auc_history_last_day"] = "YYYY-MM-DD" ile günlük kilit
    """
    now = today_utc if today_utc is not None else pd.Timestamp.utcnow()
    day_str = now.strftime("%Y-%m-%d")

    for itv in intervals:
        meta_path = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            continue

        last_day = str(meta.get("auc_history_last_day", "") or "")
        if last_day == day_str:
            # bugün zaten append edilmiş
            continue

        auc_val = meta.get("auc_used", meta.get("best_auc", 0.5))
        try:
            auc_val = float(auc_val)
        except Exception:
            auc_val = 0.5

        rec = {"ts": now.isoformat(), "auc": float(auc_val)}

        hist = meta.get("auc_history", [])
        if not isinstance(hist, list):
            hist = []
        hist.append(rec)

        # zaman sırası
        try:
            hist.sort(key=lambda x: pd.to_datetime((x or {}).get("ts"), errors="coerce", utc=True))
        except Exception:
            pass

        if max_points > 0 and len(hist) > max_points:
            hist = hist[-max_points:]

        meta["auc_history"] = hist
        meta["auc_history_last_day"] = day_str
        meta["auc_history_append_source"] = "daily_append_from_auc_used"

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            if logger:
                logger.info(
                    "[AUC-HIST] Daily append interval=%s day=%s auc_used=%.6f (hist_len=%d)",
                    itv,
                    day_str,
                    float(auc_val),
                    int(len(hist)),
                )
        except Exception as e:
            if logger:
                logger.warning("[AUC-HIST] Daily append write failed interval=%s: %s", itv, e)
