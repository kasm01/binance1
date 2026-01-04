import json
import os
import pandas as pd
from typing import List

from app_paths import MODELS_DIR


def seed_auc_history_if_missing(intervals: List[str], logger=None) -> None:
    """
    Meta dosyasında auc_history yoksa:
    - auc_used varsa onu
    - yoksa best_auc varsa onu
    - yoksa 0.5
    ile tek kayıt seed eder.

    Bu işlem idempotenttir (bir kez çalışır).
    """
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
            continue  # already seeded

        auc_val = meta.get("auc_used", meta.get("best_auc", 0.5))
        try:
            auc_val = float(auc_val)
        except Exception:
            auc_val = 0.5

        rec = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "auc": float(auc_val),
        }

        meta["auc_history"] = [rec]
        meta["auc_history_source"] = "boot_seed_from_auc_used_or_best_auc"

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            if logger:
                logger.info(
                    "[AUC-HIST] Seeded auc_history interval=%s auc=%.6f",
                    itv,
                    float(auc_val),
                )
        except Exception as e:
            if logger:
                logger.warning(
                    "[AUC-HIST] Failed to write meta interval=%s: %s",
                    itv,
                    e,
                )
