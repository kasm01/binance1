#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so `import app_paths` works when running from /training)
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import Any, Dict, List, Tuple

from app_paths import MODELS_DIR


DEFAULT_INTERVALS = ["1m", "5m", "15m", "30m", "1h"]


def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return list(default)
    parts = [p.strip() for p in str(v).split(",")]
    return [p for p in parts if p]


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def load_meta(interval: str) -> Tuple[Dict[str, Any], str, bool]:
    meta_path = os.path.join(MODELS_DIR, f"model_meta_{interval}.json")
    if not os.path.exists(meta_path):
        return {}, meta_path, False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}, meta_path, True
    except Exception:
        return {}, meta_path, True


def pick_auc(meta: Dict[str, Any]) -> Tuple[float, str]:
    """
    AUC seçimi: MTF standardı -> auc_used, yoksa geriye dönük anahtarlar.
    """
    priority = ["auc_used", "sgd_val_auc", "best_auc", "wf_auc_mean", "val_auc", "auc", "oof_auc"]
    for k in priority:
        if k in meta and meta.get(k) is not None:
            v = _safe_float(meta.get(k))
            if v == v:  # not NaN
                return v, k
    return 0.5, "fallback"


def main() -> None:
    intervals = _env_list("INTERVALS", DEFAULT_INTERVALS)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for itv in intervals:
        meta, meta_path, exists = load_meta(itv)
        if not exists:
            missing.append(itv)
            rows.append(
                {
                    "interval": itv,
                    "meta": "MISSING",
                    "auc_used": 0.5,
                    "auc_key": "fallback",
                    "sgd_alpha": float("nan"),
                    "sgd_val_auc": float("nan"),
                    "best_auc": float("nan"),
                    "n_features": -1,
                    "seq_len": -1,
                    "use_lstm_hybrid": None,
                    "lstm_long_auc": float("nan"),
                    "lstm_short_auc": float("nan"),
                }
            )
            continue

        auc_val, auc_key = pick_auc(meta)

        rows.append(
            {
                "interval": itv,
                "meta": os.path.basename(meta_path),
                "auc_used": _safe_float(meta.get("auc_used", auc_val)),
                "auc_key": str(meta.get("auc_used_source", auc_key)),
                "sgd_alpha": _safe_float(meta.get("sgd_alpha")),
                "sgd_val_auc": _safe_float(meta.get("sgd_val_auc")),
                "best_auc": _safe_float(meta.get("best_auc")),
                "n_features": _safe_int(meta.get("n_features", len(meta.get("feature_schema") or []))),
                "seq_len": _safe_int(meta.get("seq_len")),
                "use_lstm_hybrid": meta.get("use_lstm_hybrid"),
                "lstm_long_auc": _safe_float(meta.get("lstm_long_auc")),
                "lstm_short_auc": _safe_float(meta.get("lstm_short_auc")),
            }
        )

    # pretty print
    print("=" * 110)
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"INTERVALS : {intervals}")
    if missing:
        print(f"MISSING   : {missing}")
    print("-" * 110)

    header = (
        f"{'itv':<4}  {'auc_used':>8}  {'auc_key':<32}  {'sgd_alpha':>8}  {'sgd_val':>8}  {'best_auc':>8}  "
        f"{'n_feat':>6}  {'seq':>4}  {'lstm':>5}  {'lstm_long':>9}  {'lstm_short':>10}"
    )
    print(header)
    print("-" * 110)

    for r in rows:
        itv = str(r["interval"])
        auc_used = r["auc_used"]
        auc_key = str(r["auc_key"])[:32]
        sgd_alpha = r["sgd_alpha"]
        sgd_val = r["sgd_val_auc"]
        best_auc = r["best_auc"]
        n_feat = r["n_features"]
        seq = r["seq_len"]
        lstm = r["use_lstm_hybrid"]
        ll = r["lstm_long_auc"]
        ls = r["lstm_short_auc"]

        def fnum(x: Any) -> str:
            try:
                xf = float(x)
                if xf != xf:
                    return "  nan"
                return f"{xf:0.4f}"
            except Exception:
                return "  nan"

        def fnum6(x: Any) -> str:
            try:
                xf = float(x)
                if xf != xf:
                    return "   nan"
                return f"{xf:0.6f}"
            except Exception:
                return "   nan"

        lstm_str = "None" if lstm is None else ("True" if bool(lstm) else "False")

        print(
            f"{itv:<4}  {fnum(auc_used):>8}  {auc_key:<32}  {fnum6(sgd_alpha):>8}  {fnum(sgd_val):>8}  {fnum(best_auc):>8}  "
            f"{int(n_feat):>6}  {int(seq):>4}  {lstm_str:>5}  {fnum(ll):>9}  {fnum(ls):>10}"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()
