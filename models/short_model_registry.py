from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import os

from app_paths import MODELS_DIR
from features.short_economic import SHORT_ECON_MODEL_FEATURES_V1, SHORT_ECON_MODEL_FEATURE_HASH_V1
from models.short_economic_inference import ShortEconomicModel


class ShortModelRegistry:
    """Coin-specific SHORT economic model registry.

    Production defaults to MODELS_DIR and therefore never falls back to research artifacts.
    Research/shadow callers must explicitly pass a research model_dir.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir or MODELS_DIR)
        self._cache: Dict[Tuple[str, str], ShortEconomicModel] = {}


    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except Exception:
            return float(default)

    def _validate_execution_contract(self, meta: dict) -> None:
        # Entry-quality V2 predicts near-term entry quality, not lifecycle outcome.
        # Its model contract is intentionally independent from TP/SL/leverage.
        if str(meta.get("target") or "").startswith("short_entry_quality_"):
            return

        expected = {
            "leverage": self._env_float("DEFAULT_ORDER_LEVERAGE", 3.0),
            "tp_arm_roi": self._env_float("FIXED_ROI_TP_PCT", 0.0120),
            "giveback_roi": self._env_float("FIXED_ROI_TP_GIVEBACK", 0.0024),
            "hard_stop_roi": self._env_float("ROI_HARD_STOP_PCT", -0.0900),
        }
        for key, wanted in expected.items():
            if key not in meta:
                raise ValueError(f"SHORT model execution contract missing: {key}")
            got = float(meta.get(key))
            if abs(got - float(wanted)) > 1e-12:
                raise ValueError(
                    f"SHORT model execution contract mismatch: {key} meta={got} runtime={wanted}"
                )
        if str(os.getenv("SHORT_DYNAMIC_STOP_ENABLE", "1")).strip().lower() in ("1", "true", "yes", "on"):
            if not bool(meta.get("dynamic_stop_enable", False)):
                raise ValueError("SHORT model execution contract mismatch: dynamic_stop_enable")
            confirm = self._env_float("SHORT_DYNAMIC_STOP_CONFIRM_SEC", 60.0)
            catastrophic = self._env_float("SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT", 0.045)
            if abs(float(meta.get("dynamic_stop_confirm_sec", -1.0)) - confirm) > 1e-12:
                raise ValueError("SHORT model execution contract mismatch: dynamic_stop_confirm_sec")
            if abs(float(meta.get("dynamic_stop_catastrophic_pct", -1.0)) - catastrophic) > 1e-12:
                raise ValueError("SHORT model execution contract mismatch: dynamic_stop_catastrophic_pct")

    def paths_for(self, symbol: str, interval: str) -> tuple[Path, Path]:
        sym = str(symbol or '').upper().strip()
        itv = str(interval or '').strip()
        if not sym or not itv:
            raise ValueError('symbol and interval are required')
        return (
            self.model_dir / f'short_model_{sym}_{itv}_best.joblib',
            self.model_dir / f'short_model_meta_{sym}_{itv}.json',
        )

    def get(
        self,
        symbol: str,
        interval: str,
        require_promoted: bool = True,
        require_execution_contract: bool = True,
    ) -> ShortEconomicModel:
        sym = str(symbol or '').upper().strip()
        itv = str(interval or '').strip()
        key = (sym, itv)
        if key in self._cache:
            m = self._cache[key]
        else:
            model_path, meta_path = self.paths_for(sym, itv)
            if not model_path.exists() or not meta_path.exists():
                raise FileNotFoundError(f'SHORT model artifact missing for {sym} {itv}')
            m = ShortEconomicModel(model_path, meta_path)
            meta = m.meta or {}
            if str(meta.get('symbol') or '').upper() != sym:
                raise ValueError(f'SHORT model symbol mismatch: expected={sym} meta={meta.get("symbol")}')
            if str(meta.get('side') or '').lower() != 'short':
                raise ValueError('SHORT model side mismatch')
            if str(meta.get('interval') or '') != itv:
                raise ValueError('SHORT model interval mismatch')
            if str(meta.get('score_semantics') or '') != 'short_economic_probability':
                raise ValueError('SHORT model score semantics mismatch')

            expected_features = list(SHORT_ECON_MODEL_FEATURES_V1)
            if list(meta.get('features') or []) != expected_features:
                raise ValueError('SHORT model feature contract mismatch')
            if str(meta.get('feature_hash') or '') != SHORT_ECON_MODEL_FEATURE_HASH_V1:
                raise ValueError('SHORT model feature hash mismatch')
            if int(meta.get('feature_count') or 0) != len(expected_features):
                raise ValueError('SHORT model feature count mismatch')
            self._cache[key] = m
        if require_execution_contract:
            self._validate_execution_contract(dict(m.meta or {}))
        if require_promoted and not bool((m.meta or {}).get('promoted', False)):
            raise PermissionError(f'SHORT model is not promoted: {sym} {itv}')
        return m
