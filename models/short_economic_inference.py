from __future__ import annotations
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from features.short_economic import build_short_economic_features

class ShortEconomicModel:
    def __init__(self, model_path: str | Path, meta_path: str | Path):
        self.model_path=Path(model_path); self.meta_path=Path(meta_path)
        self.meta=json.loads(self.meta_path.read_text())
        if self.meta.get('score_semantics')!='short_economic_probability':
            raise ValueError('unsupported short economic score semantics')
        self.features=list(self.meta.get('features') or [])
        if not self.features: raise ValueError('missing economic feature contract')
        self.model=joblib.load(self.model_path)
    def predict_score(self, completed_klines: pd.DataFrame) -> float | None:
        if completed_klines is None or len(completed_klines)<241: return None
        f=build_short_economic_features(completed_klines)
        row=f.iloc[-1][self.features]
        x=row.to_numpy(dtype=float)
        if not np.isfinite(x).all(): return None
        p=float(self.model.predict_proba(pd.DataFrame([row.to_dict()],columns=self.features))[:,1][0])
        if not np.isfinite(p): return None
        return max(0.0,min(1.0,p))
