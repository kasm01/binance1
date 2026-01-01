# features/schema.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional

ALIASES = {
    "taker_buy_base_asset_volume": "taker_buy_base_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
    "taker_buy_base_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}

def normalize_to_schema(df: pd.DataFrame, schema: List[str], *, log_missing: Optional[callable]=None) -> pd.DataFrame:
    out = df.copy()

    # alias
    for src, dst in ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    # fill missing
    missing = [c for c in schema if c not in out.columns]
    if missing:
        if log_missing:
            log_missing(missing)
        for c in missing:
            out[c] = 0.0

    # select/order
    out = out[schema].copy()

    # numeric cleanup
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    return out
