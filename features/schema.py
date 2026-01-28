# features/schema.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Callable

# Alias: bazı kaynaklarda kolon isimleri farklı gelebiliyor.
# Burada "biri varsa diğerini de üret" mantığı var.
ALIASES = {
    "taker_buy_base_asset_volume": "taker_buy_base_volume",
    "taker_buy_quote_asset_volume": "taker_buy_quote_volume",
    "taker_buy_base_volume": "taker_buy_base_asset_volume",
    "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
}

def normalize_to_schema(
    df: pd.DataFrame,
    schema: List[str],
    *,
    log_missing: Optional[Callable[[List[str]], None]] = None,
    fill_method: str = "ffill_bfill",   # "none" | "ffill" | "bfill" | "ffill_bfill"
    dtype: str = "float32",
) -> pd.DataFrame:
    """
    df -> schema sözleşmesi:
      - alias fix (varsa diğerini üret)
      - schema'da olup df'de olmayan kolonları 0.0 ile ekle (log opsiyonel)
      - schema'da olmayan kolonları drop et
      - kolon sırasını schema ile aynı yap
      - numeric coerce, inf/-inf -> nan, (opsiyonel) ffill/bfill, fillna(0)
      - dtype cast

    NOT:
      - Bu fonksiyon feature drift'i öldürmek için deterministik olmalı.
      - MTF tarafında normalize_to_schema(df).tail(500) gibi çağrılar güvenli çalışır.
    """
    if df is None:
        # En güvenlisi: boş DF dönmek yerine schema kolonları ile tek satır 0 üretmek
        out = pd.DataFrame({c: [0.0] for c in schema})
        return out.astype(dtype, copy=False)

    out = df.copy()

    # --- alias fix ---
    # "src varsa dst yoksa dst=src" (tek yön)
    for src, dst in ALIASES.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    # --- fill missing (schema'da var ama df'de yok) ---
    missing = [c for c in schema if c not in out.columns]
    if missing:
        if log_missing:
            try:
                log_missing(missing)
            except Exception:
                pass
        for c in missing:
            out[c] = 0.0

    # --- drop extras + order ---
    # schema dışındaki kolonlar otomatik drop (out[schema] zaten bunu yapar)
    out = out.loc[:, schema].copy()

    # --- numeric cleanup ---
    # hızlı to_numeric (kolon bazlı)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # inf/-inf
    out = out.replace([np.inf, -np.inf], np.nan)

    # opsiyonel forward/backfill (özellikle bazı rolling'lerde ilk barlarda NaN olabiliyor)
    fm = str(fill_method or "").lower().strip()
    if fm == "ffill":
        out = out.ffill()
    elif fm == "bfill":
        out = out.bfill()
    elif fm == "ffill_bfill":
        out = out.ffill().bfill()
    else:
        # "none" vb: hiçbir şey yapma
        pass

    # kalan NaN -> 0
    out = out.fillna(0.0)

    # dtype
    try:
        out = out.astype(dtype, copy=False)
    except Exception:
        # dtype cast patlarsa float'a dön
        out = out.astype("float32", copy=False)

    return out
