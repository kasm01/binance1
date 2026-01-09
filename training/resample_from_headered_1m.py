#!/usr/bin/env python3
import os
import pandas as pd

SYMBOL   = os.getenv("SYMBOL", "BTCUSDT").strip().upper()
DATA_DIR = os.getenv("DATA_DIR", "data/offline_cache")

src = os.path.join(DATA_DIR, f"{SYMBOL}_1m_6m.csv")
if not os.path.exists(src):
    raise SystemExit(f"HATA: Kaynak 1m CSV yok: {src}")

df = pd.read_csv(src, low_memory=False)

# Kolon adı normalizasyonu (pipeline uyumu)
rename_map = {}
if "taker_buy_base_asset_volume" in df.columns and "taker_buy_base_volume" not in df.columns:
    rename_map["taker_buy_base_asset_volume"] = "taker_buy_base_volume"
if "taker_buy_quote_asset_volume" in df.columns and "taker_buy_quote_volume" not in df.columns:
    rename_map["taker_buy_quote_asset_volume"] = "taker_buy_quote_volume"
if rename_map:
    df = df.rename(columns=rename_map)

need = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]
for c in need:
    if c not in df.columns:
        df[c] = 0

def to_dt_utc(series: pd.Series) -> pd.Series:
    # numeric(ms) veya ISO string destekle
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().mean() > 0.8:
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

dt_open = to_dt_utc(df["open_time"])
df = df.loc[dt_open.notna()].copy()
df.index = dt_open[dt_open.notna()]
df = df.sort_index()

# Sayısal dönüşümler
num_cols = [
    "open","high","low","close","volume",
    "quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume","ignore"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")

df = df.dropna(subset=["open","high","low","close"])

agg = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_asset_volume": "sum",
    "number_of_trades": "sum",
    "taker_buy_base_volume": "sum",
    "taker_buy_quote_volume": "sum",
    "ignore": "sum",
}

def finalize_and_save(r: pd.DataFrame, freq: str, out_path: str):
    r = r.dropna(subset=["open","high","low","close"]).copy()

    # open_time/close_time ISO UTC üret
    open_ts = r.index
    close_ts = open_ts + pd.Timedelta(freq) - pd.Timedelta(milliseconds=1)

    r["open_time"] = open_ts.astype("datetime64[ns, UTC]").astype(str)
    r["close_time"] = close_ts.astype("datetime64[ns, UTC]").astype(str)

    r = r.reset_index(drop=True)

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","number_of_trades",
        "taker_buy_base_volume","taker_buy_quote_volume","ignore"
    ]
    for c in cols:
        if c not in r.columns:
            r[c] = 0
    r = r[cols]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ✅ header’lı yaz (senin 1m dosyanla aynı stil)
    r.to_csv(out_path, index=False)
    print(f"[OK] {freq} -> {out_path} | shape={r.shape}")

def resample_to(freq: str, out_path: str):
    r = df.resample(freq, label="left", closed="left").agg(agg)
    finalize_and_save(r, freq=freq, out_path=out_path)

out_3m  = os.path.join(DATA_DIR, f"{SYMBOL}_3m_6m.csv")
out_30m = os.path.join(DATA_DIR, f"{SYMBOL}_30m_6m.csv")

resample_to("3min", out_3m)
resample_to("30min", out_30m)
