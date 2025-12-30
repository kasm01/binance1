#!/usr/bin/env python
import os
import pandas as pd

SYMBOL   = os.getenv("SYMBOL", "BTCUSDT")
DATA_DIR = os.getenv("DATA_DIR", "data/offline_cache")

src = os.path.join(DATA_DIR, f"{SYMBOL}_1m_6m.csv")
if not os.path.exists(src):
    raise SystemExit(f"HATA: Kaynak 1m CSV yok: {src}")

df = pd.read_csv(src)

# Kolon adı normalizasyonu (LIVE schema ile uyum)
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

# Eksikler varsa 0 ile ekle (resample kırılmasın)
for c in need:
    if c not in df.columns:
        df[c] = 0

# open_time / close_time: hem ms(int) hem ISO string destekle
def to_dt_utc(series: pd.Series) -> pd.Series:
    s = series
    # numeric(ms) gibi görünüyorsa
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() > 0.8:
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    # ISO string (veya farklı formatlar)
    return pd.to_datetime(s, utc=True, errors="coerce")

dt_open = to_dt_utc(df["open_time"])
dt_close = to_dt_utc(df["close_time"])

df = df.loc[dt_open.notna()].copy()
df.index = dt_open[dt_open.notna()]

# Sayısal dönüşümler
num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")

# NaN OHLC varsa resample zaten düşecek ama öncesinde temizleyelim
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

def finalize_and_save(r: pd.DataFrame, out_path: str):
    # open_time / close_time'ı ms int olarak yaz (pipeline uyumlu)
    r = r.dropna(subset=["open","high","low","close"]).copy()
    r["open_time"] = (r.index.view("int64") // 10**6).astype("int64")
    # close_time: bar bitişi (index + freq - 1ms) gibi davranalım
    # (basit: bir sonraki open - 1ms)
    # freq bilgisi yoksa 0 yazma; en azından open_time var.
    r["close_time"] = r["open_time"]
    r = r.reset_index(drop=True)

    # Kolon sırasını 12'ye sabitle
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
    r.to_csv(out_path, index=False)
    print(f"[OK] -> {out_path} | shape={r.shape}")

def resample_to(freq: str, out_path: str):
    r = df.resample(freq, label="left", closed="left").agg(agg)
    finalize_and_save(r, out_path)

out_3m  = os.path.join(DATA_DIR, f"{SYMBOL}_3m_6m.csv")
out_30m = os.path.join(DATA_DIR, f"{SYMBOL}_30m_6m.csv")

resample_to("3min", out_3m)
resample_to("30min", out_30m)
