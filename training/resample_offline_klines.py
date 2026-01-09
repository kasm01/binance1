#!/usr/bin/env python3
import os
import pandas as pd

# ENV
SYMBOL   = os.getenv("SYMBOL", "BTCUSDT").strip().upper()
DATA_DIR = os.getenv("DATA_DIR", "data/offline_cache")

SRC = os.path.join(DATA_DIR, f"{SYMBOL}_1m_6m.csv")
if not os.path.exists(SRC):
    raise SystemExit(f"HATA: Kaynak 1m CSV yok: {SRC}")

# Binance 12-col schema (pipeline uyumu)
COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]

def read_1m_csv(path: str) -> pd.DataFrame:
    """
    Hem header'sız (raw 12-col) hem header'lı CSV'yi destekler.
    Çıkış: COLS kolonları garanti.
    """
    # Önce header'lı dene (kolon isimleri var mı?)
    try:
        dfh = pd.read_csv(path, low_memory=False)
        if set(COLS).issubset(set(dfh.columns)):
            df = dfh.copy()
        else:
            # Header'lı ama isimler farklı olabilir -> header'sız fallback
            raise ValueError("no expected headers")
    except Exception:
        # Header'sız raw
        df = pd.read_csv(path, header=None, names=COLS, low_memory=False)

    # Olası alternatif kolon adları -> normalize
    rename_map = {}
    if "taker_buy_base_asset_volume" in df.columns and "taker_buy_base_volume" not in df.columns:
        rename_map["taker_buy_base_asset_volume"] = "taker_buy_base_volume"
    if "taker_buy_quote_asset_volume" in df.columns and "taker_buy_quote_volume" not in df.columns:
        rename_map["taker_buy_quote_asset_volume"] = "taker_buy_quote_volume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Eksik varsa 0 ile doldur
    for c in COLS:
        if c not in df.columns:
            df[c] = 0

    return df[COLS].copy()

def to_dt_utc(series: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(series, errors="coerce")
    if s_num.notna().mean() > 0.8:
        # ms
        return pd.to_datetime(s_num, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")

df = read_1m_csv(SRC)

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

# NaN OHLC temizle
df = df.dropna(subset=["open","high","low","close"])

AGG = {
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

    # DatetimeIndex -> ms int
    open_ms = (r.index.asi8 // 1_000_000).astype("int64")  # ns -> ms
    # close_time = open + freq - 1ms
    delta_ms = int(pd.Timedelta(freq).total_seconds() * 1000)
    close_ms = (open_ms + (delta_ms - 1)).astype("int64")

    r["open_time"] = open_ms
    r["close_time"] = close_ms

    r = r.reset_index(drop=True)

    # kolon sırası 12’ye sabit
    for c in COLS:
        if c not in r.columns:
            r[c] = 0
    r = r[COLS]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ✅ header'sız yaz (pipeline header=None uyumu)
    r.to_csv(out_path, index=False, header=False)
    print(f"[OK] {freq} -> {out_path} | shape={r.shape}")

def resample_to(freq: str, out_path: str):
    r = df.resample(freq, label="left", closed="left").agg(AGG)
    finalize_and_save(r, freq=freq, out_path=out_path)

# İstersen ENV ile seçilebilir yap
# Varsayılan: 3m ve 30m
out_3m  = os.path.join(DATA_DIR, f"{SYMBOL}_3m_6m.csv")
out_30m = os.path.join(DATA_DIR, f"{SYMBOL}_30m_6m.csv")

resample_to("3min", out_3m)
resample_to("30min", out_30m)
