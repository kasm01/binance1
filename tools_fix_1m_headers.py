import os, re
import pandas as pd

DATA_DIR="data/offline_cache"
SYMBOLS=os.getenv("SYMBOLS","BTCUSDT").split(",")

raw_cols = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]
need = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]

def looks_like_no_header(cols):
    if "open_time" in cols:
        return False
    # ilk kolon ms gibi sayısal string ise header yok demektir
    return bool(re.fullmatch(r"\d{10,}", str(cols[0])))

def fix_one(sym: str):
    path=os.path.join(DATA_DIR, f"{sym}_1m_6m.csv")
    if not os.path.exists(path):
        print("[SKIP] missing", path); return

    # önce sadece header kontrolü için az oku
    probe=pd.read_csv(path, nrows=2)
    if len(probe.columns)==12 and looks_like_no_header(probe.columns):
        df=pd.read_csv(path, header=None, names=raw_cols)
        print(f"[FIX] {sym}: header missing -> reloaded header=None")
    else:
        df=pd.read_csv(path)
        print(f"[OK ] {sym}: header ok")

    # taker kolonlarını normalize et
    if "taker_buy_base_asset_volume" in df.columns and "taker_buy_base_volume" not in df.columns:
        df=df.rename(columns={"taker_buy_base_asset_volume":"taker_buy_base_volume"})
    if "taker_buy_quote_asset_volume" in df.columns and "taker_buy_quote_volume" not in df.columns:
        df=df.rename(columns={"taker_buy_quote_asset_volume":"taker_buy_quote_volume"})

    # gerekli kolonları garanti et
    for c in need:
        if c not in df.columns:
            df[c]=0

    # open_time/close_time numeric ms'e çevir (ISO ise de numeric değilse NaN olur)
    for c in ["open_time","close_time"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")

    # numeric cast
    num_cols=["open","high","low","close","volume","quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    for c in num_cols:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    df["number_of_trades"]=pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")

    # drop bad rows
    df=df.dropna(subset=["open_time","open","high","low","close"])
    df=df[need]

    df.to_csv(path, index=False)
    print(f"[WROTE] {sym}: {path} rows={len(df)} open_time_unique={df['open_time'].nunique()} min={df['open_time'].min()} max={df['open_time'].max()}")

for sym in [s.strip() for s in SYMBOLS if s.strip()]:
    fix_one(sym)
