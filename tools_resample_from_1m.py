import os
import pandas as pd

DATA_DIR="data/offline_cache"
SYMBOLS=os.getenv("SYMBOLS","BTCUSDT").split(",")

need = [
  "open_time","open","high","low","close","volume",
  "close_time","quote_asset_volume","number_of_trades",
  "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]

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
freq_map = {"3m":"3min","5m":"5min","15m":"15min","30m":"30min","1h":"1h"}

def save_schema(df: pd.DataFrame, out_path: str):
    df=df.dropna(subset=["open","high","low","close"]).copy()
    df["open_time"]=(df.index.view("int64")//10**6).astype("int64")
    df["close_time"]=df["open_time"]
    df=df.reset_index(drop=True)[need]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

for sym in [s.strip() for s in SYMBOLS if s.strip()]:
    src=os.path.join(DATA_DIR, f"{sym}_1m_6m.csv")
    if not os.path.exists(src):
        print("[SKIP] missing", src); continue

    df=pd.read_csv(src)
    # open_time ms -> datetime index
    ot=pd.to_numeric(df["open_time"], errors="coerce")
    dt=pd.to_datetime(ot, unit="ms", utc=True, errors="coerce")
    df=df.loc[dt.notna()].copy()
    df.index=dt[dt.notna()]
    df=df[need]

    print(f"[SRC] {sym} 1m rows={len(df)} open_time_unique={df['open_time'].nunique()}")

    for itv,freq in freq_map.items():
        out=os.path.join(DATA_DIR, f"{sym}_{itv}_6m.csv")
        r=df.resample(freq, label="left", closed="left").agg(agg)
        r=r.dropna(subset=["open","high","low","close"])
        save_schema(r, out)
        print(f"   [OK] {sym} {itv} rows={len(r)} -> {out}")
