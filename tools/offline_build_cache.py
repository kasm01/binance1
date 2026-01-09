#!/usr/bin/env python3
import os, time, math, argparse
import pandas as pd
import requests

COLUMNS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]

def fetch_binance_klines_spot(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def to_df(klines):
    # Binance spot kline format: 12 alan
    df = pd.DataFrame(klines, columns=COLUMNS)
    # numeric conversions
    float_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume"]
    int_cols = ["open_time","close_time","number_of_trades","ignore"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df

def download_1m(symbol: str, months: int, out_path: str, sleep_s: float = 0.25):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(months * 30.4375 * 24 * 3600 * 1000)  # approx months
    step_limit = 1000

    all_parts = []
    cur = start_ms
    n = 0
    while cur < end_ms:
        kl = fetch_binance_klines_spot(symbol, "1m", cur, end_ms, limit=step_limit)
        if not kl:
            break
        dfp = to_df(kl)
        all_parts.append(dfp)
        n += len(dfp)
        # next start = last open_time + 60s
        last_open = int(dfp["open_time"].iloc[-1])
        cur = last_open + 60_000
        time.sleep(sleep_s)

        if n % 50000 < 1000:
            print(f"[DL] {symbol} rows={n} last_open={last_open}")

    if not all_parts:
        raise SystemExit(f"[ERR] No data for {symbol}")

    df = pd.concat(all_parts, ignore_index=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} shape={df.shape}")

def resample_from_1m(src_path: str, out_path: str, rule: str):
    df = pd.read_csv(src_path)
    # ensure columns
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = 0
    # dtype
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    dt = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df.index = dt[dt.notna()]
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")
    df = df.dropna(subset=["open","high","low","close"])

    agg = {
        "open":"first","high":"max","low":"min","close":"last",
        "volume":"sum","quote_asset_volume":"sum","number_of_trades":"sum",
        "taker_buy_base_volume":"sum","taker_buy_quote_volume":"sum","ignore":"sum"
    }
    r = df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open","high","low","close"]).copy()

    r["open_time"] = (r.index.view("int64") // 10**6).astype("int64")
    r["close_time"] = r["open_time"]  # basit: pipeline için yeterli
    r = r.reset_index(drop=True)

    # column order
    for c in COLUMNS:
        if c not in r.columns:
            r[c] = 0
    r = r[COLUMNS]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    r.to_csv(out_path, index=False)
    print(f"[OK] resample {rule} -> {out_path} shape={r.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="CSV: BTCUSDT,ETHUSDT,...")
    ap.add_argument("--data-dir", default="data/offline_cache")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rules = {
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H",
    }

    for sym in symbols:
        src_1m = os.path.join(args.data_dir, f"{sym}_1m_6m.csv")
        if not os.path.exists(src_1m):
            print(f"[MAKE] downloading 1m for {sym} months={args.months}")
            download_1m(sym, args.months, src_1m, sleep_s=args.sleep)
        else:
            print(f"[SKIP] exists {src_1m}")

        for itv, rule in rules.items():
            outp = os.path.join(args.data_dir, f"{sym}_{itv}_6m.csv")
            if os.path.exists(outp):
                print(f"[SKIP] exists {outp}")
                continue
            resample_from_1m(src_1m, outp, rule)

if __name__ == "__main__":
    main()
