import time
import argparse
from datetime import datetime, timezone
import requests
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"

def ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch(symbol: str, interval: str, limit: int, start_ms=None, end_ms=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None: params["startTime"] = int(start_ms)
    if end_ms is not None: params["endTime"] = int(end_ms)
    r = requests.get(BASE, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--minutes", type=int, default=60*24*7)  # default 7 days
    ap.add_argument("--out", default="data/live_cache/BTCUSDT_5m_live.csv")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(minutes=args.minutes)

    # Binance klines: ileri doğru paginasyon (startTime) ile gidelim
    start_ms = ms(start)
    out_rows = []
    last_open = None
    while True:
        batch = fetch(args.symbol, args.interval, args.limit, start_ms=start_ms)
        if not batch:
            break
        # [ open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades,
        #   taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore ]
        for row in batch:
            ot = int(row[0])
            if last_open is not None and ot <= last_open:
                continue
            out_rows.append(row[:12])
            last_open = ot
        # sonraki sayfa: son open_time + 1
        start_ms = last_open + 1
        # bitiş kontrolü: son close_time end'i aştıysa dur
        last_close = int(batch[-1][6])
        if last_close >= ms(end):
            break
        time.sleep(args.sleep)

    df = pd.DataFrame(out_rows, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ])
    # tipleri sayısala çek
    for c in ["open","high","low","close","volume","quote_asset_volume",
              "taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["open_time","close_time","number_of_trades","ignore"]:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")

    df = df.dropna().drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {len(df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
