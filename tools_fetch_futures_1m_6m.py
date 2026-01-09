#!/usr/bin/env python3
import os, time, math, argparse
import pandas as pd
import requests
from datetime import datetime, timezone

BASE = os.getenv("BINANCE_FAPI_BASE", "https://fapi.binance.com")
ENDPOINT = "/fapi/v1/klines"

COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume","ignore"
]

def utc_ms(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_chunk(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit
    }
    r = requests.get(BASE + ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    symbol = args.symbol.upper().strip()
    months = args.months
    out_path = args.out or f"data/offline_cache/{symbol}_1m_6m.csv"

    # yaklaşık 30*months gün geriye (sende zaten 6 ay ≈ 180 gün)
    now = datetime.now(timezone.utc)
    start = now.timestamp() - (30 * months * 24 * 3600)
    start_ms = int(start * 1000)
    end_ms = int(now.timestamp() * 1000)

    print(f"[FETCH] {symbol} 1m months={months} start_ms={start_ms} end_ms={end_ms}")
    all_rows = []
    cur = start_ms
    last_open = None
    n_req = 0

    while True:
        data = fetch_chunk(symbol, "1m", cur, end_ms, limit=1500)
        n_req += 1
        if not data:
            break

        # her satır 12 kolon
        all_rows.extend(data)

        last_open_time = int(data[-1][0])
        if last_open is not None and last_open_time <= last_open:
            # ilerlemiyorsa kır (koruma)
            break
        last_open = last_open_time

        # bir sonraki sayfa: son open_time + 60_000 ms
        cur = last_open_time + 60_000

        if len(data) < 1500:
            break

        if args.sleep > 0:
            time.sleep(args.sleep)

        if n_req % 25 == 0:
            print(f"[PROG] req={n_req} rows={len(all_rows)} last_open={last_open_time}")

    if not all_rows:
        raise SystemExit(f"[ERR] No data fetched for {symbol}")

        # Binance klines: 12 kolon döner. Güvenlik için uzunluğu 12 olmayan satırları ayıkla.
    good = [r for r in all_rows if isinstance(r, (list, tuple)) and len(r) == len(COLS)]
    badn = len(all_rows) - len(good)
    if badn:
        print(f"[WARN] Dropped {badn} rows with unexpected length (expected {len(COLS)}).")
    df = pd.DataFrame(good, columns=COLS)

    # tipler
    num_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_volume","taker_buy_quote_volume","ignore"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce").astype("int64")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype("int64")

    # open_time uniq kontrol (0'lanma vb yakalar)
    uniq = df["open_time"].nunique(dropna=True)
    print(f"[DONE] rows={len(df)} open_time_unique={uniq} min={df['open_time'].min()} max={df['open_time'].max()} -> {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
