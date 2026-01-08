import os, time, math, csv
import requests
from datetime import datetime, timezone, timedelta

BASE = "https://api.binance.com/api/v3/klines"

def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def fetch(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    r = requests.get(BASE, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    symbols = os.getenv("SYMBOLS", "BTCUSDT").split(",")
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    interval = os.getenv("FETCH_INTERVAL", "1m").strip()
    months = int(os.getenv("FETCH_MONTHS", "6"))
    out_dir = os.getenv("OUT_DIR", "data/offline_cache").strip()
    os.makedirs(out_dir, exist_ok=True)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(months*30.5))

    print(f"[FETCH] symbols={symbols} interval={interval} months~={months} start={start} end={end}")

    for sym in symbols:
        out_path = os.path.join(out_dir, f"{sym}_{interval}_6m.csv")
        print(f"[FETCH] {sym} -> {out_path}")
        rows = []
        cur = start
        cur_ms = ms(cur)
        end_ms = ms(end)

        # Binance returns up to 1000 klines per call
        # For 1m: 1000 minutes ~ 16.6h per call
        # Iterate until done
        loops = 0
        while cur_ms < end_ms:
            loops += 1
            data = fetch(sym, interval, cur_ms, end_ms, 1000)
            if not data:
                break
            rows.extend(data)
            last_open = data[-1][0]
            cur_ms = last_open + 1
            if loops % 20 == 0:
                print(f"[FETCH] {sym} progress calls={loops} rows={len(rows)} last_open={last_open}")
            time.sleep(0.25)  # be gentle

        # write raw 12-col csv (no header) compatible with your loader
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r[:12])

        print(f"[FETCH] {sym} DONE rows={len(rows)}")
    print("[FETCH] ALL DONE")

if __name__ == "__main__":
    main()
