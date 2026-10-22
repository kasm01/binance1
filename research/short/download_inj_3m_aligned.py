#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "INJUSDT"
INTERVAL = "3m"
START_MS = 1758313440000
END_MS = 1789417260000
LIMIT = 1500
URL = "https://fapi.binance.com/fapi/v1/klines"
OUT = Path("/home/kasm920/binance1-pro/data/offline_cache/INJUSDT_3m_360d.csv")

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


def main() -> None:
    session = requests.Session()
    rows = []
    cursor = START_MS
    req = 0
    while cursor <= END_MS:
        r = session.get(
            URL,
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": cursor,
                "endTime": END_MS,
                "limit": LIMIT,
            },
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        req += 1
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open < cursor:
            raise RuntimeError("non-advancing Binance kline cursor")
        cursor = last_open + 1
        if req % 20 == 0:
            print(f"REQ={req} rows={len(rows)} last_open={last_open}", flush=True)
        if len(batch) < LIMIT:
            break
        time.sleep(0.06)

    df = pd.DataFrame(rows, columns=COLS)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df = df[(pd.to_numeric(df.open_time) >= START_MS) & (pd.to_numeric(df.open_time) <= END_MS)]
    if df.empty:
        raise RuntimeError("no INJ data downloaded")

    expected = int((END_MS - START_MS) // (3 * 60 * 1000)) + 1
    print("ROWS", len(df), "EXPECTED", expected, "REQUESTS", req, flush=True)
    print("FIRST", int(df.open_time.iloc[0]), "LAST", int(df.open_time.iloc[-1]), flush=True)
    if len(df) != expected:
        raise RuntimeError(f"row count mismatch: got={len(df)} expected={expected}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
