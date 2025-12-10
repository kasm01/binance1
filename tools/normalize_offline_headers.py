import pandas as pd
from pathlib import Path

data_dir = Path("data/offline_cache")

for csv_path in sorted(data_dir.glob("BTCUSDT_*_6m.csv")):
    print(f"[CHECK] {csv_path}")
    df = pd.read_csv(csv_path)

    cols = list(df.columns)
    changed = False

    rename_map = {}

    # Eski isimler → Binance standardı ile uyumlu isimler
    if "taker_buy_base_volume" in cols:
        rename_map["taker_buy_base_volume"] = "taker_buy_base_asset_volume"
        changed = True
    if "taker_buy_quote_volume" in cols:
        rename_map["taker_buy_quote_volume"] = "taker_buy_quote_asset_volume"
        changed = True

    if not changed:
        print(f"  -> skip (rename yok)")
        continue

    print(f"  -> renaming {rename_map}")

    df = df.rename(columns=rename_map)
    df.to_csv(csv_path, index=False)
    print(f"  -> saved {csv_path} (headers normalized)")
