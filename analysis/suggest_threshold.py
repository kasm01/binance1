import pandas as pd
import numpy as np

CSV_PATH = "logs/trade_journal.csv"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["bt_p_buy_raw", "pnl_usdt"])

df["is_win"] = df["pnl_usdt"] > 0

bins = np.arange(0.5, 1.01, 0.05)
df["bin"] = pd.cut(df["bt_p_buy_raw"], bins=bins, include_lowest=True)

g = (
    df.groupby("bin")
    .agg(trades=("is_win", "count"), win_rate=("is_win", "mean"), avg_pnl=("pnl_usdt", "mean"))
    .dropna()
)

# En az 15 trade olan bin’leri değerlendir (az sample aldanmasın)
g2 = g[g["trades"] >= 15].copy()

print("\n--- BIN SUMMARY (trades>=15) ---\n")
print(g2.to_string())

if len(g2):
    best = g2.sort_values(["avg_pnl", "win_rate"], ascending=False).head(1)
    print("\n--- SUGGESTED BEST BIN ---\n")
    print(best.to_string())
else:
    print("\nYeterli trade yok (>=15/bin). Daha fazla veri toplanmalı.\n")
