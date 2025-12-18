import pandas as pd
import numpy as np

CSV_PATH = "logs/trade_journal.csv"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["bt_p_buy_raw", "pnl_usdt"])

df["is_win"] = df["pnl_usdt"] > 0

bins = np.arange(0.5, 1.01, 0.05)
df["bin"] = pd.cut(df["bt_p_buy_raw"], bins=bins, include_lowest=True)

summary = (
    df.groupby("bin")
    .agg(
        trades=("is_win", "count"),
        win_rate=("is_win", "mean"),
        avg_pnl=("pnl_usdt", "mean"),
    )
    .dropna()
)

print("\n📈 ENSEMBLE_P × WIN-RATE (BINNED)\n")
print(summary.to_string())
