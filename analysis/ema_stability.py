import pandas as pd

CSV_PATH = "logs/trade_journal.csv"

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["bt_p_buy_raw", "bt_p_buy_ema"])

df["ema_diff_abs"] = (df["bt_p_buy_raw"] - df["bt_p_buy_ema"]).abs()

stats = {
    "count": len(df),
    "mean_abs_diff": float(df["ema_diff_abs"].mean()) if len(df) else None,
    "median_abs_diff": float(df["ema_diff_abs"].median()) if len(df) else None,
    "high_divergence_ratio(>0.1)": float((df["ema_diff_abs"] > 0.1).mean()) if len(df) else 0.0,
}

print("\n📉 EMA STABİLİTE\n")
for k, v in stats.items():
    print(f"{k}: {v}")
