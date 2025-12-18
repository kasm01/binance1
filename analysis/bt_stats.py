import pandas as pd

CSV_PATH = "logs/trade_journal.csv"

df = pd.read_csv(CSV_PATH)

# bt verisi olanları al
df = df.dropna(subset=["bt_p_buy_raw", "bt_p_buy_ema"])

df["is_win"] = df["pnl_usdt"] > 0

report = {
    "total_trades": len(df),
    "win_rate": float(df["is_win"].mean()) if len(df) else 0.0,
    "bt_raw_mean": float(df["bt_p_buy_raw"].mean()) if len(df) else None,
    "bt_raw_win_mean": float(df[df.is_win]["bt_p_buy_raw"].mean()) if df["is_win"].any() else None,
    "bt_raw_loss_mean": float(df[~df.is_win]["bt_p_buy_raw"].mean()) if (~df["is_win"]).any() else None,
    "high_conf_ratio(>0.9)": float((df["bt_p_buy_raw"] > 0.9).mean()) if len(df) else 0.0,
}

print("\n📊 BT_* GENEL İSTATİSTİK\n")
for k, v in report.items():
    print(f"{k}: {v}")
