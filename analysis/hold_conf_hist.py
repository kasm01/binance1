import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("logs/hold_decisions.csv")
if not CSV.exists():
    raise SystemExit("logs/hold_decisions.csv yok. Önce bot çalışsın, HOLD üretsin.")

df = pd.read_csv(CSV)

# p numeric
df["p"] = pd.to_numeric(df["p"], errors="coerce")
df = df.dropna(subset=["p"])

print("rows:", len(df))
print(df["p_source"].value_counts(dropna=False))

# histogram
plt.figure()
df["p"].hist(bins=20)
plt.title("HOLD confidence histogram (p)")
plt.xlabel("p (clamped 0..1)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("logs/hold_conf_hist.png", dpi=160)
print("✅ saved: logs/hold_conf_hist.png")
