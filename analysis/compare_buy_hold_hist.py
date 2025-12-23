import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HOLD = Path("logs/hold_decisions.csv")
TRADES = Path("logs/trade_decisions.csv")

if not HOLD.exists():
    print("missing:", HOLD)
    raise SystemExit(0)
if not TRADES.exists():
    print("missing:", TRADES)
    raise SystemExit(0)

h = pd.read_csv(HOLD)
t = pd.read_csv(TRADES)

h["p"] = pd.to_numeric(h.get("p"), errors="coerce")
t["p"] = pd.to_numeric(t.get("p"), errors="coerce")

h = h.dropna(subset=["p"])
t = t.dropna(subset=["p"])

h["p"] = h["p"].clip(0, 1)
t["p"] = t["p"].clip(0, 1)

print("HOLD rows:", len(h))
print("TRADE rows:", len(t))

if "p_source" in h.columns:
    print("\nHOLD p_source:")
    print(h["p_source"].value_counts(dropna=False))
if "p_source" in t.columns:
    print("\nTRADE p_source:")
    print(t["p_source"].value_counts(dropna=False))

plt.figure()
h["p"].hist(bins=20, alpha=0.5, label="HOLD")
t["p"].hist(bins=20, alpha=0.5, label="BUY/SELL")
plt.title("Confidence histogram: HOLD vs BUY/SELL (p clamped 0..1)")
plt.xlabel("p")
plt.ylabel("count")
plt.legend()
plt.tight_layout()
plt.savefig("logs/compare_buy_hold_hist.png", dpi=160)
print("✅ saved: logs/compare_buy_hold_hist.png")
