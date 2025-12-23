import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOGS = sorted(Path("logs").glob("system.log*"))
pat = re.compile(r"\[EXEC\]\[BT\] sizing \| p=(?P<p>[0-9.]+) src=(?P<src>\S+) factor=(?P<factor>[0-9.]+) qty=(?P<qty>[0-9.]+) notional=(?P<notional>[0-9.]+)")

rows = []
for lf in LOGS:
    try:
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pat.search(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "file": lf.name,
                "p": float(d["p"]),
                "src": d["src"],
                "factor": float(d["factor"]),
                "qty": float(d["qty"]),
                "notional": float(d["notional"]),
            })
    except Exception:
        pass

if not rows:
    print("No BT sizing logs found.")
    raise SystemExit(0)

df = pd.DataFrame(rows)

# p clamping analizi (log zaten clamp'li olabilir, yine de check)
df["p_clamped"] = df["p"].clip(0, 1)

print("\n--- SRC COUNTS ---")
print(df["src"].value_counts())

print("\n--- P=1.0 RATIO by SRC ---")
g = df.assign(is_one=(df["p_clamped"] >= 0.9999)).groupby("src")["is_one"].mean().sort_values(ascending=False)
print(g)

print("\n--- P SUMMARY ---")
print(df.groupby("src")["p_clamped"].describe()[["count","mean","min","50%","max"]])

# histogram
plt.figure()
df["p_clamped"].hist(bins=30)
plt.title("BT sizing p histogram (clamped 0..1)")
plt.xlabel("p")
plt.ylabel("count")
plt.tight_layout()
out = Path("logs/bt_sizing_p_hist.png")
plt.savefig(out, dpi=160)
print(f"✅ saved: {out}")
