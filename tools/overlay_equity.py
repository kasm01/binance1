# tools/overlay_equity.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _guess_equity_col(df: pd.DataFrame) -> str:
    candidates = ["equity", "balance", "portfolio", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last numeric column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df.columns[-1]
    return num_cols[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="equity_curve CSV path (baseline/hybrid)")
    ap.add_argument("--b", required=True, help="equity_curve CSV path (ema)")
    ap.add_argument("--out", required=True, help="output png path")
    ap.add_argument("--title", default="Equity Curve Overlay")
    args = ap.parse_args()

    pa = Path(args.a)
    pb = Path(args.b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dfa = pd.read_csv(pa)
    dfb = pd.read_csv(pb)

    ca = _guess_equity_col(dfa)
    cb = _guess_equity_col(dfb)

    xa = range(len(dfa))
    xb = range(len(dfb))

    plt.figure()
    plt.plot(list(xa), dfa[ca].astype(float).values, label=f"A ({pa.name})")
    plt.plot(list(xb), dfb[cb].astype(float).values, label=f"B ({pb.name})")
    plt.title(args.title)
    plt.xlabel("step")
    plt.ylabel("equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)

    print("[OK] saved:", str(out))


if __name__ == "__main__":
    main()
