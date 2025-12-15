# tools/whale_entry_exit_report.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="trades CSV path")
    ap.add_argument("--out", required=True, help="output report csv path")
    args = ap.parse_args()

    p = Path(args.trades)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(p)
    cols = set(df.columns)

    # heuristics
    side_col = "side" if "side" in cols else ("position_side" if "position_side" in cols else None)
    pnl_col = "pnl" if "pnl" in cols else ("realized_pnl" if "realized_pnl" in cols else None)

    entry_score = "entry_whale_score" if "entry_whale_score" in cols else None
    entry_dir = "entry_whale_dir" if "entry_whale_dir" in cols else None

    if side_col is None or pnl_col is None:
        print("[WARN] could not detect side/pnl columns. available cols:", sorted(cols)[:40])

    # Basic report
    rep = []

    if pnl_col and side_col:
        tmp = df.copy()
        tmp[pnl_col] = pd.to_numeric(tmp[pnl_col], errors="coerce").fillna(0.0)

        # If whale snapshot exists -> bucket by strength
        if entry_score:
            tmp[entry_score] = pd.to_numeric(tmp[entry_score], errors="coerce").fillna(0.0)
            # bins
            bins = [-1, 0.2, 0.5, 0.8, 10]
            labels = ["0-0.2", "0.2-0.5", "0.5-0.8", "0.8+"]
            tmp["whale_bucket"] = pd.cut(tmp[entry_score], bins=bins, labels=labels)

            gcols = ["whale_bucket"]
            if entry_dir:
                gcols.append(entry_dir)

            grp = tmp.groupby(gcols, dropna=False)[pnl_col].agg(["count", "sum", "mean"])
            grp = grp.reset_index()
            grp.rename(columns={"count": "trades", "sum": "pnl_sum", "mean": "pnl_mean"}, inplace=True)
            rep.append(grp)

        # Side summary
        side_grp = tmp.groupby(side_col)[pnl_col].agg(["count", "sum", "mean"]).reset_index()
        side_grp.rename(columns={"count": "trades", "sum": "pnl_sum", "mean": "pnl_mean"}, inplace=True)
        side_grp.insert(0, "report", "by_side")
        rep.append(side_grp)

    if not rep:
        # just dump columns
        out.write_text("no usable report; missing columns\n", encoding="utf-8")
        print("[ERR] no usable report. (need side & pnl columns; optional entry_whale_score/dir)")
        return

    # write a single CSV (stacked with report tags)
    final = []
    for t in rep:
        if "report" not in t.columns:
            t.insert(0, "report", "by_whale" if "whale_bucket" in t.columns else "by_side")
        final.append(t)

    out_df = pd.concat(final, ignore_index=True)
    out_df.to_csv(out, index=False)
    print("[OK] whale entry→exit report:", str(out))


if __name__ == "__main__":
    main()
