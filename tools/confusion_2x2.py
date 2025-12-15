from __future__ import annotations

import argparse
import pandas as pd


def _norm_sig(x: str) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    if s in ("hold", "flat", "none", ""):
        return "hold"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="trades CSV path")
    ap.add_argument("--pred_col", default="bt_signal", help="prediction signal column (default: bt_signal)")
    ap.add_argument("--entry_col", default="entry_price", help="entry price column")
    ap.add_argument("--close_col", default="close_price", help="close/exit price column")
    args = ap.parse_args()

    df = pd.read_csv(args.trades)

    need_cols = [args.pred_col, args.entry_col, args.close_col]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        print(f"[ERR] missing cols: {miss}. cols={list(df.columns)}")
        return

    # normalize
    df["_pred_sig"] = df[args.pred_col].apply(_norm_sig)

    # drop HOLD predictions (confusion only for directional preds)
    df2 = df[df["_pred_sig"].isin(["long", "short"])].copy()
    if df2.empty:
        print("[ERR] no directional predictions (only HOLD).")
        return

    # y_true: price up?
    entry = pd.to_numeric(df2[args.entry_col], errors="coerce")
    close = pd.to_numeric(df2[args.close_col], errors="coerce")
    df2 = df2[entry.notna() & close.notna()].copy()
    entry = pd.to_numeric(df2[args.entry_col], errors="coerce")
    close = pd.to_numeric(df2[args.close_col], errors="coerce")

    df2["_y_true"] = (close > entry).astype(int)                # 1=up, 0=down
    df2["_y_pred"] = (df2["_pred_sig"] == "long").astype(int)   # 1=pred up (long), 0=pred down (short)

    # confusion: TP FP TN FN
    tp = int(((df2["_y_pred"] == 1) & (df2["_y_true"] == 1)).sum())
    fp = int(((df2["_y_pred"] == 1) & (df2["_y_true"] == 0)).sum())
    tn = int(((df2["_y_pred"] == 0) & (df2["_y_true"] == 0)).sum())
    fn = int(((df2["_y_pred"] == 0) & (df2["_y_true"] == 1)).sum())

    n = len(df2)
    acc = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    # also show how many holds were skipped
    holds = int((df["_pred_sig"] == "hold").sum()) if "_pred_sig" in df.columns else int((df[args.pred_col].astype(str).str.lower() == "hold").sum())

    print(f"[OK] trades={args.trades}")
    print(f"[INFO] used={n} (directional), skipped_hold={holds}, total_rows={len(df)}")
    print("confusion (pred x true) where class=UP (long):")
    print(f"TP={tp}  FP={fp}")
    print(f"FN={fn}  TN={tn}")
    print(f"acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
