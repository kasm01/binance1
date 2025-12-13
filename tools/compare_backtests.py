import argparse
from pathlib import Path
import re
import pandas as pd


def _latest_n(outputs: Path, prefix: str, symbol: str, interval: str, n: int = 2):
    patt = re.compile(rf"^{prefix}_{symbol}_{interval}_(\d{{8}}_\d{{6}})\.csv$")
    items = []
    for p in outputs.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if m:
            items.append((m.group(1), p))
    if len(items) < n:
        raise RuntimeError(f"{prefix} için en az {n} dosya lazım")
    items.sort(key=lambda x: x[0])
    return [p for _, p in items[-n:]]


def load_summary(p: Path) -> pd.Series:
    df = pd.read_csv(p)
    return df.iloc[0]


def load_trades(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def whale_alignment_stats(tr: pd.DataFrame) -> pd.DataFrame:
    if "whale_dir" not in tr.columns:
        return pd.DataFrame()

    def align(row):
        if row.get("whale_score", 0) <= 0:
            return "no_whale"
        if row["side"] == "long" and row["whale_dir"] == "buy":
            return "aligned"
        if row["side"] == "short" and row["whale_dir"] == "sell":
            return "aligned"
        if row["side"] == "long" and row["whale_dir"] == "sell":
            return "opposed"
        if row["side"] == "short" and row["whale_dir"] == "buy":
            return "opposed"
        return "other"

    tr = tr.copy()
    tr["alignment"] = tr.apply(align, axis=1)

    def agg(df):
        return pd.Series({
            "n_trades": len(df),
            "winrate": (df["realized_pnl"] > 0).mean() * 100 if len(df) else 0,
            "pnl_sum": df["realized_pnl"].sum()
        })

    return tr.groupby("alignment").apply(agg).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    out = Path(args.outputs)

    sum_files = _latest_n(out, "summary", args.symbol, args.interval)
    tr_files = _latest_n(out, "trades", args.symbol, args.interval)

    rows = []

    for tag, s_path, t_path in zip(["older", "latest"], sum_files, tr_files):
        sm = load_summary(s_path)
        tr = load_trades(t_path)

        base = {
            "run": tag,
            "ending_equity": sm["ending_equity"],
            "pnl": sm["pnl"],
            "pnl_pct": sm["pnl_pct"],
            "n_trades": sm["n_trades"],
            "winrate": sm["winrate"],
            "max_dd_pct": sm["max_drawdown_pct"],
        }

        wa = whale_alignment_stats(tr)
        for _, r in wa.iterrows():
            base[f"{r['alignment']}_trades"] = r["n_trades"]
            base[f"{r['alignment']}_winrate"] = r["winrate"]
            base[f"{r['alignment']}_pnl"] = r["pnl_sum"]

        rows.append(base)

    df = pd.DataFrame(rows)
    out_path = out / f"compare_latest_{args.symbol}_{args.interval}.csv"
    df.to_csv(out_path, index=False)

    print("[COMPARE] saved:", out_path)
    print(df)


if __name__ == "__main__":
    main()
