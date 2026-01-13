# tools/compare_latest.py
import argparse
from pathlib import Path
import re
import pandas as pd


def _latest_n(outputs: Path, prefix: str, symbol: str, interval: str, n: int = 2):
    patt = re.compile(rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$")
    items = []
    for p in outputs.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if m:
            items.append((m.group(1), p))
    if len(items) < n:
        raise RuntimeError(f"{prefix} için en az {n} dosya lazım (bulunan={len(items)})")
    items.sort(key=lambda x: x[0])
    return [p for _, p in items[-n:]]


def load_summary(p: Path) -> pd.Series:
    df = pd.read_csv(p)
    return df.iloc[0]


def load_trades(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _detect_side_col(tr: pd.DataFrame):
    for c in ["side", "position_side", "entry_side", "direction", "pos_side", "bt_signal"]:
        if c in tr.columns:
            return c
    return None


def _normalize_side(v) -> str:
    s = str(v).lower().strip()
    if "long" in s or s == "buy":
        return "long"
    if "short" in s or s == "sell":
        return "short"
    return "unknown"


def _normalize_whale_dir(v) -> str:
    s = str(v).lower().strip()
    if s in ("buy", "long", "up"):
        return "long"
    if s in ("sell", "short", "down"):
        return "short"
    return "none"


def whale_alignment_stats(tr: pd.DataFrame, thr: float = 0.50) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame()

    pnl_col = None
    for c in ["realized_pnl", "pnl", "pnl_usdt", "profit"]:
        if c in tr.columns:
            pnl_col = c
            break
    if not pnl_col:
        tr = tr.copy()
        tr["realized_pnl"] = 0.0
    else:
        tr = tr.copy()
        tr["realized_pnl"] = pd.to_numeric(tr[pnl_col], errors="coerce").fillna(0.0)

    side_col = _detect_side_col(tr)
    tr["side_norm"] = tr[side_col].apply(_normalize_side) if side_col else "unknown"

    if "whale_score" in tr.columns:
        tr["whale_score"] = pd.to_numeric(tr["whale_score"], errors="coerce").fillna(0.0)
    else:
        tr["whale_score"] = 0.0

    if "whale_dir" in tr.columns:
        tr["whale_dir_norm"] = tr["whale_dir"].apply(_normalize_whale_dir)
    else:
        tr["whale_dir_norm"] = "none"

    def align(row):
        ws = float(row.get("whale_score", 0.0) or 0.0)
        wd = row.get("whale_dir_norm", "none")
        if ws < thr or wd == "none":
            return "no_whale"
        sig = row.get("side_norm", "unknown")
        if sig in ("long", "short") and wd in ("long", "short"):
            return "aligned" if sig == wd else "opposed"
        return "other"

    tr["alignment"] = tr.apply(align, axis=1)

    def agg(df):
        n = len(df)
        return pd.Series({
            "n_trades": float(n),
            "winrate": (df["realized_pnl"] > 0).mean() * 100.0 if n else 0.0,
            "pnl_sum": float(df["realized_pnl"].sum()) if n else 0.0,
        })

    return tr.groupby("alignment").apply(agg).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--whale-thr", type=float, default=0.50)
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
            "ending_equity": float(sm.get("ending_equity", float("nan"))),
            "pnl": float(sm.get("pnl", float("nan"))),
            "pnl_pct": float(sm.get("pnl_pct", float("nan"))),
            "n_trades": float(sm.get("n_trades", float("nan"))),
            "winrate": float(sm.get("winrate", float("nan"))),
            "max_dd_pct": float(sm.get("max_drawdown_pct", float("nan"))),
        }

        wa = whale_alignment_stats(tr, thr=float(args.whale_thr))
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
