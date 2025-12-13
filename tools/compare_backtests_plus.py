import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def _is_probably_valid_csv(p: Path, min_bytes: int = 20) -> bool:
    try:
        if not p.exists() or p.stat().st_size < min_bytes:
            return False
        head = p.open("r", encoding="utf-8", errors="ignore").readline().strip()
        # header boşsa ya da virgül yoksa şüpheli
        if not head or ("," not in head):
            return False
        return True
    except Exception:
        return False


def _list_valid(outputs: Path, prefix: str, symbol: str, interval: str):
    patt = re.compile(rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$")
    items = []
    for p in outputs.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if not m:
            continue
        if not _is_probably_valid_csv(p):
            continue
        items.append((m.group(1), p))
    items.sort(key=lambda x: x[0])
    return items


def _latest_n(outputs: Path, prefix: str, symbol: str, interval: str, n: int = 2):
    items = _list_valid(outputs, prefix, symbol, interval)
    if len(items) < n:
        raise RuntimeError(f"{prefix} için en az {n} adet VALID dosya lazım. (bulunan={len(items)})")
    return [p for _, p in items[-n:]]


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # valid filtreden geçse bile edge-case olabilir
        return pd.DataFrame()


def _read_equity(path: Path) -> pd.DataFrame:
    eq = _read_csv_safe(path)
    if eq.empty:
        return eq
    if "time" in eq.columns:
        eq["time"] = pd.to_datetime(eq["time"], errors="coerce", utc=True)
    # bar yoksa indeks üret
    if "bar" not in eq.columns:
        eq["bar"] = range(len(eq))
    return eq


def _read_trades(path: Path) -> pd.DataFrame:
    tr = _read_csv_safe(path)
    if tr.empty:
        return tr
    # pnl kolonu normalize
    if "realized_pnl" in tr.columns:
        tr["realized_pnl"] = pd.to_numeric(tr["realized_pnl"], errors="coerce").fillna(0.0)
    elif "pnl" in tr.columns:
        tr["realized_pnl"] = pd.to_numeric(tr["pnl"], errors="coerce").fillna(0.0)
    else:
        tr["realized_pnl"] = 0.0

    # whale_score numeric
    if "whale_score" in tr.columns:
        tr["whale_score"] = pd.to_numeric(tr["whale_score"], errors="coerce").fillna(0.0)
    else:
        tr["whale_score"] = 0.0

    # whale_dir normalize
    if "whale_dir" in tr.columns:
        tr["whale_dir"] = tr["whale_dir"].astype(str).str.lower().str.strip()
    else:
        tr["whale_dir"] = "none"

    # side normalize
    side_col = None
    for c in ["side", "position_side", "entry_side", "direction", "pos_side", "bt_signal"]:
        if c in tr.columns:
            side_col = c
            break

    def norm_side(v):
        s = str(v).lower().strip()
        if "long" in s or s == "buy":
            return "long"
        if "short" in s or s == "sell":
            return "short"
        return "unknown"

    tr["side_norm"] = tr[side_col].apply(norm_side) if side_col else "unknown"
    return tr


def _overlay_equity(eq_old: pd.DataFrame, eq_new: pd.DataFrame, out: Path, symbol: str, interval: str):
    if eq_old.empty or eq_new.empty:
        return None
    plt.figure()
    plt.plot(eq_old["bar"], eq_old["equity"], label="older")
    plt.plot(eq_new["bar"], eq_new["equity"], label="latest")
    plt.title(f"Equity Overlay | {symbol} {interval}")
    plt.xlabel("bar")
    plt.ylabel("equity")
    plt.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _summary_metrics(sm_path: Path):
    sm = _read_csv_safe(sm_path)
    if sm.empty:
        return {"ending_equity": None, "pnl": None, "pnl_pct": None, "n_trades": None, "winrate": None, "max_dd_pct": None}
    row = sm.iloc[0].to_dict()
    # farklı isimler için tolerans
    def g(*keys, default=None):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return float(row[k])
        return default
    return {
        "ending_equity": g("ending_equity", "equity", default=None),
        "pnl": g("pnl", default=None),
        "pnl_pct": g("pnl_pct", default=None),
        "n_trades": g("n_trades", default=None),
        "winrate": g("winrate", default=None),
        "max_dd_pct": g("max_drawdown_pct", "max_dd_pct", default=None),
    }


def _attach_alignment(tr: pd.DataFrame, thr: float):
    if tr.empty:
        return tr
    whale_on = (tr["whale_score"] >= thr) & (~tr["whale_dir"].isin(["none", "nan", "null", ""]))
    tr = tr.copy()
    tr["whale_on"] = whale_on

    def align(row):
        if not row["whale_on"]:
            return "no_whale"
        sig = row.get("side_norm", "unknown")
        wd = row.get("whale_dir", "none")
        if sig in ("long", "short") and wd in ("long", "short"):
            return "aligned" if sig == wd else "opposed"
        return "other"

    tr["alignment"] = tr.apply(align, axis=1)
    return tr


def _agg_by_alignment(tr: pd.DataFrame):
    if tr.empty:
        return pd.DataFrame(columns=["alignment","n_trades","winrate","pnl_sum"])
    def agg(df):
        n = len(df)
        wins = int((df["realized_pnl"] > 0).sum())
        return pd.Series({
            "n_trades": float(n),
            "winrate": (wins / n * 100.0) if n else 0.0,
            "pnl_sum": float(df["realized_pnl"].sum()),
        })
    return tr.groupby("alignment").apply(agg).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--thr-min", type=float, default=0.0)
    ap.add_argument("--thr-max", type=float, default=1.0)
    ap.add_argument("--thr-step", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.outputs)

    eq_old_path, eq_new_path = _latest_n(out_dir, "equity_curve", args.symbol, args.interval, n=2)
    tr_old_path, tr_new_path = _latest_n(out_dir, "trades", args.symbol, args.interval, n=2)
    sm_old_path, sm_new_path = _latest_n(out_dir, "summary", args.symbol, args.interval, n=2)

    print(f"[LOAD] older equity: {eq_old_path}")
    print(f"[LOAD] latest equity: {eq_new_path}")
    print(f"[LOAD] older trades: {tr_old_path}")
    print(f"[LOAD] latest trades: {tr_new_path}")
    print(f"[LOAD] older summary: {sm_old_path}")
    print(f"[LOAD] latest summary: {sm_new_path}")

    eq_old, eq_new = _read_equity(eq_old_path), _read_equity(eq_new_path)
    tr_old, tr_new = _read_trades(tr_old_path), _read_trades(tr_new_path)

    # equity overlay
    overlay_path = out_dir / "plots" / f"equity_overlay_{args.symbol}_{args.interval}.png"
    p = _overlay_equity(eq_old, eq_new, overlay_path, args.symbol, args.interval)
    if p:
        print(f"[PLOT] saved: {p}")

    # delta metrics (summary bazlı)
    oldm = _summary_metrics(sm_old_path)
    newm = _summary_metrics(sm_new_path)
    delta = {
        "older_n_trades": oldm["n_trades"],
        "latest_n_trades": newm["n_trades"],
        "n_trades_change_pct": ( (newm["n_trades"] - oldm["n_trades"]) / oldm["n_trades"] * 100.0 ) if oldm["n_trades"] else None,
        "older_pnl": oldm["pnl"],
        "latest_pnl": newm["pnl"],
        "pnl_change_pct": ( (newm["pnl"] - oldm["pnl"]) / abs(oldm["pnl"]) * 100.0 ) if oldm["pnl"] not in (None, 0.0) else None,
        "older_max_dd_pct": oldm["max_dd_pct"],
        "latest_max_dd_pct": newm["max_dd_pct"],
        "older_winrate": oldm["winrate"],
        "latest_winrate": newm["winrate"],
    }
    delta_path = out_dir / f"compare_delta_{args.symbol}_{args.interval}.csv"
    pd.DataFrame([delta]).to_csv(delta_path, index=False)
    print(f"[DELTA] saved: {delta_path}")
    print(pd.DataFrame([delta]))

    # thr sweep (latest run üzerinden)
    rows = []
    thr = args.thr_min
    while thr <= args.thr_max + 1e-9:
        tr2 = _attach_alignment(tr_new, thr)
        rep = _agg_by_alignment(tr2)
        # score: aligned_pnl - opposed_pnl (basit)
        aligned_pnl = float(rep.loc[rep["alignment"]=="aligned","pnl_sum"].sum()) if not rep.empty else 0.0
        opposed_pnl = float(rep.loc[rep["alignment"]=="opposed","pnl_sum"].sum()) if not rep.empty else 0.0
        total_pnl = float(tr2["realized_pnl"].sum()) if not tr2.empty else 0.0

        def get_n(name):
            if rep.empty:
                return 0.0
            s = rep.loc[rep["alignment"]==name,"n_trades"]
            return float(s.iloc[0]) if len(s) else 0.0

        n_total = float(len(tr2)) if not tr2.empty else 0.0

        score = aligned_pnl - opposed_pnl  # hedef: opposed zararını azalt / aligned’i artır
        rows.append({
            "thr": float(thr),
            "score": float(score),
            "total_pnl": float(total_pnl),
            "aligned_pnl": float(aligned_pnl),
            "opposed_pnl": float(opposed_pnl),
            "n_trades": n_total,
            "aligned_trades": get_n("aligned"),
            "opposed_trades": get_n("opposed"),
            "no_whale_trades": get_n("no_whale"),
            "other_trades": get_n("other"),
        })
        thr += args.thr_step

    sweep = pd.DataFrame(rows).sort_values(["score","total_pnl"], ascending=[False, False])
    sweep_path = out_dir / f"thr_sweep_{args.symbol}_{args.interval}.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"[SWEEP] saved: {sweep_path}")

    best = sweep.head(1)
    best_path = out_dir / f"thr_best_{args.symbol}_{args.interval}.csv"
    best.to_csv(best_path, index=False)
    print(f"[BEST]  saved: {best_path}")

    print("\n[BEST] top-10 thresholds:")
    print(sweep.head(10))

    # score plot
    plt.figure()
    plt.plot(sweep.sort_values("thr")["thr"], sweep.sort_values("thr")["score"])
    plt.title(f"Threshold Score Sweep | {args.symbol} {args.interval}")
    plt.xlabel("thr")
    plt.ylabel("score (aligned_pnl - opposed_pnl)")
    score_plot = out_dir / "plots" / f"thr_score_{args.symbol}_{args.interval}.png"
    score_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(score_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] saved: {score_plot}")


if __name__ == "__main__":
    main()
