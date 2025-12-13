import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def _pick_latest(outputs_dir: Path, prefix: str, symbol: str, interval: str) -> Path:
    patt = re.compile(rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$")
    candidates = []
    for p in outputs_dir.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if m:
            candidates.append((m.group(1), p))
    if not candidates:
        raise FileNotFoundError(f"{outputs_dir} içinde {prefix}_{symbol}_{interval}_*.csv bulunamadı")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _detect_side_col(trades_df: pd.DataFrame):
    for c in ["side", "position_side", "entry_side", "direction", "pos_side"]:
        if c in trades_df.columns:
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
        return "buy"
    if s in ("sell", "short", "down"):
        return "sell"
    return "none"


def _grp_stats(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    wins = int((df["pnl"] > 0).sum()) if n else 0
    losses = int((df["pnl"] < 0).sum()) if n else 0
    winrate = (wins / n * 100.0) if n else 0.0
    return pd.Series({
        "n_trades": float(n),
        "wins": float(wins),
        "losses": float(losses),
        "winrate_pct": float(winrate),
        "pnl_sum": float(df["pnl"].sum()) if n else 0.0,
        "pnl_mean": float(df["pnl"].mean()) if n else 0.0,
        "pnl_median": float(df["pnl"].median()) if n else 0.0,
        "avg_whale_score": float(df["whale_score"].mean()) if ("whale_score" in df.columns and n) else 0.0,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="outputs klasörü")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--equity", default="", help="equity_curve csv yolu (boşsa latest seçer)")
    ap.add_argument("--trades", default="", help="trades csv yolu (boşsa latest seçer)")
    ap.add_argument("--summary", default="", help="summary csv yolu (boşsa latest seçer)")
    ap.add_argument("--plots-dir", default="outputs/plots")
    ap.add_argument("--whale-thr", type=float, default=0.50, help="whale_score strong eşiği")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    equity_path = Path(args.equity) if args.equity else _pick_latest(out_dir, "equity_curve", args.symbol, args.interval)
    trades_path = Path(args.trades) if args.trades else _pick_latest(out_dir, "trades", args.symbol, args.interval)
    summary_path = Path(args.summary) if args.summary else _pick_latest(out_dir, "summary", args.symbol, args.interval)

    print(f"[LOAD] equity:  {equity_path}")
    print(f"[LOAD] trades:  {trades_path}")
    print(f"[LOAD] summary: {summary_path}")

    eq = pd.read_csv(equity_path)
    tr = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    sm = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # ---------------------------
    # 1) Equity curve çizimi
    # ---------------------------
    if "equity" not in eq.columns:
        raise ValueError("equity_curve csv içinde 'equity' kolonu yok.")

    x = eq["bar"] if "bar" in eq.columns else range(len(eq))

    plt.figure()
    plt.plot(x, eq["equity"])
    plt.title(f"Equity Curve | {args.symbol} {args.interval}")
    plt.xlabel("bar")
    plt.ylabel("equity")
    p1 = plots_dir / f"equity_curve_{args.symbol}_{args.interval}.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    dd_col = None
    for c in ["max_drawdown_pct", "drawdown_pct", "dd_pct"]:
        if c in eq.columns:
            dd_col = c
            break
    if dd_col:
        plt.figure()
        plt.plot(x, eq[dd_col])
        plt.title(f"Drawdown (%) | {args.symbol} {args.interval}")
        plt.xlabel("bar")
        plt.ylabel("drawdown_pct")
        p2 = plots_dir / f"drawdown_{args.symbol}_{args.interval}.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[PLOT] saved: {p1}")
    if dd_col:
        print(f"[PLOT] saved: {p2}")

    # ---------------------------
    # Trades normalize
    # ---------------------------
    if tr.empty:
        print("[WARN] trades CSV boş. Raporlar atlandı.")
        print("\n[DONE]")
        return

    pnl_col = None
    for c in ["pnl", "realized_pnl", "pnl_usdt", "profit"]:
        if c in tr.columns:
            pnl_col = c
            break
    if not pnl_col:
        print("[WARN] trades CSV içinde pnl kolonu bulunamadı. Raporlar atlandı.")
        print("\n[DONE]")
        return

    side_col = _detect_side_col(tr)

    tr2 = tr.copy()
    tr2["pnl"] = pd.to_numeric(tr2[pnl_col], errors="coerce").fillna(0.0)
    tr2["side_norm"] = tr2[side_col].apply(_normalize_side) if side_col else "unknown"

    # whale cols direkt trades içinden
    if "whale_score" in tr2.columns:
        tr2["whale_score"] = pd.to_numeric(tr2["whale_score"], errors="coerce").fillna(0.0)
    else:
        tr2["whale_score"] = 0.0

    if "whale_dir" in tr2.columns:
        tr2["whale_dir_norm"] = tr2["whale_dir"].apply(_normalize_whale_dir)
    else:
        tr2["whale_dir_norm"] = "none"

    # ---------------------------
    # 2) Long/Short performansı
    # ---------------------------
    ls = tr2.groupby("side_norm").apply(_grp_stats).reset_index().rename(columns={"side_norm": "side"})
    ls_path = out_dir / f"report_long_short_{args.symbol}_{args.interval}.csv"
    ls.to_csv(ls_path, index=False)
    print(f"[REPORT] long/short saved: {ls_path}")

    # ---------------------------
    # 3) Whale var/yok (mevcut)
    # ---------------------------
    tr2["whale_on"] = tr2["whale_score"] > 0.0
    rep = tr2.groupby("whale_on").apply(_grp_stats).reset_index()
    rep["whale_on"] = rep["whale_on"].map({True: "whale_on", False: "whale_off"})
    whale_report_path = out_dir / f"report_whale_compare_{args.symbol}_{args.interval}.csv"
    rep.to_csv(whale_report_path, index=False)
    print(f"[REPORT] whale compare saved: {whale_report_path}")

    # ---------------------------
    # 3b) Whale strength (off/weak/strong)
    # ---------------------------
    thr = float(args.whale_thr)
    def _bucket(ws: float) -> str:
        if ws <= 0.0:
            return "off"
        if ws < thr:
            return "weak"
        return "strong"

    tr2["whale_bucket"] = tr2["whale_score"].apply(_bucket)
    rep2 = tr2.groupby("whale_bucket").apply(_grp_stats).reset_index().rename(columns={"whale_bucket": "bucket"})
    rep2_path = out_dir / f"report_whale_strength_{args.symbol}_{args.interval}.csv"
    rep2.to_csv(rep2_path, index=False)
    print(f"[REPORT] whale strength saved: {rep2_path} (thr={thr})")

    # ---------------------------
    # 3c) Whale alignment (dir vs trade side)
    # ---------------------------
    def _aligned(row) -> str:
        side = row["side_norm"]
        wd = row["whale_dir_norm"]
        if wd == "none":
            return "no_whale"
        if side == "long" and wd == "buy":
            return "aligned"
        if side == "short" and wd == "sell":
            return "aligned"
        if side in ("long", "short"):
            return "opposed"
        return "unknown"

    tr2["whale_alignment"] = tr2.apply(_aligned, axis=1)
    rep3 = tr2.groupby("whale_alignment").apply(_grp_stats).reset_index().rename(columns={"whale_alignment": "alignment"})
    rep3_path = out_dir / f"report_whale_alignment_{args.symbol}_{args.interval}.csv"
    rep3.to_csv(rep3_path, index=False)
    print(f"[REPORT] whale alignment saved: {rep3_path}")

    # ---------------------------
    # 4) Notebook ipucu
    # ---------------------------
    print("\n[NOTEBOOK] Pandas ile hızlı yükleme örneği:")
    print("  import pandas as pd")
    print(f"  eq = pd.read_csv(r'{equity_path}')")
    print(f"  tr = pd.read_csv(r'{trades_path}')")
    print(f"  sm = pd.read_csv(r'{summary_path}')")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
