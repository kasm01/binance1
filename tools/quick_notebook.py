from pathlib import Path
import argparse
import pandas as pd

def pick_latest(out_dir: Path, prefix: str, symbol: str, interval: str) -> Path:
    files = sorted(out_dir.glob(f"{prefix}_{symbol}_{interval}_*.csv"))
    if not files:
        raise FileNotFoundError(f"{out_dir} içinde {prefix}_{symbol}_{interval}_*.csv yok")
    return files[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    eq_p = pick_latest(out_dir, "equity_curve", args.symbol, args.interval)
    tr_p = pick_latest(out_dir, "trades", args.symbol, args.interval)
    sm_p = pick_latest(out_dir, "summary", args.symbol, args.interval)

    eq = pd.read_csv(eq_p)
    tr = pd.read_csv(tr_p)
    sm = pd.read_csv(sm_p)

    print("[FILES]")
    print(" equity:", eq_p)
    print(" trades:", tr_p)
    print(" summary:", sm_p)

    print("\n[SUMMARY CSV]")
    print(sm.to_string(index=False))

    print("\n[TRADES HEAD]")
    print(tr.head(5).to_string(index=False))

    if "realized_pnl" in tr.columns:
        pnl = pd.to_numeric(tr["realized_pnl"], errors="coerce").fillna(0.0)
        print("\n[PNL STATS]")
        print(" trades:", len(tr))
        print(" pnl_sum:", float(pnl.sum()))
        print(" pnl_mean:", float(pnl.mean()))
        print(" winrate:", float((pnl > 0).mean() * 100.0))

if __name__ == "__main__":
    main()
