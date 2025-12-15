import os
import re
import glob
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np

# -----------------------
# Helpers
# -----------------------
def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _max_drawdown(equity: np.ndarray) -> float:
    # equity: absolute equity series
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

def _sharpe(returns: np.ndarray, periods_per_year: float = 365.0) -> float:
    # returns: per-bar returns (not log), e.g. equity pct change
    r = returns[np.isfinite(returns)]
    if len(r) < 3:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * np.sqrt(periods_per_year))

def _guess_periods_per_year(interval: str) -> float:
    # very rough; ok for relative comparisons
    # crypto runs 24/7 -> 365 days
    m = re.match(r"^(\d+)([mhd])$", interval.strip())
    if not m:
        return 365.0 * 24.0 * 12.0  # default assume 5m-ish
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        bars_per_day = (24*60) / n
    elif unit == "h":
        bars_per_day = 24 / n
    else:  # d
        bars_per_day = 1 / n
    return bars_per_day * 365.0

@dataclass
class RunResult:
    name: str
    symbol: str
    interval: str
    stamp: str
    equity_csv: str
    trades_csv: str
    summary_csv: Optional[str]
    metrics: Dict[str, Any]

def load_run_by_stamp(stamp: str, symbol: Optional[str]=None, interval: Optional[str]=None) -> RunResult:
    # Find best matching files from outputs/
    eq_glob = f"outputs/equity_curve_*_{stamp}.csv"
    tr_glob = f"outputs/trades_*_{stamp}.csv"
    sm_glob = f"outputs/summary_*_{stamp}.csv"

    eq_files = sorted(glob.glob(eq_glob))
    tr_files = sorted(glob.glob(tr_glob))
    sm_files = sorted(glob.glob(sm_glob))

    if not eq_files or not tr_files:
        raise SystemExit(f"[ERR] Missing outputs for stamp={stamp}. Found equity={len(eq_files)} trades={len(tr_files)}")

    # If multiple, just pick first; optionally filter by symbol/interval substring
    def _pick(files):
        if symbol and interval:
            for f in files:
                if f"_{symbol}_" in f and f"_{interval}_" in f:
                    return f
        if symbol:
            for f in files:
                if f"_{symbol}_" in f:
                    return f
        return files[0]

    eq = _pick(eq_files)
    tr = _pick(tr_files)
    sm = _pick(sm_files) if sm_files else None

    # parse symbol/interval from filename if possible
    # equity_curve_BTCUSDT_5m_YYYYMMDD_HHMMSS.csv
    base = os.path.basename(eq)
    parts = base.replace(".csv","").split("_")
    # ["equity", "curve", SYMBOL, INTERVAL, STAMP...]
    sym = symbol or (parts[2] if len(parts) > 3 else "UNK")
    itv = interval or (parts[3] if len(parts) > 4 else "UNK")

    metrics = compute_metrics(eq, tr, interval=itv)
    return RunResult(
        name="run",
        symbol=sym,
        interval=itv,
        stamp=stamp,
        equity_csv=eq,
        trades_csv=tr,
        summary_csv=sm,
        metrics=metrics,
    )

def compute_metrics(equity_csv: str, trades_csv: str, interval: str) -> Dict[str, Any]:
    eq = _safe_read_csv(equity_csv)
    tr = _safe_read_csv(trades_csv)

    eq_col = _find_col(eq, ["equity", "balance", "equity_usdt", "account_equity", "value"])
    if eq_col is None:
        # fallback: first numeric col
        num_cols = [c for c in eq.columns if pd.api.types.is_numeric_dtype(eq[c])]
        eq_col = num_cols[-1] if num_cols else None
    if eq_col is None:
        raise SystemExit(f"[ERR] Could not find equity column in {equity_csv}. cols={list(eq.columns)}")

    equity = eq[eq_col].astype(float).to_numpy()
    rets = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    # Trades: try to infer pnl column + side
    pnl_col = _find_col(tr, ["pnl", "realized_pnl", "pnl_usdt", "profit", "profit_usdt"])
    side_col = _find_col(tr, ["side", "position_side", "direction"])
    if pnl_col is None:
        # try any numeric col named like *pnl* or *profit*
        for c in tr.columns:
            lc = c.lower()
            if "pnl" in lc or "profit" in lc:
                pnl_col = c
                break

    pnl = tr[pnl_col].astype(float).to_numpy() if pnl_col else np.array([])
    wins = int(np.sum(pnl > 0)) if pnl.size else 0
    losses = int(np.sum(pnl < 0)) if pnl.size else 0
    trades_n = int(len(tr))

    winrate = float(wins / trades_n) if trades_n else 0.0

    dd = _max_drawdown(equity)
    sharpe = _sharpe(rets, periods_per_year=_guess_periods_per_year(interval))

    total_return = float((equity[-1] / equity[0]) - 1.0) if len(equity) > 1 and equity[0] != 0 else 0.0

    return {
        "equity_col": eq_col,
        "trades": trades_n,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "total_return": total_return,
        "max_drawdown": dd,          # negative number
        "sharpe": sharpe,
        "pnl_col": pnl_col,
        "side_col": side_col,
        "equity_start": float(equity[0]) if len(equity) else None,
        "equity_end": float(equity[-1]) if len(equity) else None,
    }

def compare(A: RunResult, B: RunResult) -> Dict[str, Any]:
    keys = ["total_return", "max_drawdown", "sharpe", "winrate", "trades"]
    out = {}
    for k in keys:
        out[k] = {"A": A.metrics.get(k), "B": B.metrics.get(k)}
    # simple "score" heuristic (bigger better)
    # reward: return + sharpe + winrate, penalize drawdown magnitude
    def score(m):
        return (m.get("total_return", 0) * 100.0) + (m.get("sharpe", 0) * 10.0) + (m.get("winrate", 0) * 20.0) + (m.get("max_drawdown", 0) * 50.0)
    out["score"] = {"A": score(A.metrics), "B": score(B.metrics)}
    out["winner"] = "A" if out["score"]["A"] >= out["score"]["B"] else "B"
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stampA", required=True)
    ap.add_argument("--stampB", required=True)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--interval", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    A = load_run_by_stamp(args.stampA, symbol=args.symbol, interval=args.interval)
    B = load_run_by_stamp(args.stampB, symbol=args.symbol, interval=args.interval)

    cmpd = compare(A, B)

    print("\n=== A METRICS ===")
    print(json.dumps(A.metrics, indent=2, sort_keys=True))
    print("\n=== B METRICS ===")
    print(json.dumps(B.metrics, indent=2, sort_keys=True))
    print("\n=== COMPARE ===")
    print(json.dumps(cmpd, indent=2, sort_keys=True))

    if args.json_out:
        payload = {"A": A.__dict__, "B": B.__dict__, "compare": cmpd}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] wrote {args.json_out}")

if __name__ == "__main__":
    main()
