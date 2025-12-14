import os
import sys
import glob
import subprocess
from datetime import datetime
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _latest(pattern: str):
    paths = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return paths[-1] if paths else None

def _run_case(name: str, env_overrides: dict):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})

    cmd = [sys.executable, os.path.join(ROOT, "backtest_mtf.py")]
    print(f"\n=== RUN {name} ===")
    print("ENV:", {k: env_overrides[k] for k in sorted(env_overrides.keys())})
    subprocess.check_call(cmd, cwd=ROOT, env=env)

    eq = _latest("outputs/equity_curve_*_*.csv")
    tr = _latest("outputs/trades_*_*.csv")
    sm = _latest("outputs/summary_*_*.csv")
    return eq, tr, sm

def _summarize(eq_path: str, tr_path: str, sm_path: str):
    sm = pd.read_csv(sm_path).iloc[0].to_dict()

    eq = pd.read_csv(eq_path)
    tr = pd.read_csv(tr_path)

    out = {
        "ending_equity": float(sm.get("ending_equity", float("nan"))),
        "pnl": float(sm.get("pnl", float("nan"))),
        "pnl_pct": float(sm.get("pnl_pct", float("nan"))),
        "n_trades": int(sm.get("n_trades", 0)),
        "winrate": float(sm.get("winrate", float("nan"))),
        "max_drawdown_pct": float(sm.get("max_drawdown_pct", float("nan"))),
        "eq_rows": int(len(eq)),
        "tr_rows": int(len(tr)),
    }

    if "whale_dir" in eq.columns:
        out["eq_whale_dir_counts"] = eq["whale_dir"].astype(str).str.lower().value_counts().head(6).to_dict()
    if "whale_score" in eq.columns:
        ws = pd.to_numeric(eq["whale_score"], errors="coerce").fillna(0.0)
        out["eq_whale_score_mean"] = float(ws.mean())
        out["eq_whale_score_p75"] = float(ws.quantile(0.75))

    if "whale_on" in tr.columns:
        out["tr_whale_on_true"] = int(pd.to_numeric(tr["whale_on"], errors="coerce").fillna(0).astype(int).sum())
    if "whale_alignment" in tr.columns:
        out["tr_alignment_counts"] = tr["whale_alignment"].astype(str).str.lower().value_counts().to_dict()

    return out

def main():
    # Varsayılanlar
    symbol = os.getenv("BT_SYMBOL", "BTCUSDT")
    interval = os.getenv("BT_MAIN_INTERVAL", "5m")
    limit = os.getenv("BT_DATA_LIMIT", "2000")
    warmup = os.getenv("BT_WARMUP_BARS", "200")
    thr = os.getenv("BT_WHALE_THR", "0.50")

    # CASE A: Whale kapalı (baseline)
    caseA_env = {
        "BT_SYMBOL": symbol,
        "BT_MAIN_INTERVAL": interval,
        "BT_DATA_LIMIT": limit,
        "BT_WARMUP_BARS": warmup,
        "BT_WHALE_FILTER": "false",
        "BT_WHALE_ONLY": "false",
        "BT_WHALE_THR": thr,
        "BT_WHALE_VETO_OPPOSED": "false",
        "BT_WHALE_OPPOSED_SCALE": "1.00",
        "BT_WHALE_ALIGNED_BOOST": "1.00",
    }

    # CASE B: Whale aktif (senin policy)
    caseB_env = {
        "BT_SYMBOL": symbol,
        "BT_MAIN_INTERVAL": interval,
        "BT_DATA_LIMIT": limit,
        "BT_WARMUP_BARS": warmup,
        "BT_WHALE_FILTER": "true",
        "BT_WHALE_ONLY": os.getenv("BT_WHALE_ONLY", "false"),
        "BT_WHALE_THR": thr,
        "BT_WHALE_VETO_OPPOSED": os.getenv("BT_WHALE_VETO_OPPOSED", "false"),
        "BT_WHALE_OPPOSED_SCALE": os.getenv("BT_WHALE_OPPOSED_SCALE", "0.30"),
        "BT_WHALE_ALIGNED_BOOST": os.getenv("BT_WHALE_ALIGNED_BOOST", "1.00"),
    }

    # outputs klasöründe eski dosyalar çoksa, karışmaması için zaman damgası verelim
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"[AB] start stamp={stamp} symbol={symbol} interval={interval} thr={thr}")

    eqA, trA, smA = _run_case("A_BASELINE_NO_WHALE", caseA_env)
    sumA = _summarize(eqA, trA, smA)

    eqB, trB, smB = _run_case("B_WHALE_POLICY", caseB_env)
    sumB = _summarize(eqB, trB, smB)

    report = pd.DataFrame([
        {"case": "A_BASELINE_NO_WHALE", "equity_csv": eqA, "trades_csv": trA, "summary_csv": smA, **sumA},
        {"case": "B_WHALE_POLICY", "equity_csv": eqB, "trades_csv": trB, "summary_csv": smB, **sumB},
    ])

    out_path = os.path.join(ROOT, "outputs", f"ab_whale_report_{symbol}_{interval}_{stamp}.csv")
    report.to_csv(out_path, index=False)
    print("\n[AB] report saved:", out_path)
    print(report[["case","pnl","pnl_pct","n_trades","winrate","max_drawdown_pct","tr_rows"]].to_string(index=False))

if __name__ == "__main__":
    main()
