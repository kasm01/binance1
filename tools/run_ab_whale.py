# tools/run_ab_whale.py
import os
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _pick_latest(outputs: Path, prefix: str, symbol: str, interval: str) -> Optional[str]:
    patt = re.compile(rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$")
    best = None
    for p in outputs.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if not m:
            continue
        stamp = m.group(1)
        if best is None or stamp > best[0]:
            best = (stamp, p)
    return str(best[1]) if best else None


def _run_case(name: str, env_overrides: Dict[str, str]) -> Tuple[str, str, str]:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})

    cmd = [sys.executable, str(ROOT / "backtest_mtf.py")]
    print(f"\n=== RUN {name} ===")
    print("ENV:", {k: env_overrides[k] for k in sorted(env_overrides.keys())})

    subprocess.check_call(cmd, cwd=str(ROOT), env=env)

    outputs = ROOT / "outputs"
    eq = _pick_latest(outputs, "equity_curve", env_overrides["BT_SYMBOL"], env_overrides["BT_MAIN_INTERVAL"])
    tr = _pick_latest(outputs, "trades", env_overrides["BT_SYMBOL"], env_overrides["BT_MAIN_INTERVAL"])
    sm = _pick_latest(outputs, "summary", env_overrides["BT_SYMBOL"], env_overrides["BT_MAIN_INTERVAL"])

    if not (eq and tr and sm):
        raise RuntimeError(f"[{name}] çıktı dosyaları bulunamadı. eq={eq} tr={tr} sm={sm}")

    return eq, tr, sm


def _summarize(eq_path: str, tr_path: str, sm_path: str) -> Dict:
    sm_row = pd.read_csv(sm_path).iloc[0].to_dict()
    eq = pd.read_csv(eq_path)
    tr = pd.read_csv(tr_path)

    out: Dict = {
        "ending_equity": float(sm_row.get("ending_equity", float("nan"))),
        "pnl": float(sm_row.get("pnl", float("nan"))),
        "pnl_pct": float(sm_row.get("pnl_pct", float("nan"))),
        "n_trades": int(sm_row.get("n_trades", 0) or 0),
        "winrate": float(sm_row.get("winrate", float("nan"))),
        "max_drawdown_pct": float(sm_row.get("max_drawdown_pct", float("nan"))),
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
        w = tr["whale_on"]
        if w.dtype == bool:
            out["tr_whale_on_true"] = int(w.sum())
        else:
            w2 = w.astype(str).str.lower().isin(["1", "true", "yes", "on"])
            out["tr_whale_on_true"] = int(w2.sum())

    if "whale_alignment" in tr.columns:
        out["tr_alignment_counts"] = tr["whale_alignment"].astype(str).str.lower().value_counts().to_dict()

    return out


def main() -> None:
    symbol = os.getenv("BT_SYMBOL", "BTCUSDT")
    interval = os.getenv("BT_MAIN_INTERVAL", "5m")
    limit = os.getenv("BT_DATA_LIMIT", "2000")
    warmup = os.getenv("BT_WARMUP_BARS", "200")
    thr = os.getenv("BT_WHALE_THR", "0.50")

    caseA_env = {
        "BT_SYMBOL": symbol,
        "BT_MAIN_INTERVAL": interval,
        "BT_DATA_LIMIT": limit,
        "BT_WARMUP_BARS": warmup,
        "BT_OUT_DIR": "outputs",
        "BT_WHALE_FILTER": "false",
        "BT_WHALE_ONLY": "false",
        "BT_WHALE_THR": thr,
        "BT_WHALE_VETO_OPPOSED": "false",
        "BT_WHALE_OPPOSED_SCALE": "1.00",
        "BT_WHALE_ALIGNED_BOOST": "1.00",
    }

    caseB_env = {
        "BT_SYMBOL": symbol,
        "BT_MAIN_INTERVAL": interval,
        "BT_DATA_LIMIT": limit,
        "BT_WARMUP_BARS": warmup,
        "BT_OUT_DIR": "outputs",
        "BT_WHALE_FILTER": "true",
        "BT_WHALE_ONLY": os.getenv("BT_WHALE_ONLY", "false"),
        "BT_WHALE_THR": thr,
        "BT_WHALE_VETO_OPPOSED": os.getenv("BT_WHALE_VETO_OPPOSED", "false"),
        "BT_WHALE_OPPOSED_SCALE": os.getenv("BT_WHALE_OPPOSED_SCALE", "0.30"),
        "BT_WHALE_ALIGNED_BOOST": os.getenv("BT_WHALE_ALIGNED_BOOST", "1.00"),
    }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"[AB] start stamp={stamp} symbol={symbol} interval={interval} thr={thr}")

    eqA, trA, smA = _run_case("A_BASELINE_NO_WHALE", caseA_env)
    sumA = _summarize(eqA, trA, smA)

    eqB, trB, smB = _run_case("B_WHALE_POLICY", caseB_env)
    sumB = _summarize(eqB, trB, smB)

    report = pd.DataFrame(
        [
            {"case": "A_BASELINE_NO_WHALE", "equity_csv": eqA, "trades_csv": trA, "summary_csv": smA, **sumA},
            {"case": "B_WHALE_POLICY", "equity_csv": eqB, "trades_csv": trB, "summary_csv": smB, **sumB},
        ]
    )

    out_path = ROOT / "outputs" / f"ab_whale_report_{symbol}_{interval}_{stamp}.csv"
    report.to_csv(out_path, index=False)

    print("\n[AB] report saved:", out_path)
    cols = ["case", "pnl", "pnl_pct", "n_trades", "winrate", "max_drawdown_pct", "tr_rows"]
    print(report[cols].to_string(index=False))


if __name__ == "__main__":
    main()
