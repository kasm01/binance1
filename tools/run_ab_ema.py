# tools/run_ab_ema.py
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

RE_SUMMARY = re.compile(r"\[BT-CSV\]\s+summary:\s+([^\s]+\.csv)", re.IGNORECASE)

ROOT = Path(__file__).resolve().parents[1]


def _read_summary(summary_path: str) -> Dict[str, float]:
    p = (ROOT / summary_path).resolve() if not summary_path.startswith("/") else Path(summary_path)
    if not p.exists():
        return {}

    # summary csv typically has 2 columns: metric,value OR header row with fields
    rows = list(csv.reader(p.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return {}

    out: Dict[str, float] = {}

    # try key,value pairs
    if len(rows[0]) >= 2 and rows[0][0].lower() != "metric":
        for r in rows:
            if len(r) >= 2:
                k = (r[0] or "").strip()
                v = (r[1] or "").strip()
                try:
                    out[k] = float(v)
                except Exception:
                    pass
        return out

    # try header-based
    header = [c.strip() for c in rows[0]]
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        for k, v in zip(header, r):
            try:
                out[k] = float(v)
            except Exception:
                pass
        break

    return out


def _run_case(name: str, env_overrides: Dict[str, str], log_file: Path) -> Optional[str]:
    env = os.environ.copy()
    env.update(env_overrides)

    cmd = [str(ROOT / "venv" / "bin" / "python"), str(ROOT / "backtest_mtf.py")]

    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    summary_path = None
    assert p.stdout is not None

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(f"\n=== RUN {name} ===\nENV: {env_overrides}\n")
        for line in p.stdout:
            sys.stdout.write(line)
            fp.write(line)
            m = RE_SUMMARY.search(line)
            if m:
                summary_path = m.group(1)

    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return summary_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("BT_SYMBOL", "BTCUSDT"))
    ap.add_argument("--interval", default=os.getenv("BT_MAIN_INTERVAL", "5m"))
    ap.add_argument("--data_limit", default=os.getenv("BT_DATA_LIMIT", "2000"))
    ap.add_argument("--warmup", default=os.getenv("BT_WARMUP_BARS", "200"))
    ap.add_argument("--whale_filter", default=os.getenv("BT_WHALE_FILTER", "true"))
    ap.add_argument("--whale_thr", default=os.getenv("BT_WHALE_THR", "0.50"))
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"ab_ema_{stamp}.log"

    base_env = {
        "BT_SYMBOL": str(args.symbol),
        "BT_MAIN_INTERVAL": str(args.interval),
        "BT_DATA_LIMIT": str(args.data_limit),
        "BT_WARMUP_BARS": str(args.warmup),
        "BT_WHALE_FILTER": str(args.whale_filter),
        "BT_WHALE_THR": str(args.whale_thr),
        # ensure dry backtest behavior
        "DRY_RUN": "true",
    }

    # A: EMA OFF
    envA = {
        **base_env,
        "USE_PBUY_STABILIZER_SIGNAL": "false",
    }

    # B: EMA ON
    envB = {
        **base_env,
        "USE_PBUY_STABILIZER_SIGNAL": "true",
        # optional whale gating in EMA
        "EMA_WHALE_ONLY": os.getenv("EMA_WHALE_ONLY", "false"),
        "EMA_WHALE_THR": os.getenv("EMA_WHALE_THR", str(args.whale_thr)),
    }

    sumA = _run_case("A_EMA_OFF", envA, log_path)
    sumB = _run_case("B_EMA_ON", envB, log_path)

    rows = []
    for tag, sp in [("A_EMA_OFF", sumA), ("B_EMA_ON", sumB)]:
        metrics = _read_summary(sp) if sp else {}
        rows.append({
            "case": tag,
            "summary": sp or "",
            "total_return": metrics.get("total_return", metrics.get("return", 0.0)),
            "sharpe": metrics.get("sharpe", 0.0),
            "max_drawdown": metrics.get("max_drawdown", metrics.get("max_dd", 0.0)),
            "winrate": metrics.get("winrate", 0.0),
            "trades": metrics.get("trades", metrics.get("n_trades", 0.0)),
        })

    out_csv = args.out_csv or str(ROOT / "outputs" / f"ab_ema_report_{args.symbol}_{args.interval}_{stamp}.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n[OK] A/B EMA report written:", out_csv)
    print("[OK] Log:", str(log_path))


if __name__ == "__main__":
    main()
