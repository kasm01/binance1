import sys
import re
import subprocess
import os, subprocess, sys, re, json
from itertools import product
from datetime import datetime
from pathlib import Path

RE_SUMMARY = re.compile(r"\[BT-CSV\]\s+summary:\s+([^\s]+\.csv)", re.IGNORECASE)

def run_case(cmd, env, log_fp, cwd=None):
    """
    Runs a subprocess and tees stdout/stderr into both console and log file.
    Returns the detected summary CSV path (or None if not found).
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd or os.getcwd(),
    )

    summary_path = None
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
        log_fp.write(line)
        log_fp.flush()
        m = RE_SUMMARY.search(line)
        if m:
            summary_path = m.group(1)

    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

    return summary_path


ROOT = Path(__file__).resolve().parents[1]
STAMP_RE = re.compile(r"outputs/(equity_curve|trades|summary)_[A-Z0-9]+_[0-9a-zA-Z]+_(\d{8}_\d{6})\.csv")

def _run(env: dict, name: str) -> str:
    print(f"\n=== RUN {name} ===")
    print("ENV:", {k: env.get(k) for k in sorted(env) if k.startswith("BT_")})
    cmd = [str(ROOT/"venv/bin/python"), str(ROOT/"backtest_mtf.py")]
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stamp = None
    for line in p.stdout:
        sys.stdout.write(line)
        m = STAMP_RE.search(line)
        if m:
            stamp = m.group(2)
    rc = p.wait()
    if rc != 0:
        raise SystemExit(f"[ERR] {name} failed rc={rc}")
    if not stamp:
        raise SystemExit(f"[ERR] Could not detect stamp for {name}")
    print(f"[OK] {name} stamp={stamp}")
    return stamp

def main():
    base = os.environ.copy()
    base.update({
        "PYTHONUNBUFFERED": "1",
        "DRY_RUN": os.environ.get("DRY_RUN", "true"),
        "BT_SYMBOL": os.environ.get("BT_SYMBOL", "BTCUSDT"),
        "BT_MAIN_INTERVAL": os.environ.get("BT_MAIN_INTERVAL", "5m"),
        "BT_DATA_LIMIT": os.environ.get("BT_DATA_LIMIT", "2000"),
        "BT_WARMUP_BARS": os.environ.get("BT_WARMUP_BARS", "200"),
        # Whale policy vars
        "BT_WHALE_FILTER": os.environ.get("BT_WHALE_FILTER", "true"),
        "BT_WHALE_ONLY": os.environ.get("BT_WHALE_ONLY", "false"),
        "BT_WHALE_ALIGNED_BOOST": os.environ.get("BT_WHALE_ALIGNED_BOOST", "1.00"),
    })

    thr_grid = [0.40, 0.50, 0.60]
    opposed_scale_grid = [1.00, 0.50, 0.30]
    veto_grid = ["false", "true"]  # OFF/ON

    results = []
    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for thr, opp_scale, veto in product(thr_grid, opposed_scale_grid, veto_grid):
        env = base.copy()
        env.update({
            "BT_WHALE_THR": f"{thr:.2f}",
            "BT_WHALE_OPPOSED_SCALE": f"{opp_scale:.2f}",
            "BT_WHALE_VETO_OPPOSED": veto,
        })
        name = f"GRID_thr{thr:.2f}_opp{opp_scale:.2f}_veto{veto}"
        stamp = _run(env, name)

        # metric report for this stamp
        out_json = ROOT / f"logs/grid_whale_{run_stamp}_{name}.json".replace(":", "_")
        cmd = [
            str(ROOT/"venv/bin/python"),
            str(ROOT/"tools/report_metrics.py"),
            "--stampA", stamp,
            "--stampB", stamp,  # same stamp -> just prints metrics; compare is trivial
            "--symbol", env["BT_SYMBOL"],
            "--interval", env["BT_MAIN_INTERVAL"],
            "--json_out", str(out_json),
        ]
        run_case(cmd, cwd=str(ROOT, log_fp=log_fp, cwd=ROOT if 'ROOT' in globals() else None), env=os.environ.copy())

        results.append({
            "name": name,
            "stamp": stamp,
            "thr": thr,
            "opposed_scale": opp_scale,
            "veto": veto,
            "report": str(out_json),
        })

    # write index
    index_path = ROOT / f"logs/grid_whale_index_{run_stamp}.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Grid index: {index_path}")

if __name__ == "__main__":
    main()
