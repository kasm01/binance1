import os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAMP_RE = re.compile(r"outputs/(equity_curve|trades|summary)_[A-Z0-9]+_[0-9a-zA-Z]+_(\d{8}_\d{6})\.csv")

def _env(base: dict, extra: dict) -> dict:
    e = os.environ.copy()
    e.update(base)
    e.update(extra)
    return e

def _run_case(name: str, env: dict) -> str:
    print(f"\n=== RUN {name} ===")
    print("ENV:", {k: env.get(k) for k in sorted(env) if k.startswith("BT_") or k in ("USE_PBUY_STABILIZER_SIGNAL","DRY_RUN","PYTHONUNBUFFERED")})
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
        raise SystemExit(f"[ERR] Could not detect stamp from output for {name}")
    print(f"[OK] {name} stamp={stamp}")
    return stamp

def main():
    base_env = {
        "PYTHONUNBUFFERED": "1",
        "DRY_RUN": os.environ.get("DRY_RUN", "true"),
        "BT_SYMBOL": os.environ.get("BT_SYMBOL", "BTCUSDT"),
        "BT_MAIN_INTERVAL": os.environ.get("BT_MAIN_INTERVAL", "5m"),
        "BT_DATA_LIMIT": os.environ.get("BT_DATA_LIMIT", "2000"),
        "BT_WARMUP_BARS": os.environ.get("BT_WARMUP_BARS", "200"),
    }

    stampA = _run_case("A_EMA_OFF", _env(base_env, {"USE_PBUY_STABILIZER_SIGNAL": "false"}))
    stampB = _run_case("B_EMA_ON",  _env(base_env, {"USE_PBUY_STABILIZER_SIGNAL": "true"}))

    out_json = ROOT / f"logs/ab_ema_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    cmd = [
        str(ROOT/"venv/bin/python"),
        str(ROOT/"tools/report_metrics.py"),
        "--stampA", stampA,
        "--stampB", stampB,
        "--symbol", base_env["BT_SYMBOL"],
        "--interval", base_env["BT_MAIN_INTERVAL"],
        "--json_out", str(out_json),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT), env=os.environ.copy())
    print(f"\n[OK] EMA A/B report: {out_json}")

if __name__ == "__main__":
    main()
