#!/usr/bin/env python3
import os, sys, subprocess, argparse, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv" / "bin" / "python")
BT = str(ROOT / "backtest_mtf.py")

def run_one(name: str, env: dict) -> int:
    print(f"\n=== RUN {name} ===")
    print("ENV:", {k: env[k] for k in sorted(env.keys()) if k.startswith(("BT_", "USE_", "EMA_", "PBUY_"))})
    cmd = [PY, BT]
    return subprocess.call(cmd, cwd=str(ROOT), env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("BT_SYMBOL", "BTCUSDT"))
    ap.add_argument("--interval", default=os.getenv("BT_MAIN_INTERVAL", "5m"))
    ap.add_argument("--limit", default=os.getenv("BT_DATA_LIMIT", "2000"))
    ap.add_argument("--warmup", default=os.getenv("BT_WARMUP_BARS", "200"))
    ap.add_argument("--whale_thr", default=os.getenv("BT_WHALE_THR", "0.50"))
    args = ap.parse_args()

    base = os.environ.copy()
    base.update({
        "BT_SYMBOL": str(args.symbol),
        "BT_MAIN_INTERVAL": str(args.interval),
        "BT_DATA_LIMIT": str(args.limit),
        "BT_WARMUP_BARS": str(args.warmup),
        "BT_WHALE_THR": str(args.whale_thr),
    })

    # 2x2 grid
    cases = [
        ("EMA_OFF__WHALE_OFF", {"USE_PBUY_STABILIZER_SIGNAL":"false", "BT_WHALE_FILTER":"false"}),
        ("EMA_ON__WHALE_OFF",  {"USE_PBUY_STABILIZER_SIGNAL":"true",  "BT_WHALE_FILTER":"false"}),
        ("EMA_OFF__WHALE_ON",  {"USE_PBUY_STABILIZER_SIGNAL":"false", "BT_WHALE_FILTER":"true"}),
        ("EMA_ON__WHALE_ON",   {"USE_PBUY_STABILIZER_SIGNAL":"true",  "BT_WHALE_FILTER":"true"}),
    ]

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"grid_ema_whale_{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # log'a da yaz
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"[GRID] stamp={stamp} symbol={args.symbol} interval={args.interval}\n")
        fp.flush()
        rc_all = 0
        for name, extra_env in cases:
            env = base.copy()
            env.update(extra_env)
            fp.write(f"\n=== RUN {name} ===\nENV: {extra_env}\n")
            fp.flush()
            rc = run_one(name, env)
            fp.write(f"[RC] {name} rc={rc}\n")
            fp.flush()
            if rc != 0:
                rc_all = 1
        print(f"\n[OK] log={log_path}")
        return rc_all

if __name__ == "__main__":
    raise SystemExit(main())
