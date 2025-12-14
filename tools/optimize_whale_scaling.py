import os
import re
import glob
import time
import json
import math
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not path or (not os.path.exists(path)) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _sharpe_from_equity(eq: pd.Series) -> float:
    # basit per-bar sharpe (risk-free=0). equity_curve bar frekansına göre göreceli kıyas için yeterli.
    if eq is None or len(eq) < 5:
        return 0.0
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if len(eq) < 5:
        return 0.0
    ret = eq.pct_change().dropna()
    if ret.std() == 0 or len(ret) < 5:
        return 0.0
    return float((ret.mean() / ret.std()) * math.sqrt(len(ret)))


def _max_dd_from_equity(eq: pd.Series) -> float:
    if eq is None or len(eq) < 2:
        return 0.0
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if len(eq) < 2:
        return 0.0
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, pd.NA)
    dd = dd.dropna()
    return float(dd.max()) if len(dd) else 0.0


def _score(pnl: float, max_dd: float, sharpe: float, n_trades: int, min_trades: int) -> float:
    # tek skor: pnl'ı ödüllendir, dd'yi cezalandır, sharpe bonus.
    # min_trades altı ise ağır ceza (optimizasyon “hiç trade yapma”ya kaçmasın).
    if n_trades < min_trades:
        return -1e9 + pnl  # en dibe it
    return (pnl * 1_000_000.0) + (sharpe * 1000.0) - (max_dd * 1000.0)


def run_one(cfg: dict, args) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["BT_SYMBOL"] = args.symbol
    env["BT_MAIN_INTERVAL"] = args.interval
    env["BT_DATA_LIMIT"] = str(args.data_limit)
    env["BT_WARMUP_BARS"] = str(args.warmup)

    # whale policy parametreleri
    env["BT_WHALE_FILTER"] = "1"
    env["BT_WHALE_VETO_OPPOSED"] = "0"
    env["BT_WHALE_ONLY"] = "0"

    env["BT_WHALE_THR"] = f"{cfg['thr']:.2f}"
    env["BT_WHALE_OPPOSED_SCALE"] = f"{cfg['opposed_scale']:.2f}"
    env["BT_WHALE_ALIGNED_BOOST"] = f"{cfg['aligned_boost']:.2f}"

    # notional davranışını sabitlemek istersen:
    # env["BT_BASE_ORDER_NOTIONAL"] = "50"
    # env["BT_MAX_POSITION_NOTIONAL"] = "500"

    # çıktılar aynı klasöre
    env["BT_OUT_DIR"] = args.out_dir

    # backtest çalıştır
    cmd = [args.python, "backtest_mtf.py"]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0

    # en son üretilen dosyaları yakala
    # backtest_mtf.py dosya adında timestamp var, bu yüzden "latest" ile alıyoruz.
    eq_path = _latest(f"{args.out_dir}/equity_curve_{args.symbol}_{args.interval}_*.csv")
    tr_path = _latest(f"{args.out_dir}/trades_{args.symbol}_{args.interval}_*.csv")
    sm_path = _latest(f"{args.out_dir}/summary_{args.symbol}_{args.interval}_*.csv")

    eq = _safe_read_csv(eq_path)
    tr = _safe_read_csv(tr_path)
    sm = _safe_read_csv(sm_path)

    # metrikler
    pnl = float(sm["pnl"].iloc[0]) if ("pnl" in sm.columns and len(sm)) else 0.0
    pnl_pct = float(sm["pnl_pct"].iloc[0]) if ("pnl_pct" in sm.columns and len(sm)) else 0.0
    n_trades = int(sm["n_trades"].iloc[0]) if ("n_trades" in sm.columns and len(sm)) else int(len(tr))

    max_dd = float(sm["max_drawdown_pct"].iloc[0] / 100.0) if ("max_drawdown_pct" in sm.columns and len(sm)) else _max_dd_from_equity(eq.get("equity"))
    sharpe = _sharpe_from_equity(eq.get("equity"))

    s = _score(pnl=pnl, max_dd=max_dd, sharpe=sharpe, n_trades=n_trades, min_trades=args.min_trades)

    return {
        "thr": cfg["thr"],
        "opposed_scale": cfg["opposed_scale"],
        "aligned_boost": cfg["aligned_boost"],
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "n_trades": n_trades,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "score": s,
        "sec": dt,
        "ok": int(p.returncode == 0),
        "stderr_tail": "\n".join((p.stderr or "").splitlines()[-15:]),
        "eq_path": eq_path or "",
        "tr_path": tr_path or "",
        "sm_path": sm_path or "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--data-limit", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--python", default="venv/bin/python")

    # arama uzayı
    ap.add_argument("--thr", default="0.40,0.50,0.60")  # virgüllü
    ap.add_argument("--opposed", default="0.10,0.20,0.30,0.40,0.50")
    ap.add_argument("--aligned", default="1.00")  # istersen 1.00,1.10,1.20

    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    thr_list = [float(x) for x in args.thr.split(",") if x.strip()]
    opp_list = [float(x) for x in args.opposed.split(",") if x.strip()]
    ali_list = [float(x) for x in args.aligned.split(",") if x.strip()]

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.out_dir) / f"opt_whale_scaling_{args.symbol}_{args.interval}_{stamp}.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    combos = []
    for thr in thr_list:
        for opp in opp_list:
            for ali in ali_list:
                combos.append({"thr": thr, "opposed_scale": opp, "aligned_boost": ali})

    rows = []
    print(f"[OPT] combos={len(combos)} symbol={args.symbol} interval={args.interval} out={args.out_dir}")

    for idx, cfg in enumerate(combos, 1):
        print(f"[OPT] ({idx}/{len(combos)}) thr={cfg['thr']:.2f} opp={cfg['opposed_scale']:.2f} ali={cfg['aligned_boost']:.2f}")
        r = run_one(cfg, args)
        rows.append(r)

        # ara kayıt
        df = pd.DataFrame(rows).sort_values("score", ascending=False)
        df.to_csv(report_path, index=False)

        best = df.iloc[0].to_dict() if len(df) else {}
        print(f"[OPT] best score={best.get('score')} pnl={best.get('pnl')} dd={best.get('max_dd')} sharpe={best.get('sharpe')} trades={best.get('n_trades')}")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(report_path, index=False)

    print("\n[OPT] report saved:", str(report_path))
    print(df.head(args.topk)[["thr","opposed_scale","aligned_boost","score","pnl","pnl_pct","n_trades","max_dd","sharpe","ok"]])


if __name__ == "__main__":
    main()
