import os, re, glob, argparse
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

RE_ENV_LINE = re.compile(r"ENV:\s*(\{.*\})\s*$")
RE_STAMP = re.compile(r"\bstamp=([0-9]{8}(?:_[0-9]{6})?)\b")
RE_SUMMARY_ANY = re.compile(r"((?:outputs|out|results)[^\s]*summary[^\s]*\.csv)", re.IGNORECASE)
RE_BTCSV_SUMMARY = re.compile(r"\[BT-CSV\]\s+summary:\s+([^\s]+\.csv)", re.IGNORECASE)
RE_KEYVAL = re.compile(r"'([^']+)'\s*:\s*'([^']*)'")

def _pick_latest(pattern: str) -> str:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[ERR] no files: {pattern}")
    return files[-1]

def _parse_env_dict(env_text: str) -> Dict[str, str]:
    return {k: v for k, v in RE_KEYVAL.findall(env_text)}

def _read_summary_metrics(path: str) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {}
    r = df.iloc[0].to_dict()

    def pick(*names, default=None):
        for n in names:
            if n in r and pd.notna(r[n]):
                return r[n]
        return default

    total_return = float(pick("total_return", "return", "total_return_pct", default=0.0) or 0.0)
    sharpe = float(pick("sharpe", "sharpe_ratio", default=0.0) or 0.0)
    max_dd = float(pick("max_drawdown", "max_dd", "drawdown", default=0.0) or 0.0)
    winrate = float(pick("winrate", "win_rate", default=0.0) or 0.0)
    trades = int(pick("trades", "n_trades", default=0) or 0)

    score = (total_return * 100.0) + (sharpe * 10.0) + (winrate * 20.0) + (max_dd * 50.0)
    return {"score": score, "total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd, "winrate": winrate, "trades": trades}

def parse_grid_log(log_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    lines = open(log_path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    runs: List[Dict[str, Any]] = []
    stats = {"env_blocks": 0, "stamps": 0, "summary_hits": 0}

    cur_env: Optional[Dict[str, str]] = None
    cur_stamp: Optional[str] = None
    cur_summary_raw: Optional[str] = None

    def flush():
        nonlocal cur_env, cur_stamp, cur_summary_raw
        if cur_env is None and cur_stamp is None and cur_summary_raw is None:
            return
        runs.append({"env": cur_env or {}, "stamp": cur_stamp, "summary_raw": cur_summary_raw})
        cur_env = None
        cur_stamp = None
        cur_summary_raw = None

    for ln in lines:
        m = RE_STAMP.search(ln)
        if m:
            stats["stamps"] += 1
            if cur_env or cur_stamp or cur_summary_raw:
                flush()
            cur_stamp = m.group(1)

        m = RE_ENV_LINE.search(ln)
        if m:
            stats["env_blocks"] += 1
            if cur_env or cur_stamp or cur_summary_raw:
                flush()
            cur_env = _parse_env_dict(m.group(1))

        m = RE_BTCSV_SUMMARY.search(ln)
        if m:
            stats["summary_hits"] += 1
            cur_summary_raw = m.group(1)

        m = RE_SUMMARY_ANY.search(ln)
        if m:
            stats["summary_hits"] += 1
            cur_summary_raw = m.group(1)

    flush()
    cleaned = [r for r in runs if r.get("env") or r.get("stamp") or r.get("summary_raw")]
    return cleaned, stats

def _fallback_summaries_near_log(log_path: str, minutes: int = 180) -> List[str]:
    t0 = os.path.getmtime(log_path)
    lo = t0 - minutes * 60
    hi = t0 + minutes * 60

    cands = []
    for p in glob.glob("outputs/**/summary_*.csv", recursive=True):
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if lo <= mt <= hi:
            cands.append((abs(mt - t0), p))
    cands.sort(key=lambda x: x[0])
    return [p for _, p in cands]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--csv_out", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fallback_minutes", type=int, default=180)
    args = ap.parse_args()

    log_path = args.log or _pick_latest("logs/grid_whale_*.log")
    runs, st = parse_grid_log(log_path)

    if args.debug:
        print(f"[DEBUG] parsed runs={len(runs)} stats={st}")
        for i, r in enumerate(runs[:10]):
            print(f"[DEBUG] run#{i} stamp={r.get('stamp')} summary_raw={r.get('summary_raw')} env_keys={list((r.get('env') or {}).keys())}")

    table: List[Dict[str, Any]] = []

    # Normal yol: log'da summary varsa
    for r in runs:
        sr = r.get("summary_raw")
        if not sr:
            continue
        if not os.path.exists(sr):
            # basename fallback
            base = os.path.basename(sr)
            hits = sorted(glob.glob(f"outputs/**/{base}", recursive=True))
            if hits:
                sr = hits[-1]
        if not os.path.exists(sr):
            continue

        met = _read_summary_metrics(sr)
        if not met:
            continue

        env = r.get("env") or {}
        row = {
            **met,
            "thr": env.get("BT_WHALE_THR"),
            "opposed_scale": env.get("BT_WHALE_OPPOSED_SCALE"),
            "veto": env.get("BT_WHALE_VETO_OPPOSED"),
            "stamp": r.get("stamp"),
            "summary": sr,
        }
        table.append(row)

    # Fallback: log'da summary yoksa
    if not table and st.get("summary_hits", 0) == 0:
        near = _fallback_summaries_near_log(log_path, minutes=args.fallback_minutes)
        if args.debug:
            print(f"[DEBUG] fallback found {len(near)} summaries within ±{args.fallback_minutes}m")
            for p in near[:10]:
                print("[DEBUG]  ", p)

        for p in near:
            met = _read_summary_metrics(p)
            if not met:
                continue
            table.append({**met, "thr": None, "opposed_scale": None, "veto": None, "stamp": None, "summary": p})

    if not table:
        raise SystemExit(f"[ERR] no usable rows. log={log_path} stats={st}\n"
                         f"Fix: run grid with PYTHONUNBUFFERED=1 and tee log; then re-run summarize_grid.")

    df = pd.DataFrame(table)
    df = df.sort_values(by=["score", "total_return", "sharpe"], ascending=[False, False, False]).reset_index(drop=True)

    cols = ["score", "total_return", "sharpe", "max_drawdown", "winrate", "trades", "thr", "opposed_scale", "veto", "stamp", "summary"]

    print(f"\n[OK] log: {log_path}")
    print(f"[INFO] stats={st} usable_rows={len(df)}")

    print(f"\n=== TOP {args.top} (score desc) ===")
    print(df[cols].head(args.top).to_string(index=False))

    best = df.iloc[0].to_dict()
    print("\n=== BEST CONFIG ===")
    print(f"score={best.get('score'):.4f} return={best.get('total_return'):.4f} sharpe={best.get('sharpe'):.4f} "
          f"maxDD={best.get('max_drawdown'):.4f} winrate={best.get('winrate'):.4f} trades={best.get('trades')}")
    print(f"summary={best.get('summary')}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        df.to_csv(args.csv_out, index=False)
        print(f"\n[OK] wrote CSV: {args.csv_out}")

if __name__ == "__main__":
    main()
