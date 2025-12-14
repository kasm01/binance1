import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt


def _is_probably_valid_csv(p: Path, min_bytes: int = 20) -> bool:
    """
    Çok küçük / header'sız / bozuk CSV'leri elemek için hafif bir kontrol.
    """
    try:
        if (not p.exists()) or p.stat().st_size < min_bytes:
            return False

        with p.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.readline().strip()

        # header boşsa ya da virgül yoksa şüpheli
        if not head or ("," not in head):
            return False

        return True
    except Exception:
        return False


def _list_valid(outputs: Path, prefix: str, symbol: str, interval: str) -> List[Tuple[str, Path]]:
    """
    prefix_symbol_interval_YYYYMMDD_HHMMSS.csv formatına uyan ve valid görünen dosyaları listeler.
    """
    patt = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$"
    )

    items: List[Tuple[str, Path]] = []
    for p in outputs.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if not m:
            continue
        if not _is_probably_valid_csv(p):
            continue
        items.append((m.group(1), p))

    items.sort(key=lambda x: x[0])
    return items


def _latest_n(outputs: Path, prefix: str, symbol: str, interval: str, n: int = 2) -> List[Path]:
    items = _list_valid(outputs, prefix, symbol, interval)
    if len(items) < n:
        raise RuntimeError(f"{prefix} için en az {n} adet VALID dosya lazım. (bulunan={len(items)})")
    return [p for _, p in items[-n:]]


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # valid filtreden geçse bile edge-case olabilir
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _read_equity(path: Path) -> pd.DataFrame:
    eq = _read_csv_safe(path)
    if eq.empty:
        return eq

    # bar yoksa indeks üret
    if "bar" not in eq.columns:
        eq["bar"] = range(len(eq))

    # equity yoksa boş döndür (plot kırılmasın)
    if "equity" not in eq.columns:
        return pd.DataFrame()

    if "time" in eq.columns:
        eq["time"] = pd.to_datetime(eq["time"], errors="coerce", utc=True)

    return eq


def _read_trades(path: Path) -> pd.DataFrame:
    tr = _read_csv_safe(path)
    if tr.empty:
        return tr

    # pnl kolonu normalize
    if "realized_pnl" in tr.columns:
        tr["realized_pnl"] = pd.to_numeric(tr["realized_pnl"], errors="coerce").fillna(0.0)
    elif "pnl" in tr.columns:
        tr["realized_pnl"] = pd.to_numeric(tr["pnl"], errors="coerce").fillna(0.0)
    else:
        tr["realized_pnl"] = 0.0

    # whale_score numeric
    if "whale_score" in tr.columns:
        tr["whale_score"] = pd.to_numeric(tr["whale_score"], errors="coerce").fillna(0.0)
    else:
        tr["whale_score"] = 0.0

    # whale_dir normalize
    if "whale_dir" in tr.columns:
        tr["whale_dir"] = tr["whale_dir"].astype(str).str.lower().str.strip()
    else:
        tr["whale_dir"] = "none"

    # side normalize
    side_col: Optional[str] = None
    for c in ["side", "position_side", "entry_side", "direction", "pos_side", "bt_signal"]:
        if c in tr.columns:
            side_col = c
            break

    def norm_side(v: Any) -> str:
        s = str(v).lower().strip()
        if "long" in s or s == "buy":
            return "long"
        if "short" in s or s == "sell":
            return "short"
        return "unknown"

    if side_col:
        tr["side_norm"] = tr[side_col].apply(norm_side)
    else:
        tr["side_norm"] = "unknown"

    return tr


def _overlay_equity(eq_old: pd.DataFrame, eq_new: pd.DataFrame, out: Path, symbol: str, interval: str) -> Optional[Path]:
    if eq_old.empty or eq_new.empty:
        return None
    if ("bar" not in eq_old.columns) or ("bar" not in eq_new.columns):
        return None
    if ("equity" not in eq_old.columns) or ("equity" not in eq_new.columns):
        return None

    plt.figure()
    plt.plot(eq_old["bar"], eq_old["equity"], label="older")
    plt.plot(eq_new["bar"], eq_new["equity"], label="latest")
    plt.title(f"Equity Overlay | {symbol} {interval}")
    plt.xlabel("bar")
    plt.ylabel("equity")
    plt.legend()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def _summary_metrics(sm_path: Path) -> Dict[str, Optional[float]]:
    sm = _read_csv_safe(sm_path)
    if sm.empty:
        return {
            "ending_equity": None,
            "pnl": None,
            "pnl_pct": None,
            "n_trades": None,
            "winrate": None,
            "max_dd_pct": None,
        }

    row = sm.iloc[0].to_dict()

    def g(*keys: str, default: Optional[float] = None) -> Optional[float]:
        for k in keys:
            if k in row and pd.notna(row[k]):
                try:
                    return float(row[k])
                except Exception:
                    return default
        return default

    return {
        "ending_equity": g("ending_equity", "equity", default=None),
        "pnl": g("pnl", default=None),
        "pnl_pct": g("pnl_pct", default=None),
        "n_trades": g("n_trades", default=None),
        "winrate": g("winrate", default=None),
        "max_dd_pct": g("max_drawdown_pct", "max_dd_pct", default=None),
    }


def _attach_alignment(tr: pd.DataFrame, thr: float) -> pd.DataFrame:
    if tr.empty:
        return tr

    tr2 = tr.copy()

    whale_on = (tr2["whale_score"] >= thr) & (~tr2["whale_dir"].isin(["none", "nan", "null", ""]))
    tr2["whale_on"] = whale_on

    def align(row: pd.Series) -> str:
        if not bool(row.get("whale_on", False)):
            return "no_whale"
        sig = str(row.get("side_norm", "unknown"))
        wd = str(row.get("whale_dir", "none"))
        if sig in ("long", "short") and wd in ("long", "short"):
            return "aligned" if sig == wd else "opposed"
        return "other"

    tr2["alignment"] = tr2.apply(align, axis=1)
    return tr2


def _agg_by_alignment(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame(columns=["alignment", "n_trades", "winrate", "pnl_sum"])

    def agg(df: pd.DataFrame) -> pd.Series:
        n = len(df)
        wins = int((df["realized_pnl"] > 0).sum())
        return pd.Series(
            {
                "n_trades": float(n),
                "winrate": (wins / n * 100.0) if n else 0.0,
                "pnl_sum": float(df["realized_pnl"].sum()),
            }
        )

    # groupby.apply ile alignment kolonu korunur
    rep = tr.groupby("alignment", dropna=False).apply(agg).reset_index()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--thr-min", type=float, default=0.0)
    ap.add_argument("--thr-max", type=float, default=1.0)
    ap.add_argument("--thr-step", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.outputs)

    eq_old_path, eq_new_path = _latest_n(out_dir, "equity_curve", args.symbol, args.interval, n=2)
    tr_old_path, tr_new_path = _latest_n(out_dir, "trades", args.symbol, args.interval, n=2)
    sm_old_path, sm_new_path = _latest_n(out_dir, "summary", args.symbol, args.interval, n=2)

    print(f"[LOAD] older equity: {eq_old_path}")
    print(f"[LOAD] latest equity: {eq_new_path}")
    print(f"[LOAD] older trades: {tr_old_path}")
    print(f"[LOAD] latest trades: {tr_new_path}")
    print(f"[LOAD] older summary: {sm_old_path}")
    print(f"[LOAD] latest summary: {sm_new_path}")

    eq_old = _read_equity(eq_old_path)
    eq_new = _read_equity(eq_new_path)
    tr_old = _read_trades(tr_old_path)
    tr_new = _read_trades(tr_new_path)

    # equity overlay
    overlay_path = out_dir / "plots" / f"equity_overlay_{args.symbol}_{args.interval}.png"
    p = _overlay_equity(eq_old, eq_new, overlay_path, args.symbol, args.interval)
    if p:
        print(f"[PLOT] saved: {p}")

    # delta metrics (summary bazlı)
    oldm = _summary_metrics(sm_old_path)
    newm = _summary_metrics(sm_new_path)

    old_n = oldm.get("n_trades")
    new_n = newm.get("n_trades")
    old_pnl = oldm.get("pnl")
    new_pnl = newm.get("pnl")

    delta = {
        "older_n_trades": old_n,
        "latest_n_trades": new_n,
        "n_trades_change_pct": ((new_n - old_n) / old_n * 100.0) if (old_n not in (None, 0.0) and new_n is not None) else None,
        "older_pnl": old_pnl,
        "latest_pnl": new_pnl,
        "pnl_change_pct": ((new_pnl - old_pnl) / abs(old_pnl) * 100.0) if (old_pnl not in (None, 0.0) and new_pnl is not None) else None,
        "older_max_dd_pct": oldm.get("max_dd_pct"),
        "latest_max_dd_pct": newm.get("max_dd_pct"),
        "older_winrate": oldm.get("winrate"),
        "latest_winrate": newm.get("winrate"),
    }

    delta_path = out_dir / f"compare_delta_{args.symbol}_{args.interval}.csv"
    pd.DataFrame([delta]).to_csv(delta_path, index=False)
    print(f"[DELTA] saved: {delta_path}")
    print(pd.DataFrame([delta]))

    # thr sweep (latest run üzerinden)
    rows: List[Dict[str, Any]] = []

    thr_min = float(args.thr_min)
    thr_max = float(args.thr_max)
    thr_step = float(args.thr_step)

    if thr_step <= 0:
        raise ValueError("--thr-step 0'dan büyük olmalı")

    n_steps = int(round((thr_max - thr_min) / thr_step)) + 1
    for i in range(max(0, n_steps)):
        thr = thr_min + i * thr_step
        # floating tolerance
        if thr > thr_max + 1e-12:
            break

        tr2 = _attach_alignment(tr_new, thr)
        rep = _agg_by_alignment(tr2)

        aligned_pnl = float(rep.loc[rep["alignment"] == "aligned", "pnl_sum"].sum()) if not rep.empty else 0.0
        opposed_pnl = float(rep.loc[rep["alignment"] == "opposed", "pnl_sum"].sum()) if not rep.empty else 0.0
        total_pnl = float(tr2["realized_pnl"].sum()) if not tr2.empty else 0.0

        def get_n(name: str) -> float:
            if rep.empty:
                return 0.0
            s = rep.loc[rep["alignment"] == name, "n_trades"]
            return float(s.iloc[0]) if len(s) else 0.0

        n_total = float(len(tr2)) if not tr2.empty else 0.0

        # score: aligned_pnl - opposed_pnl (basit hedef)
        score = aligned_pnl - opposed_pnl

        rows.append(
            {
                "thr": float(thr),
                "score": float(score),
                "total_pnl": float(total_pnl),
                "aligned_pnl": float(aligned_pnl),
                "opposed_pnl": float(opposed_pnl),
                "n_trades": n_total,
                "aligned_trades": get_n("aligned"),
                "opposed_trades": get_n("opposed"),
                "no_whale_trades": get_n("no_whale"),
                "other_trades": get_n("other"),
            }
        )

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        raise RuntimeError("thr sweep boş çıktı (trade yok / csv boş / kolonlar eksik olabilir)")

    sweep = sweep.sort_values(["score", "total_pnl"], ascending=[False, False]).reset_index(drop=True)

    sweep_path = out_dir / f"thr_sweep_{args.symbol}_{args.interval}.csv"
    sweep.to_csv(sweep_path, index=False)
    print(f"[SWEEP] saved: {sweep_path}")

    best = sweep.head(1)
    best_path = out_dir / f"thr_best_{args.symbol}_{args.interval}.csv"
    best.to_csv(best_path, index=False)
    print(f"[BEST]  saved: {best_path}")

    print("\n[BEST] top-10 thresholds:")
    print(sweep.head(10))

    # score plot
    plt.figure()
    sweep_thr_sorted = sweep.sort_values("thr")
    plt.plot(sweep_thr_sorted["thr"], sweep_thr_sorted["score"])
    plt.title(f"Threshold Score Sweep | {args.symbol} {args.interval}")
    plt.xlabel("thr")
    plt.ylabel("score (aligned_pnl - opposed_pnl)")

    score_plot = out_dir / "plots" / f"thr_score_{args.symbol}_{args.interval}.png"
    score_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(score_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] saved: {score_plot}")


if __name__ == "__main__":
    main()
