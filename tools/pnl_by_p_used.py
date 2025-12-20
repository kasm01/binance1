from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd


def pick_latest(out_dir: Path, prefix: str, symbol: str, interval: str) -> Path:
    files = sorted(out_dir.glob(f"{prefix}_{symbol}_{interval}_*.csv"))
    if not files:
        raise FileNotFoundError(f"{out_dir} içinde {prefix}_{symbol}_{interval}_*.csv yok")
    return files[-1]


def compute_realized_pnl(df: pd.DataFrame) -> pd.Series:
    # realized_pnl varsa onu kullan
    if "realized_pnl" in df.columns:
        return pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)

    # yoksa entry/close + side’dan hesapla (fallback)
    need = {"side", "entry_price", "close_price"}
    if not need.issubset(set(df.columns)):
        raise RuntimeError(f"PNL hesaplanamadı. realized_pnl yok ve {need} kolonları da yok.")

    side = df["side"].astype(str).str.lower()
    entry = pd.to_numeric(df["entry_price"], errors="coerce")
    close = pd.to_numeric(df["close_price"], errors="coerce")
    pnl = close - entry
    pnl = pnl.where(side == "long", -pnl)
    return pnl.fillna(0.0)


def find_p_used_col(df: pd.DataFrame) -> str:
    # backtest_mtf bazen bt_p_used, bazen p_used basıyor
    for c in ("bt_p_used", "p_used", "ensemble_p", "p_ens"):
        if c in df.columns:
            return c
    raise RuntimeError(f"p_used kolonu bulunamadı. Mevcut kolonlar: {list(df.columns)[:40]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--out", default="")
    ap.add_argument("--bin_step", type=float, default=0.05)
    ap.add_argument("--bin_min", type=float, default=0.35)
    ap.add_argument("--bin_max", type=float, default=0.85)
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    trades_path = pick_latest(out_dir, "trades", args.symbol, args.interval)

    df = pd.read_csv(trades_path)
    if df.empty:
        raise RuntimeError(f"Trades CSV boş: {trades_path}")

    p_col = find_p_used_col(df)
    p = pd.to_numeric(df[p_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pnl = compute_realized_pnl(df)

    # bins
    edges = []
    x = args.bin_min
    while x < args.bin_max + 1e-9:
        edges.append(round(x, 4))
        x += args.bin_step
    if edges[-1] < args.bin_max:
        edges.append(args.bin_max)

    # pd.cut için edges düzeni
    bins = edges
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    df2 = pd.DataFrame({"p": p, "pnl": pnl})
    df2["bin"] = pd.cut(df2["p"], bins=bins, labels=labels, include_lowest=True, right=False)

    rep = (
        df2.groupby("bin", dropna=False)
        .agg(
            n=("pnl", "count"),
            winrate=("pnl", lambda s: float((s > 0).mean() * 100.0) if len(s) else 0.0),
            pnl_sum=("pnl", "sum"),
            pnl_mean=("pnl", "mean"),
            pnl_median=("pnl", "median"),
        )
        .reset_index()
    )

    # Basit “öneri scale” (ham): winrate*(pnl_mean) sinyalini normalize edip clamp
    # Bu sadece başlangıç; sonraki adımda bunu optimize edeceğiz.
    score = rep["pnl_mean"].fillna(0.0) * (rep["winrate"].fillna(0.0) / 100.0)
    if score.abs().max() > 0:
        norm = score / score.abs().max()
    else:
        norm = score * 0.0

    rep["suggest_scale"] = (1.0 + 0.8 * norm).clip(0.2, 2.0)  # 0.2..2.0

    out_path = args.out.strip()
    if not out_path:
        out_path = str(out_dir / f"pnl_by_p_used_{args.symbol}_{args.interval}.csv")

    rep.to_csv(out_path, index=False)

    print("[OK] trades:", trades_path)
    print("[OK] saved:", out_path)
    print(rep.to_string(index=False))


if __name__ == "__main__":
    main()
