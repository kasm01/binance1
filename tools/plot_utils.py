from __future__ import annotations

from pathlib import Path
import pandas as pd

def save_equity_curve_png(equity_csv: str, out_png: str | None = None) -> str | None:
    """
    equity_curve_*.csv -> equity_curve_*.png
    X ekseni: bar (varsa) yoksa index
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless / tmux için
        import matplotlib.pyplot as plt
    except Exception:
        return None

    p = Path(equity_csv)
    if not p.exists() or p.stat().st_size == 0:
        return None

    df = pd.read_csv(p)
    if df is None or df.empty:
        return None
    if "equity" not in df.columns:
        return None

    x = df["bar"] if "bar" in df.columns else range(len(df))
    y = df["equity"]

    out = Path(out_png) if out_png else p.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x, y)
    plt.title(f"Equity Curve - {p.name}")
    plt.xlabel("bar" if "bar" in df.columns else "index")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    return str(out)
