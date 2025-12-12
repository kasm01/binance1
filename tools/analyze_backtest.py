import argparse
from pathlib import Path
import re
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _pick_latest(outputs_dir: Path, prefix: str, symbol: str, interval: str) -> Path:
    patt = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(symbol)}_{re.escape(interval)}_(\d{{8}}_\d{{6}})\.csv$"
    )
    candidates = []
    for p in outputs_dir.glob(f"{prefix}_{symbol}_{interval}_*.csv"):
        m = patt.match(p.name)
        if m:
            candidates.append((m.group(1), p))

    if not candidates:
        raise FileNotFoundError(
            f"{outputs_dir} içinde {prefix}_{symbol}_{interval}_*.csv bulunamadı.\n"
            f"Beklenen örnek: {prefix}_{symbol}_{interval}_YYYYMMDD_HHMMSS.csv\n"
            f"Çözüm: Önce backtest_mtf.py çalıştırılıp outputs/ altına CSV yazıldığından emin ol."
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _detect_side_col(trades_df: pd.DataFrame) -> Optional[str]:
    for c in ["side", "position_side", "entry_side", "direction", "pos_side"]:
        if c in trades_df.columns:
            return c
    return None


def _normalize_side(v) -> str:
    s = str(v).lower().strip()
    if "long" in s or s == "buy":
        return "long"
    if "short" in s or s == "sell":
        return "short"
    return "unknown"


def _detect_pnl_col(trades_df: pd.DataFrame) -> Optional[str]:
    for c in ["pnl", "realized_pnl", "pnl_usdt", "profit"]:
        if c in trades_df.columns:
            return c
    return None


def _try_merge_whale(eq: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """
    trades içine whale_score/whale_dir eklemeye çalışır.
    1) bar üzerinden merge (en temiz)
    2) zaman üzerinden merge (trade timestamp/open_time ile eq time/open_time)
    """
    if tr.empty:
        return tr

    if ("whale_score" in tr.columns) and ("whale_dir" in tr.columns):
        return tr

    # 1) bar üzerinden merge
    if ("bar" in tr.columns) and ("bar" in eq.columns) and ("whale_score" in eq.columns):
        cols = ["bar", "whale_score"]
        if "whale_dir" in eq.columns:
            cols.append("whale_dir")
        return tr.merge(eq[cols], on="bar", how="left")

    # 2) zaman üzerinden merge (best-effort)
    # trades tarafında olası timestamp alanları:
    tr_time_col = None
    for c in ["timestamp", "open_time", "time", "openTime"]:
        if c in tr.columns:
            tr_time_col = c
            break

    # equity tarafında olası zaman alanları:
    eq_time_col = None
    for c in ["time", "open_time", "timestamp"]:
        if c in eq.columns:
            eq_time_col = c
            break

    if (tr_time_col is None) or (eq_time_col is None) or ("whale_score" not in eq.columns):
        return tr

    tr2 = tr.copy()
    eq2 = eq.copy()

    # open_time ms epoch olabilir → numeric ise aynı türde eşitlemeye çalış
    def _to_numeric_ms(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        return x

    tr_num = _to_numeric_ms(tr2[tr_time_col])
    eq_num = _to_numeric_ms(eq2[eq_time_col])

    if tr_num.notna().any() and eq_num.notna().any():
        tr2["_t"] = tr_num.astype("Int64")
        eq2["_t"] = eq_num.astype("Int64")
        cols = ["_t", "whale_score"]
        if "whale_dir" in eq2.columns:
            cols.append("whale_dir")
        merged = tr2.merge(eq2[cols], on="_t", how="left").drop(columns=["_t"], errors="ignore")
        return merged

    # ISO string merge denemesi
    tr2["_t"] = tr2[tr_time_col].astype(str)
    eq2["_t"] = eq2[eq_time_col].astype(str)
    cols = ["_t", "whale_score"]
    if "whale_dir" in eq2.columns:
        cols.append("whale_dir")
    merged = tr2.merge(eq2[cols], on="_t", how="left").drop(columns=["_t"], errors="ignore")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="outputs klasörü")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--equity", default="", help="equity_curve csv yolu (boşsa latest seçer)")
    ap.add_argument("--trades", default="", help="trades csv yolu (boşsa latest seçer)")
    ap.add_argument("--summary", default="", help="summary csv yolu (boşsa latest seçer)")
    ap.add_argument("--plots-dir", default="outputs/plots")
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    equity_path = Path(args.equity) if args.equity else _pick_latest(out_dir, "equity_curve", args.symbol, args.interval)
    trades_path = Path(args.trades) if args.trades else _pick_latest(out_dir, "trades", args.symbol, args.interval)
    summary_path = Path(args.summary) if args.summary else _pick_latest(out_dir, "summary", args.symbol, args.interval)

    print(f"[LOAD] equity:  {equity_path}")
    print(f"[LOAD] trades:  {trades_path}")
    print(f"[LOAD] summary: {summary_path}")

    eq = pd.read_csv(equity_path)
    tr = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    sm = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # ---------------------------
    # 1) Equity curve çizimi
    # ---------------------------
    if "equity" not in eq.columns:
        raise ValueError("equity_curve csv içinde 'equity' kolonu yok. backtest_mtf.py equity_rows alanını kontrol et.")

    x = eq["bar"] if "bar" in eq.columns else range(len(eq))

    plt.figure()
    plt.plot(x, eq["equity"])
    plt.title(f"Equity Curve | {args.symbol} {args.interval}")
    plt.xlabel("bar")
    plt.ylabel("equity")
    p1 = plots_dir / f"equity_curve_{args.symbol}_{args.interval}.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # Drawdown grafiği (varsa)
    dd_col = None
    for c in ["max_drawdown_pct", "drawdown_pct", "dd_pct"]:
        if c in eq.columns:
            dd_col = c
            break

    p2 = None
    if dd_col:
        plt.figure()
        plt.plot(x, eq[dd_col])
        plt.title(f"Drawdown (%) | {args.symbol} {args.interval}")
        plt.xlabel("bar")
        plt.ylabel("drawdown_pct")
        p2 = plots_dir / f"drawdown_{args.symbol}_{args.interval}.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[PLOT] saved: {p1}")
    if p2 is not None:
        print(f"[PLOT] saved: {p2}")

    # ---------------------------
    # 2) Long/Short performansı
    # ---------------------------
    pnl_col = _detect_pnl_col(tr)
    side_col = _detect_side_col(tr)

    if not tr.empty and pnl_col:
        tr2 = tr.copy()
        tr2["pnl"] = pd.to_numeric(tr2[pnl_col], errors="coerce").fillna(0.0)

        if side_col:
            tr2["side_norm"] = tr2[side_col].apply(_normalize_side)
        else:
            tr2["side_norm"] = "unknown"

        def _side_stats(df: pd.DataFrame):
            n = len(df)
            wins = int((df["pnl"] > 0).sum())
            losses = int((df["pnl"] < 0).sum())
            winrate = (wins / n * 100.0) if n > 0 else 0.0
            return {
                "n_trades": n,
                "wins": wins,
                "losses": losses,
                "winrate_pct": winrate,
                "pnl_sum": float(df["pnl"].sum()),
                "pnl_mean": float(df["pnl"].mean()) if n > 0 else 0.0,
                "pnl_median": float(df["pnl"].median()) if n > 0 else 0.0,
            }

        longshort_report = {
            "long": _side_stats(tr2[tr2["side_norm"] == "long"]),
            "short": _side_stats(tr2[tr2["side_norm"] == "short"]),
            "unknown": _side_stats(tr2[tr2["side_norm"] == "unknown"]),
        }

        ls_df = pd.DataFrame([{"side": k, **v} for k, v in longshort_report.items()]).sort_values("side")
        ls_path = out_dir / f"report_long_short_{args.symbol}_{args.interval}.csv"
        ls_df.to_csv(ls_path, index=False)
        print(f"[REPORT] long/short saved: {ls_path}")
    else:
        print("[WARN] trades CSV içinde pnl kolonu bulunamadı veya dosya boş. Long/Short raporu üretilemedi.")

    # ---------------------------
    # 3) Whale var/yok karşılaştırma
    # ---------------------------
    whale_report_path = None
    if not tr.empty and pnl_col:
        trw = tr.copy()
        trw["pnl"] = pd.to_numeric(trw[pnl_col], errors="coerce").fillna(0.0)

        trw = _try_merge_whale(eq=eq, tr=trw)

        if "whale_score" in trw.columns:
            trw["whale_score"] = pd.to_numeric(trw["whale_score"], errors="coerce").fillna(0.0)
            trw["whale_on"] = trw["whale_score"] > 0.0

            def _grp(df):
                n = len(df)
                wins = int((df["pnl"] > 0).sum())
                winrate = (wins / n * 100.0) if n > 0 else 0.0
                return pd.Series(
                    {
                        "n_trades": n,
                        "winrate_pct": winrate,
                        "pnl_sum": float(df["pnl"].sum()),
                        "pnl_mean": float(df["pnl"].mean()) if n > 0 else 0.0,
                        "avg_whale_score": float(df["whale_score"].mean()) if n > 0 else 0.0,
                    }
                )

            rep = trw.groupby("whale_on").apply(_grp).reset_index()
            rep["whale_on"] = rep["whale_on"].map({True: "whale_on", False: "whale_off"})
            whale_report_path = out_dir / f"report_whale_compare_{args.symbol}_{args.interval}.csv"
            rep.to_csv(whale_report_path, index=False)
            print(f"[REPORT] whale compare saved: {whale_report_path}")
        else:
            print("[WARN] whale_score bulunamadı (ne trades’de ne equity’de). Whale karşılaştırma üretilemedi.")
    else:
        print("[WARN] trades CSV boş veya pnl kolonu yok → Whale karşılaştırma atlandı.")

    # ---------------------------
    # 4) Notebook için “tek komutla yükle” ipucu
    # ---------------------------
    print("\n[NOTEBOOK] Pandas ile hızlı yükleme örneği:")
    print("  import pandas as pd")
    print(f"  eq = pd.read_csv(r'{equity_path}')")
    print(f"  tr = pd.read_csv(r'{trades_path}')")
    print(f"  sm = pd.read_csv(r'{summary_path}')")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
