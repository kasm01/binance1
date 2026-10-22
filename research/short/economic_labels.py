from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShortLifecycleSpec:
    """Price-space lifecycle specification for SHORT research labels.

    tp_price_pct / sl_price_pct are unlevered price moves. For the current
    production ROI contract at 2x leverage, 0.55% ROI TP and 0.60% ROI SL map
    to 0.275% favorable price move and 0.30% adverse price move.
    """

    tp_price_pct: float = 0.00275
    sl_price_pct: float = 0.00300
    horizon_bars: int = 6
    round_trip_fee_pct: float = 0.00080
    conservative_tie_stop: bool = True

    def validate(self) -> None:
        if self.tp_price_pct <= 0:
            raise ValueError("tp_price_pct must be > 0")
        if self.sl_price_pct <= 0:
            raise ValueError("sl_price_pct must be > 0")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be >= 1")
        if self.round_trip_fee_pct < 0:
            raise ValueError("round_trip_fee_pct must be >= 0")


def _as_float_array(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"missing column: {col}")
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def build_short_economic_labels(
    candles: pd.DataFrame,
    rows: Optional[Iterable[int]] = None,
    spec: ShortLifecycleSpec = ShortLifecycleSpec(),
) -> pd.DataFrame:
    """Build leakage-safe SHORT economic outcomes for candidate rows.

    Candidate row r uses the *next candle open* (r + 1) as entry. Future bars
    from that entry candle through horizon_bars are inspected. If TP and SL are
    both touched in the same bar, the default policy is conservative: SL wins.

    Returned excursions are price-space moves before leverage/fees:
      mfe_pct = best favorable SHORT move = 1 - low / entry
      mae_pct = worst adverse SHORT move  = high / entry - 1

    net_return_pct includes the configured round-trip fee exactly once.
    """

    spec.validate()
    o = _as_float_array(candles, "open")
    h = _as_float_array(candles, "high")
    l = _as_float_array(candles, "low")
    c = _as_float_array(candles, "close")

    n = len(candles)
    max_candidate = n - spec.horizon_bars
    if rows is None:
        row_arr = np.arange(max(0, max_candidate), dtype=int)
    else:
        row_arr = np.asarray(list(rows), dtype=int)

    if row_arr.size == 0:
        return pd.DataFrame(columns=[
            "row", "entry_index", "entry_price", "outcome", "hit_bar",
            "gross_return_pct", "net_return_pct", "mfe_pct", "mae_pct",
            "mfe_bar", "mae_bar", "timeout_return_pct",
        ])

    valid = (row_arr >= 0) & (row_arr < max_candidate)
    row_arr = row_arr[valid]
    if row_arr.size == 0:
        return pd.DataFrame()

    entry_idx = row_arr + 1
    entry = o[entry_idx]
    offsets = np.arange(spec.horizon_bars, dtype=int)
    idx = entry_idx[:, None] + offsets[None, :]
    hw = h[idx]
    lw = l[idx]

    favorable = 1.0 - (lw / entry[:, None])
    adverse = (hw / entry[:, None]) - 1.0
    tp_hit = favorable >= spec.tp_price_pct
    sl_hit = adverse >= spec.sl_price_pct

    sentinel = spec.horizon_bars + 1
    tp_first = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), sentinel)
    sl_first = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), sentinel)

    if spec.conservative_tie_stop:
        is_sl = (sl_first <= tp_first) & (sl_first < spec.horizon_bars)
        is_tp = (tp_first < sl_first) & (tp_first < spec.horizon_bars)
    else:
        is_tp = (tp_first <= sl_first) & (tp_first < spec.horizon_bars)
        is_sl = (sl_first < tp_first) & (sl_first < spec.horizon_bars)

    timeout = ~(is_tp | is_sl)
    outcome = np.full(row_arr.size, "timeout", dtype=object)
    outcome[is_tp] = "tp"
    outcome[is_sl] = "sl"

    hit_bar = np.full(row_arr.size, spec.horizon_bars, dtype=int)
    hit_bar[is_tp] = tp_first[is_tp] + 1
    hit_bar[is_sl] = sl_first[is_sl] + 1

    timeout_exit_idx = entry_idx + spec.horizon_bars - 1
    timeout_gross = 1.0 - (c[timeout_exit_idx] / entry)
    gross = timeout_gross.copy()
    gross[is_tp] = spec.tp_price_pct
    gross[is_sl] = -spec.sl_price_pct
    net = gross - spec.round_trip_fee_pct

    mfe_bar0 = np.argmax(favorable, axis=1)
    mae_bar0 = np.argmax(adverse, axis=1)
    mfe = favorable[np.arange(row_arr.size), mfe_bar0]
    mae = adverse[np.arange(row_arr.size), mae_bar0]

    return pd.DataFrame({
        "row": row_arr,
        "entry_index": entry_idx,
        "entry_price": entry,
        "outcome": outcome,
        "hit_bar": hit_bar,
        "gross_return_pct": gross,
        "net_return_pct": net,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "mfe_bar": mfe_bar0 + 1,
        "mae_bar": mae_bar0 + 1,
        "timeout_return_pct": timeout_gross - spec.round_trip_fee_pct,
    })


def economic_summary(labels: pd.DataFrame) -> dict:
    if labels.empty:
        return {"n": 0, "net_bps": float("nan"), "win_rate": float("nan"), "profit_factor": float("nan")}
    r = labels["net_return_pct"].to_numpy(dtype=float)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    return {
        "n": int(len(r)),
        "net_bps": float(np.mean(r) * 1e4),
        "win_rate": float(np.mean(r > 0)),
        "profit_factor": float(pos / max(neg, 1e-12)),
        "tp_rate": float(np.mean(labels["outcome"].to_numpy() == "tp")),
        "sl_rate": float(np.mean(labels["outcome"].to_numpy() == "sl")),
        "timeout_rate": float(np.mean(labels["outcome"].to_numpy() == "timeout")),
        "mean_mfe_bps": float(labels["mfe_pct"].mean() * 1e4),
        "mean_mae_bps": float(labels["mae_pct"].mean() * 1e4),
        "mean_hit_bar": float(labels["hit_bar"].mean()),
    }
