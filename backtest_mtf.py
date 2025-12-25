# backtest_mtf.py
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from config.load_env import load_environment_variables
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from models.hybrid_inference import HybridModel
from core.hybrid_mtf import MultiTimeframeHybridEnsemble
from data.whale_detector import MultiTimeframeWhaleDetector
from data.anomaly_detection import AnomalyDetector

system_logger = logging.getLogger("system")


# -------------------------
# Helpers
# -------------------------
def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure df has all columns in `cols` (missing -> NaN), and order them."""
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Atomic CSV write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _force_close_eob(
    *,
    position_manager: Any,
    symbol: str,
    df: Optional[pd.DataFrame],
    closed_trades: List[Dict[str, Any]],
    system_logger: logging.Logger,
) -> None:
    """
    Force close open position at end-of-backtest so trades>0 even if no SL/TP hit.
    Safe: does nothing if no open pos / missing pm api.
    """
    try:
        if not position_manager:
            return
        has_open = getattr(position_manager, "has_open_position", None)
        close_pos = getattr(position_manager, "close_position", None)
        if not callable(has_open) or not callable(close_pos):
            return

        if not has_open(symbol):
            return

        last_price = 0.0
        if df is not None and hasattr(df, "columns") and "close" in df.columns and len(df) > 0:
            last_price = float(df["close"].iloc[-1])

        closed = close_pos(symbol, price=last_price, reason="eob_force_close")
        if closed:
            closed_trades.append(closed)

    except Exception as e:
        system_logger.error("[BT] force-close failed: %s", e, exc_info=True)


def _save_backtest_csv(
    *,
    out_dir: Path,
    symbol: str,
    main_interval: str,
    tag: str,
    equity_rows: List[Dict[str, Any]],
    closed_trades: List[Dict[str, Any]],
    bt_stats: Any,
    system_logger: logging.Logger,
) -> None:
    """Write equity/trades/summary CSV outputs."""
    equity_path = out_dir / f"equity_curve_{symbol}_{main_interval}_{tag}.csv"
    trades_path = out_dir / f"trades_{symbol}_{main_interval}_{tag}.csv"
    summary_path = out_dir / f"summary_{symbol}_{main_interval}_{tag}.csv"

    EQ_COLS = [
        "bar",
        "time",
        "symbol",
        "interval",
        "price",
        "signal",
        "p_used",
        "p_single",
        "p_1m",
        "p_5m",
        "p_15m",
        "p_1h",
        "whale_dir",
        "whale_score",
        "whale_on",
        "whale_alignment",
        "whale_thr",
        "model_confidence_factor",
        "ens_scale",
        "ens_notional",
        "atr",
        "equity",
        "peak_equity",
        "max_drawdown_pct",
        "pnl_total",
    ]

    TR_COLS = [
        "symbol",
        "side",
        "qty",
        "entry_price",
        "notional",
        "interval",
        "opened_at",
        "sl_price",
        "tp_price",
        "trailing_pct",
        "atr_value",
        "highest_price",
        "lowest_price",
        "meta",
        "closed_at",
        "close_price",
        "realized_pnl",
        "close_reason",
        "bt_symbol",
        "bt_interval",
        "bt_bar",
        "bt_time",
        "bt_signal",
        "bt_price",
        "bt_p_used",
        "bt_p_single",
        "bt_p_1m",
        "bt_p_5m",
        "bt_p_15m",
        "bt_p_1h",
        "whale_dir",
        "whale_score",
        "whale_on",
        "whale_alignment",
        "whale_thr",
        "model_confidence_factor",
        "ens_scale",
        "ens_notional",
        "bt_atr",
    ]

    eq_df = _ensure_columns(pd.DataFrame(equity_rows), EQ_COLS)
    tr_df = _ensure_columns(pd.DataFrame(closed_trades), TR_COLS)

    try:
        summary_dict = bt_stats.summary_dict() if hasattr(bt_stats, "summary_dict") else {}
    except Exception:
        summary_dict = {}

    sm_df = pd.DataFrame([summary_dict])

    _atomic_to_csv(eq_df, equity_path)
    _atomic_to_csv(tr_df, trades_path)
    _atomic_to_csv(sm_df, summary_path)

    system_logger.info("[BT-CSV] equity_curve: %s (rows=%d)", str(equity_path), len(eq_df))
    system_logger.info("[BT-CSV] trades:      %s (trades=%d)", str(trades_path), len(tr_df))
    system_logger.info("[BT-CSV] summary:     %s", str(summary_path))


# -------------------------
# Backtest main
# -------------------------
async def run_backtest() -> None:
    """
    Not: Bu fonksiyonun içeriği sende zaten vardı. Buraya sadece kritik kısım eklendi:
      - EOB force-close
      - CSV export
    Aşağıdaki değişkenler senin mevcut akışından gelmeli:
      out_dir, symbol, main_interval, tag, equity_rows, closed_trades, bt_stats, df, position_manager
    """
    global system_logger
    system_logger = logging.getLogger("system")

    # ---- TODO: burada senin backtest setup/loop kodun olacak ----
    # Aşağıdaki satırlar "placeholder" değişkenler; kendi kodunda zaten var.
    out_dir = Path(os.getenv("BT_OUT_DIR", "backtests"))
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    main_interval = os.getenv("INTERVAL", "5m")
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    equity_rows: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []
    bt_stats = type("BTStats", (), {"summary_dict": lambda self: {}})()

    df: Optional[pd.DataFrame] = None
    position_manager = None

    # ---- TODO: backtest loop burada doldurulacak ----

    # ----------------------------------------------------------
    # Force close at end of backtest (so trades>0 even if no SL/TP hit)
    # ----------------------------------------------------------
    _force_close_eob(
        position_manager=position_manager,
        symbol=symbol,
        df=df,
        closed_trades=closed_trades,
        system_logger=system_logger,
    )

    # ----------------------------------------------------------
    # Write CSV outputs
    # ----------------------------------------------------------
    _save_backtest_csv(
        out_dir=out_dir,
        symbol=symbol,
        main_interval=main_interval,
        tag=tag,
        equity_rows=equity_rows,
        closed_trades=closed_trades,
        bt_stats=bt_stats,
        system_logger=system_logger,
    )


# ==========================================================
# Entry
# ==========================================================
async def async_main() -> None:
    global system_logger
    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")
    await run_backtest()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
