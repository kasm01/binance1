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
    equity_path = out_dir / f"equity_curve_{symbol}_{main_interval}_{tag}.csv"
    # ----------------------------------------------------------
    # Force close at end of backtest (so trades>0 even if no SL/TP hit)
    # ----------------------------------------------------------
    try:
        pm = (
            locals().get("position_manager")
            or locals().get("pm")
            or locals().get("position_mgr")
            or locals().get("pos_manager")
        )
        _sym = locals().get("symbol")
        if pm and _sym and getattr(pm, "has_open_position", None) and pm.has_open_position(_sym):
            _df = (
                locals().get("df")
                or locals().get("data")
                or locals().get("bars")
                or locals().get("ohlcv_df")
                or locals().get("klines_df")
            )
            if _df is not None and hasattr(_df, "columns") and "close" in _df.columns:
                last_price = float(_df["close"].iloc[-1])
            else:
                last_price = float(locals().get("price") or locals().get("last_price") or 0.0)
    
            closed = pm.close_position(_sym, price=last_price, reason="eob_force_close")
            if closed:
                ct = locals().get("closed_trades")
                if isinstance(ct, list):
                    ct.append(closed)
    except Exception as e:
        if system_logger:
            system_logger.error("[BT] force-close failed: %s", e, exc_info=True)

    trades_path = out_dir / f"trades_{symbol}_{main_interval}_{tag}.csv"
    summary_path = out_dir / f"summary_{symbol}_{main_interval}_{tag}.csv"

    EQ_COLS = [
        "bar", "time", "symbol", "interval", "price", "signal",
        "p_used", "p_single", "p_1m", "p_5m", "p_15m", "p_1h",
        "whale_dir", "whale_score", "whale_on", "whale_alignment", "whale_thr",
        "model_confidence_factor",
        "ens_scale", "ens_notional",
        "atr",
        "equity", "peak_equity", "max_drawdown_pct", "pnl_total"
    ]

    TR_COLS = [
        "symbol", "side", "qty", "entry_price", "notional", "interval", "opened_at",
        "sl_price", "tp_price", "trailing_pct", "atr_value", "highest_price", "lowest_price",
        "meta", "closed_at", "close_price", "realized_pnl", "close_reason",
        "bt_symbol", "bt_interval", "bt_bar", "bt_time", "bt_signal", "bt_price",
        "bt_p_used", "bt_p_single", "bt_p_1m", "bt_p_5m", "bt_p_15m", "bt_p_1h",
        "whale_dir", "whale_score", "whale_on", "whale_alignment", "whale_thr",
        "model_confidence_factor",
        "ens_scale", "ens_notional",
        "bt_atr"
    ]

    eq_df = _ensure_columns(pd.DataFrame(equity_rows), EQ_COLS)
    tr_df = _ensure_columns(pd.DataFrame(closed_trades), TR_COLS)
    sm_df = pd.DataFrame([bt_stats.summary_dict()])

    _atomic_to_csv(eq_df, equity_path)
    _atomic_to_csv(tr_df, trades_path)
    _atomic_to_csv(sm_df, summary_path)

    system_logger.info("[BT-CSV] equity_curve: %s (rows=%d)", str(equity_path), len(eq_df))
    system_logger.info("[BT-CSV] trades:      %s (trades=%d)", str(trades_path), len(tr_df))
    system_logger.info("[BT-CSV] summary:     %s", str(summary_path))


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
