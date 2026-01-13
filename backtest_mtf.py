# backtest_mtf.py
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
from datetime import datetime
from pathlib import Path

from config.load_env import load_environment_variables
from core.logger import setup_logger

from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from data.anomaly_detection import AnomalyDetector

try:
    from features.pipeline import build_features  # type: ignore
except Exception:
    build_features = None  # noqa

system_logger = logging.getLogger("system")


# -------------------------
# Helpers
# -------------------------
def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _save_backtest_csv(
    *,
    out_dir: Path,
    symbol: str,
    main_interval: str,
    tag: str,
    equity_rows: List[Dict[str, Any]],
    closed_trades: List[Dict[str, Any]],
    bt_stats: Dict[str, Any],
    system_logger: logging.Logger,
) -> None:
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
    sm_df = pd.DataFrame([bt_stats])

    _atomic_to_csv(eq_df, equity_path)
    _atomic_to_csv(tr_df, trades_path)
    _atomic_to_csv(sm_df, summary_path)

    system_logger.info("[BT-CSV] equity_curve: %s (rows=%d)", str(equity_path), len(eq_df))
    system_logger.info("[BT-CSV] trades:      %s (trades=%d)", str(trades_path), len(tr_df))
    system_logger.info("[BT-CSV] summary:     %s", str(summary_path))


def _pick_env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def _to_dt_from_ms(ms: Any) -> Optional[str]:
    try:
        x = float(ms)
        if x > 1e12:  # ms epoch
            return datetime.utcfromtimestamp(x / 1000.0).isoformat()
        if x > 1e9:  # sec epoch
            return datetime.utcfromtimestamp(x).isoformat()
    except Exception:
        pass
    return None


def _simple_p_from_returns(close: pd.Series) -> float:
    """
    Model yoksa basit bir fallback:
    - son 3 bar momentum -> p ~ sigmoid benzeri
    """
    try:
        if len(close) < 6:
            return 0.5
        r1 = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2])
        r3 = (float(close.iloc[-1]) - float(close.iloc[-4])) / float(close.iloc[-4])
        x = 8.0 * r1 + 4.0 * r3
        # basit squash
        p = 0.5 + max(-0.49, min(0.49, x))
        return float(max(0.0, min(1.0, p)))
    except Exception:
        return 0.5


async def run_backtest() -> None:
    global system_logger
    system_logger = logging.getLogger("system")

    # tools/*.py ile uyumlu env isimleri
    symbol = _pick_env("BT_SYMBOL", "SYMBOL", default="BTCUSDT").upper()
    main_interval = _pick_env("BT_MAIN_INTERVAL", "INTERVAL", default="5m")
    out_dir = Path(_pick_env("BT_OUT_DIR", default="outputs"))
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    offline_dir = Path(_pick_env("OFFLINE_DIR", default="data/offline_cache"))
    data_limit = int(_pick_env("BT_DATA_LIMIT", "DATA_LIMIT", default="2000"))
    warmup = int(_pick_env("BT_WARMUP_BARS", default="200"))
    start_equity = float(_pick_env("BT_START_EQUITY", default="1000"))

    # karar eşiği
    p_thr = float(_pick_env("BT_P_THR", default="0.55"))
    inv_thr = 1.0 - p_thr

    dry_run = str(_pick_env("DRY_RUN", default="true")).lower() in ("1", "true", "yes", "on")

    # CSV input
    main_csv = offline_dir / f"{symbol}_{main_interval}_6m.csv"
    if not main_csv.exists():
        system_logger.error("[BT] Offline CSV bulunamadı: %s", str(main_csv))
        return

    df = pd.read_csv(main_csv)
    if data_limit > 0 and len(df) > data_limit:
        df = df.tail(data_limit).reset_index(drop=True)

    if "close" not in df.columns:
        system_logger.error("[BT] CSV içinde 'close' kolonu yok: %s", str(main_csv))
        return

    # time kolonu üret (varsa open_time kullan)
    if "time" not in df.columns:
        if "open_time" in df.columns:
            df["time"] = df["open_time"].apply(_to_dt_from_ms)
        else:
            df["time"] = [None] * len(df)

    # Feature pipeline + anomaly
    anomaly_detector = AnomalyDetector(logger=system_logger)
    feat_df: Optional[pd.DataFrame] = None

    if build_features is None:
        system_logger.warning("[BT] build_features import edilemedi -> direkt df ile ilerliyorum (p fallback)")
    else:
        try:
            feat_df = build_features(df)
            # schema sende varsa burada set et (şimdilik None)
            sch_main = None
            feat_df = anomaly_detector.filter_anomalies(
                feat_df,
                schema=sch_main,
                context=f"heavy:{symbol}:{main_interval}",
            )
        except Exception as e:
            system_logger.warning("[BT] feature/anomaly aşamasında hata: %s (fallback)", e)
            feat_df = None

    # Risk + executor
    rm = RiskManager(equity_start_of_day=start_equity, logger=system_logger)
    te = TradeExecutor(
        client=None,
        risk_manager=rm,
        position_manager=None,  # backtestte state TE içinde kalsın
        logger=system_logger,
        dry_run=dry_run,
    )

    equity_rows: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []

    equity = float(start_equity)
    peak = float(start_equity)
    max_dd_pct = 0.0
    pnl_total = 0.0

    # Loop
    n = len(df)
    warmup = max(0, min(warmup, n - 1))

    for i in range(warmup, n):
        price = float(df["close"].iloc[i])
        t = df["time"].iloc[i] if "time" in df.columns else None
        t = str(t) if t is not None else ""

        # 1) trailing/SL/TP check (her barda)
        te._check_sl_tp_trailing(symbol=symbol, price=price, interval=main_interval)

        # 2) P üretimi (model yoksa fallback)
        # (feat_df varsa da şu aşamada “p” fallback kullanıyoruz; istersen HybridModel entegrasyonunu buraya ekleriz)
        p_single = _simple_p_from_returns(df["close"].iloc[: i + 1])

        # 3) sinyal
        if p_single >= p_thr:
            signal = "BUY"
        elif p_single <= inv_thr:
            signal = "SELL"
        else:
            signal = "HOLD"

        probs = {
            "p_single": float(p_single),
            "p_used": float(p_single),
        }

        extra = {
            "signal_source": "p_fallback",
            "model_confidence_factor": 1.0,
            "atr": float(df["close"].iloc[max(0, i - 1)]) * 0.0,  # atr yoksa 0
            "whale_dir": "none",
            "whale_score": 0.0,
            "ensemble_p": None,
        }

        # 4) decision
        await te.execute_decision(
            signal=signal,
            symbol=symbol,
            price=price,
            size=None,
            interval=main_interval,
            training_mode=False,
            hybrid_mode=True,
            probs=probs,
            extra=extra,
        )

        # 5) kapanan trade buffer
        for tr in te.pop_closed_trades():
            pnl = float(tr.get("realized_pnl") or 0.0)
            pnl_total += pnl
            equity += pnl

            # backtest meta ekle
            tr2 = dict(tr)
            tr2.update({
                "bt_symbol": symbol,
                "bt_interval": main_interval,
                "bt_bar": i,
                "bt_time": t,
                "bt_signal": signal,
                "bt_price": price,
                "bt_p_used": float(p_single),
                "bt_p_single": float(p_single),
                "bt_p_1m": None,
                "bt_p_5m": None,
                "bt_p_15m": None,
                "bt_p_1h": None,
                "whale_dir": "none",
                "whale_score": 0.0,
                "whale_on": False,
                "whale_alignment": "no_whale",
                "whale_thr": None,
                "model_confidence_factor": 1.0,
                "ens_scale": None,
                "ens_notional": None,
                "bt_atr": None,
            })
            closed_trades.append(tr2)

        # dd
        if equity > peak:
            peak = equity
        dd_pct = 0.0 if peak <= 0 else (peak - equity) / peak * 100.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

        equity_rows.append({
            "bar": i,
            "time": t,
            "symbol": symbol,
            "interval": main_interval,
            "price": price,
            "signal": signal,
            "p_used": float(p_single),
            "p_single": float(p_single),
            "p_1m": None,
            "p_5m": None,
            "p_15m": None,
            "p_1h": None,
            "whale_dir": "none",
            "whale_score": 0.0,
            "whale_on": False,
            "whale_alignment": "no_whale",
            "whale_thr": None,
            "model_confidence_factor": 1.0,
            "ens_scale": None,
            "ens_notional": None,
            "atr": None,
            "equity": float(equity),
            "peak_equity": float(peak),
            "max_drawdown_pct": float(max_dd_pct),
            "pnl_total": float(pnl_total),
        })

    # EOB force close
    if te.has_open_position(symbol):
        last_price = float(df["close"].iloc[-1])
        te.close_position(symbol, last_price, reason="eob_force_close", interval=main_interval)
        for tr in te.pop_closed_trades():
            pnl = float(tr.get("realized_pnl") or 0.0)
            pnl_total += pnl
            equity += pnl
            tr2 = dict(tr)
            tr2.update({
                "bt_symbol": symbol,
                "bt_interval": main_interval,
                "bt_bar": n - 1,
                "bt_time": str(df["time"].iloc[-1]) if "time" in df.columns else "",
                "bt_signal": "EOB_CLOSE",
                "bt_price": last_price,
                "bt_p_used": None,
                "bt_p_single": None,
                "bt_p_1m": None,
                "bt_p_5m": None,
                "bt_p_15m": None,
                "bt_p_1h": None,
                "whale_dir": "none",
                "whale_score": 0.0,
                "whale_on": False,
                "whale_alignment": "no_whale",
                "whale_thr": None,
                "model_confidence_factor": 1.0,
                "ens_scale": None,
                "ens_notional": None,
                "bt_atr": None,
            })
            closed_trades.append(tr2)

    # summary
    n_trades = len(closed_trades)
    winrate = (sum(1 for t in closed_trades if float(t.get("realized_pnl") or 0.0) > 0.0) / n_trades * 100.0) if n_trades else 0.0

    bt_stats = {
        "symbol": symbol,
        "interval": main_interval,
        "starting_equity": float(start_equity),
        "ending_equity": float(equity),
        "pnl": float(pnl_total),
        "pnl_pct": float((equity - start_equity) / start_equity * 100.0) if start_equity else 0.0,
        "n_trades": int(n_trades),
        "winrate": float(winrate),
        "max_drawdown_pct": float(max_dd_pct),
        "tag": tag,
    }

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
