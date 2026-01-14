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
from core.position_manager import PositionManager
from core.trade_executor import TradeExecutor
from models.hybrid_inference import HybridModel
from data.anomaly_detection import AnomalyDetector

import importlib
import traceback


def _resolve_build_features(logger: logging.Logger):
    """
    build_features fonksiyonunu güvenli şekilde bulur:
      - önce features.fe_labels.build_features
      - sonra features.pipeline içindeki adaylar
    """
    # 1) fe_labels kesin var (rg çıktın bunu gösteriyor)
    try:
        m = importlib.import_module("features.fe_labels")
        fn = getattr(m, "build_features", None)
        if callable(fn):
            logger.info("[BT] Feature builder resolved: features.fe_labels.build_features")
            return fn
    except Exception as e:
        logger.error("[BT] features.fe_labels import hata: %s", e)
        logger.error("[BT] traceback:\n%s", traceback.format_exc())

    # 2) pipeline fallback
    mod_name = "features.pipeline"
    try:
        m = importlib.import_module(mod_name)
    except Exception as e:
        logger.error("[BT] %s import ederken hata: %s", mod_name, e)
        logger.error("[BT] traceback:\n%s", traceback.format_exc())
        return None

    candidates = [
        "build_features",
        "build_features_df",
        "make_features",
        "build_feature_pipeline",
        "build_features_online",
    ]

    for name in candidates:
        fn = getattr(m, name, None)
        if callable(fn):
            logger.info("[BT] Feature builder resolved: %s.%s", mod_name, name)
            return fn

    public = sorted([x for x in dir(m) if not x.startswith("_")])
    logger.error("[BT] %s içinde build_features benzeri fonksiyon bulunamadı.", mod_name)
    logger.error("[BT] %s public members: %s", mod_name, public)
    return None


system_logger = logging.getLogger("system")


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


def _compute_drawdown_pct(equity: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return float((peak - equity) / peak * 100.0)


class BTStats:
    def __init__(self, start_equity: float = 1000.0) -> None:
        self.trades = 0
        self.pnl_total = 0.0
        self.max_dd_pct = 0.0
        self.peak_equity = float(start_equity)
        self.last_equity = float(start_equity)
        self.start_equity = float(start_equity)

    def on_trade_close(self, realized_pnl: float) -> None:
        self.trades += 1
        self.pnl_total += float(realized_pnl)
        self.last_equity = self.start_equity + self.pnl_total
        if self.last_equity > self.peak_equity:
            self.peak_equity = self.last_equity
        dd = _compute_drawdown_pct(self.last_equity, self.peak_equity)
        if dd > self.max_dd_pct:
            self.max_dd_pct = dd

    def summary_dict(self) -> Dict[str, Any]:
        ending_equity = float(self.last_equity)
        pnl = float(self.pnl_total)
        pnl_pct = (pnl / self.start_equity * 100.0) if self.start_equity > 0 else 0.0
        return {
            "ending_equity": ending_equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "n_trades": int(self.trades),
            "trades": int(self.trades),
            "max_drawdown_pct": float(self.max_dd_pct),
            "peak_equity": float(self.peak_equity),
        }


def _save_backtest_csv(
    *,
    out_dir: Path,
    symbol: str,
    main_interval: str,
    tag: str,
    equity_rows: List[Dict[str, Any]],
    closed_trades: List[Dict[str, Any]],
    bt_stats: BTStats,
    system_logger: logging.Logger,
) -> None:
    equity_path = out_dir / f"equity_curve_{symbol}_{main_interval}_{tag}.csv"
    trades_path = out_dir / f"trades_{symbol}_{main_interval}_{tag}.csv"
    summary_path = out_dir / f"summary_{symbol}_{main_interval}_{tag}.csv"

    EQ_COLS = [
        "bar", "time", "symbol", "interval", "price", "signal",
        "p_used", "p_single",
        "whale_dir", "whale_score", "whale_on", "whale_alignment", "whale_thr",
        "model_confidence_factor", "ens_scale", "ens_notional", "atr",
        "equity", "peak_equity", "max_drawdown_pct", "pnl_total",
    ]

    TR_COLS = [
        "symbol","side","qty","entry_price","notional","interval","opened_at",
        "sl_price","tp_price","trailing_pct","atr_value","highest_price","lowest_price","meta",
        "closed_at","close_price","realized_pnl","close_reason",
        "bt_symbol","bt_interval","bt_bar","bt_time","bt_signal","bt_price","bt_p_used","bt_p_single","bt_atr",
        "whale_dir","whale_score","whale_on","whale_alignment","whale_thr",
        "model_confidence_factor","ens_scale","ens_notional",
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


class BTTradeExecutor(TradeExecutor):
    def __init__(self, *args: Any, closed_trades: List[Dict[str, Any]], bt_stats: BTStats, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._bt_closed_trades = closed_trades
        self._bt_stats = bt_stats
        self._bt_ctx: Dict[str, Any] = {}

    def set_bt_context(self, **ctx: Any) -> None:
        self._bt_ctx = dict(ctx)

    def _close_position(self, symbol: str, price: float, reason: str, interval: str) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        closed = super()._close_position(symbol, price, reason, interval)
        if not closed:
            return None

        try:
            ctx = self._bt_ctx or {}
            closed["bt_symbol"] = ctx.get("bt_symbol")
            closed["bt_interval"] = ctx.get("bt_interval")
            closed["bt_bar"] = ctx.get("bt_bar")
            closed["bt_time"] = ctx.get("bt_time")
            closed["bt_signal"] = ctx.get("bt_signal")
            closed["bt_price"] = ctx.get("bt_price")
            closed["bt_p_used"] = ctx.get("bt_p_used")
            closed["bt_p_single"] = ctx.get("bt_p_single")
            closed["bt_atr"] = ctx.get("bt_atr")
        except Exception:
            pass

        try:
            rp = float(closed.get("realized_pnl") or 0.0)
            self._bt_stats.on_trade_close(rp)
        except Exception:
            pass

        self._bt_closed_trades.append(closed)
        return closed


def _force_close_eob(
    *,
    executor: BTTradeExecutor,
    position_manager: PositionManager,
    symbol: str,
    last_price: float,
    interval: str,
    system_logger: logging.Logger,
) -> None:
    try:
        pos = position_manager.get_position(symbol)
        if not pos:
            return

        executor.set_bt_context(
            bt_symbol=symbol, bt_interval=interval, bt_bar="EOB", bt_time=None,
            bt_signal="FORCE_CLOSE", bt_price=last_price, bt_p_used=None, bt_p_single=None, bt_atr=None,
        )
        executor._close_position(symbol, float(last_price), reason="EOB_FORCE_CLOSE", interval=interval)
    except Exception as e:
        system_logger.error("[BT] force-close failed: %s", e, exc_info=True)


async def run_backtest() -> None:
    global system_logger
    system_logger = logging.getLogger("system")

    out_dir = Path(os.getenv("BT_OUT_DIR", "outputs"))
    symbol = os.getenv("BT_SYMBOL", os.getenv("SYMBOL", "BTCUSDT")).upper()
    main_interval = os.getenv("BT_MAIN_INTERVAL", os.getenv("INTERVAL", "5m"))
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    data_limit = int(float(os.getenv("BT_DATA_LIMIT", "2000")))
    warmup = int(float(os.getenv("BT_WARMUP_BARS", "200")))
    buy_thr = float(os.getenv("BT_BUY_THR", "0.55"))
    sell_thr = float(os.getenv("BT_SELL_THR", "0.45"))
    window = int(float(os.getenv("BT_MODEL_WINDOW", "200")))
    start_equity = float(os.getenv("BT_START_EQUITY", "1000.0"))

    progress_every = int(float(os.getenv("BT_PROGRESS_EVERY", "200")))

    equity_rows: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []
    bt_stats = BTStats(start_equity=start_equity)

    anomaly_detector = AnomalyDetector(logger=system_logger)

    risk = RiskManager(equity_start_of_day=start_equity, logger=system_logger)
    pm = PositionManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_db=int(os.getenv("REDIS_DB", "0")),
        redis_key_prefix=os.getenv("REDIS_POS_KEY_PREFIX", "positions"),
        enable_pg=str(os.getenv("ENABLE_PG_POS_LOG", "0")).lower() in ("1", "true", "yes", "on"),
        pg_dsn=os.getenv("PG_DSN"),
        logger=system_logger,
    )

    execu = BTTradeExecutor(
        client=None,
        risk_manager=risk,
        position_manager=pm,
        tg_bot=None,
        logger=system_logger,
        dry_run=True,
        base_order_notional=float(os.getenv("BT_BASE_ORDER_NOTIONAL", "50")),
        max_position_notional=float(os.getenv("BT_MAX_POSITION_NOTIONAL", "500")),
        max_leverage=float(os.getenv("BT_MAX_LEVERAGE", "3")),
        sl_pct=float(os.getenv("BT_SL_PCT", "0.01")),
        tp_pct=float(os.getenv("BT_TP_PCT", "0.02")),
        trailing_pct=float(os.getenv("BT_TRAILING_PCT", "0.01")),
        use_atr_sltp=str(os.getenv("BT_USE_ATR_SLTP", "1")).lower() in ("1", "true", "yes", "on"),
        atr_sl_mult=float(os.getenv("BT_ATR_SL_MULT", "1.5")),
        atr_tp_mult=float(os.getenv("BT_ATR_TP_MULT", "3.0")),
        closed_trades=closed_trades,
        bt_stats=bt_stats,
    )

    model = HybridModel(model_dir=os.getenv("MODELS_DIR", None), interval=main_interval, logger=system_logger)

    offline_dir = Path(os.getenv("OFFLINE_DIR", "data/offline_cache"))
    offline_main = offline_dir / f"{symbol}_{main_interval}_6m.csv"
    if not offline_main.exists():
        system_logger.error("[BT] Offline CSV bulunamadı: %s", str(offline_main))
        return

    df = pd.read_csv(offline_main)
    if data_limit > 0 and len(df) > data_limit:
        df = df.tail(data_limit).reset_index(drop=True)

    if "close" not in df.columns:
        system_logger.error("[BT] CSV içinde 'close' kolonu yok: %s", str(offline_main))
        return

    build_features_fn = _resolve_build_features(system_logger)
    if build_features_fn is None:
        system_logger.error("[BT] Feature builder bulunamadı.")
        return

    feat = build_features_fn(df, interval=main_interval)  # fe_labels build_features bunu destekliyor
    feat = anomaly_detector.filter_anomalies(feat, schema=None, context=f"heavy:{symbol}:{main_interval}")

    time_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else None)

    start_i = max(warmup, window)
    if len(feat) <= start_i + 1:
        system_logger.error("[BT] Veri yetersiz: rows=%d start_i=%d", len(feat), start_i)
        return

    peak = bt_stats.peak_equity

    system_logger.info("[BT] START loop | rows=%d start_i=%d end=%d window=%d", len(feat), start_i, len(feat)-1, window)

    for i in range(start_i, len(feat)):
        if progress_every > 0 and (i % progress_every == 0):
            system_logger.info("[BT] progress i=%d/%d equity=%.2f trades=%d", i, len(feat)-1, bt_stats.last_equity, bt_stats.trades)

        price = float(df["close"].iloc[i])
        Xw = feat.iloc[i - window : i].copy()

        p_arr, dbg = model.predict_proba(Xw)
        p_single = float(p_arr[-1]) if len(p_arr) else 0.5
        p_used = p_single

        if p_used >= buy_thr:
            sig = "BUY"
        elif p_used <= sell_thr:
            sig = "SELL"
        else:
            sig = "HOLD"

        bt_time = df[time_col].iloc[i] if time_col is not None else None

        execu.set_bt_context(
            bt_symbol=symbol, bt_interval=main_interval, bt_bar=i, bt_time=str(bt_time) if bt_time is not None else None,
            bt_signal=sig, bt_price=price, bt_p_used=p_used, bt_p_single=p_single,
            bt_atr=float(Xw["atr"].iloc[-1]) if "atr" in Xw.columns and len(Xw) else None,
        )

        try:
            risk.tick()
        except Exception:
            pass

        # trade decision
        call_exec = True
        # HOLD iken pozisyon yoksa executor çağırma (log/performans)
        if sig == "HOLD":
            try:
                call_exec = (pm.get_position(symbol) is not None)
            except Exception:
                call_exec = True
        
        if call_exec:
            await execu.execute_decision(
            signal=sig,
            symbol=symbol,
            price=price,
            size=None,
            interval=main_interval,
            training_mode=False,
            hybrid_mode=True,
            probs={"p_single": p_single, "p_used": p_used},
            extra={
                "p_buy_raw": p_single,
                "p_buy_source": "hybrid_single",
                "model_confidence_factor": float(dbg.get("p_hybrid_mean", 0.5) or 0.5),
                "atr": float(Xw["atr"].iloc[-1]) if "atr" in Xw.columns and len(Xw) else 0.0,
                "whale_dir": "none",
                "whale_score": 0.0,
            },
        )

        equity = bt_stats.last_equity
        if equity > peak:
            peak = equity
        dd_pct = _compute_drawdown_pct(equity, peak)

        equity_rows.append(
            {
                "bar": i,
                "time": str(bt_time) if bt_time is not None else None,
                "symbol": symbol,
                "interval": main_interval,
                "price": price,
                "signal": sig,
                "p_used": p_used,
                "p_single": p_single,
                "whale_dir": "none",
                "whale_score": 0.0,
                "whale_on": False,
                "whale_alignment": "none",
                "whale_thr": None,
                "model_confidence_factor": float(dbg.get("p_hybrid_mean", 0.5) or 0.5),
                "ens_scale": None,
                "ens_notional": None,
                "atr": float(Xw["atr"].iloc[-1]) if "atr" in Xw.columns and len(Xw) else None,
                "equity": equity,
                "peak_equity": peak,
                "max_drawdown_pct": dd_pct,
                "pnl_total": bt_stats.pnl_total,
            }
        )

    last_price = float(df["close"].iloc[-1])
    _force_close_eob(executor=execu, position_manager=pm, symbol=symbol, last_price=last_price, interval=main_interval, system_logger=system_logger)

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

    try:
        await run_backtest()
    except Exception as e:
        system_logger.error("[BT] FATAL: %s", e)
        system_logger.error("[BT] traceback:\n%s", traceback.format_exc())
        raise


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
