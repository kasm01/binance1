import os
import asyncio
import logging
from typing import Dict, Any, Optional

import pandas as pd

from datetime import datetime
from pathlib import Path

from config.load_env import load_environment_variables
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from models.hybrid_inference import HybridModel
from core.hybrid_mtf import MultiTimeframeHybridEnsemble
from data.whale_detector import MultiTimeframeWhaleDetector
from data.anomaly_detection import AnomalyDetector

from dataclasses import dataclass


@dataclass
class BacktestStats:
    starting_equity: float = 1000.0
    equity: float = 1000.0
    peak_equity: float = 1000.0
    max_drawdown: float = 0.0
    n_trades: int = 0          # kapanan trade sayısı (pnl_delta != 0 olduğunda artar)
    n_wins: int = 0
    n_losses: int = 0

    def on_pnl_delta(self, delta: float) -> None:
        if delta == 0:
            return

        self.equity += delta
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = 0.0
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self.n_trades += 1
        if delta > 0:
            self.n_wins += 1
        elif delta < 0:
            self.n_losses += 1

    def summary_dict(self) -> Dict[str, float]:
        pnl = self.equity - self.starting_equity
        pnl_pct = pnl / self.starting_equity * 100 if self.starting_equity != 0 else 0.0
        winrate = self.n_wins / self.n_trades * 100 if self.n_trades > 0 else 0.0

        return {
            "starting_equity": self.starting_equity,
            "ending_equity": self.equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "winrate": winrate,
            "max_drawdown_pct": self.max_drawdown * 100.0,
        }


# main.py içindeki helper'ları aynen kullanıyoruz
from main import build_features, compute_atr_from_klines, build_labels


system_logger: Optional[logging.Logger] = None

# MTF interval listesi (backtest için sabit)
MTF_INTERVALS = ["1m", "5m", "15m", "1h"]


def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


# ----------------------------------------------------------------------
# OFFLINE kline yükleyici (sadece CSV'den)
# ----------------------------------------------------------------------
def load_offline_klines(
    symbol: str,
    interval: str,
    limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Offline kline dosyası bulunamadı: {csv_path}")

    df = pd.read_csv(csv_path)
    if limit is not None and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)

    if logger:
        logger.info(
            "[BT-DATA] OFFLINE CSV yüklendi: %s interval=%s shape=%s",
            csv_path,
            interval,
            df.shape,
        )
    return df


def _safe_bar_time_iso(raw_df: pd.DataFrame, idx: int) -> str:
    """
    open_time ms epoch ise iso string üretir; değilse stringe çevirir.
    """
    try:
        if "open_time" not in raw_df.columns:
            return str(idx)
        ot = raw_df["open_time"].iloc[idx]
        if isinstance(ot, (int, float)) and pd.notna(ot):
            # Binance open_time genelde ms
            return datetime.utcfromtimestamp(float(ot) / 1000.0).isoformat()
        return str(ot)
    except Exception:
        return str(idx)


# ----------------------------------------------------------------------
# Backtest ana fonksiyonu
# ----------------------------------------------------------------------
async def run_backtest() -> None:
    global system_logger

    symbol = os.getenv("BT_SYMBOL", "BTCUSDT")
    main_interval = os.getenv("BT_MAIN_INTERVAL", "5m")
    data_limit = int(os.getenv("BT_DATA_LIMIT", "500"))

    HYBRID_MODE = get_bool_env("HYBRID_MODE", True)
    USE_MTF_ENS = get_bool_env("USE_MTF_ENS", True)

    if main_interval not in MTF_INTERVALS:
        raise ValueError(
            f"BT_MAIN_INTERVAL={main_interval} MTF_INTERVALS içinde olmalı: {MTF_INTERVALS}"
        )

    system_logger.info(
        "[BT] Backtest başlıyor | symbol=%s main_interval=%s HYBRID_MODE=%s USE_MTF_ENS=%s",
        symbol,
        main_interval,
        HYBRID_MODE,
        USE_MTF_ENS,
    )

    # --------------------------------------------------------------
    # 1) Tüm TF'ler için offline kline + feature + anomaly filter
    # --------------------------------------------------------------
    anomaly_detector = AnomalyDetector(logger=system_logger)

    raw_by_interval: Dict[str, pd.DataFrame] = {}
    feat_by_interval: Dict[str, pd.DataFrame] = {}

    # Backward-compat alias map
    alias_map = {
        "taker_buy_base_volume": "taker_buy_base_asset_volume",
        "taker_buy_quote_volume": "taker_buy_quote_asset_volume",
    }

    for itv in MTF_INTERVALS:
        raw_df = load_offline_klines(
            symbol=symbol,
            interval=itv,
            limit=data_limit,
            logger=system_logger,
        )
        feat_df = build_features(raw_df)

        # alias fix
        for old_col, new_col in alias_map.items():
            if old_col not in feat_df.columns and new_col in feat_df.columns:
                feat_df[old_col] = feat_df[new_col]

        # Anomali filtresi
        feat_df = anomaly_detector.filter_anomalies(feat_df)

        raw_by_interval[itv] = raw_df.reset_index(drop=True)
        feat_by_interval[itv] = feat_df.reset_index(drop=True)

    # Tüm TF'lerin ortak minimum uzunluğu
    min_len = min(len(df) for df in feat_by_interval.values())
    if min_len < 200:
        raise RuntimeError(f"Backtest için yeterli veri yok (min_len={min_len})")

    # --------------------------------------------------------------
    # 2) Modeller: her interval için HybridModel + MTF ensemble
    # --------------------------------------------------------------
    mtf_models: Dict[str, HybridModel] = {}
    main_model: Optional[HybridModel] = None

    for itv in MTF_INTERVALS:
        model = HybridModel(model_dir="models", interval=itv, logger=system_logger)
        try:
            if hasattr(model, "use_lstm_hybrid"):
                model.use_lstm_hybrid = HYBRID_MODE
        except Exception:
            pass

        mtf_models[itv] = model
        system_logger.info("[BT] HybridModel yüklendi | interval=%s", itv)

        if itv == main_interval:
            main_model = model

    if main_model is None:
        raise RuntimeError("Main interval için HybridModel bulunamadı.")

    mtf_ensemble = MultiTimeframeHybridEnsemble(
        models_by_interval=mtf_models,
    )

    if USE_MTF_ENS:
        system_logger.info(
            "[BT] MultiTimeframeHybridEnsemble aktif: intervals=%s",
            list(mtf_models.keys()),
        )

    # Whale detector
    whale_detector: Optional[MultiTimeframeWhaleDetector] = None
    try:
        whale_detector = MultiTimeframeWhaleDetector()
        system_logger.info("[BT-WHALE] MultiTimeframeWhaleDetector init OK")
    except Exception as e:
        system_logger.warning("[BT-WHALE] init hata: %s (whale kapalı)", e)
        whale_detector = None

    # --------------------------------------------------------------
    # 3) RiskManager + TradeExecutor (sadece local state, PG/Redis yok)
    # --------------------------------------------------------------
    daily_max_loss_usdt = float(os.getenv("BT_DAILY_MAX_LOSS_USDT", "100"))
    daily_max_loss_pct = float(os.getenv("BT_DAILY_MAX_LOSS_PCT", "0.03"))
    max_consecutive_losses = int(os.getenv("BT_MAX_CONSECUTIVE_LOSSES", "5"))
    max_open_trades = int(os.getenv("BT_MAX_OPEN_TRADES", "3"))
    equity_start_of_day = float(os.getenv("BT_EQUITY_START_OF_DAY", "1000"))

    risk_manager = RiskManager(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_loss_pct=daily_max_loss_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_open_trades=max_open_trades,
        equity_start_of_day=equity_start_of_day,
        logger=system_logger,
    )

    base_order_notional = float(os.getenv("BT_BASE_ORDER_NOTIONAL", "50"))
    max_position_notional = float(os.getenv("BT_MAX_POSITION_NOTIONAL", "500"))
    max_leverage = float(os.getenv("BT_MAX_LEVERAGE", "3"))

    sl_pct = float(os.getenv("BT_SL_PCT", "0.01"))
    tp_pct = float(os.getenv("BT_TP_PCT", "0.02"))
    trailing_pct = float(os.getenv("BT_TRAILING_PCT", "0.01"))

    use_atr_sltp = get_bool_env("BT_USE_ATR_SLTP", True)
    atr_sl_mult = float(os.getenv("BT_ATR_SL_MULT", "1.5"))
    atr_tp_mult = float(os.getenv("BT_ATR_TP_MULT", "3.0"))

    whale_risk_boost = float(os.getenv("BT_WHALE_RISK_BOOST", "2.0"))

    trade_executor = TradeExecutor(
        client=None,                     # gerçek borsa yok
        risk_manager=risk_manager,
        position_manager=None,           # sadece local dict
        logger=system_logger,
        dry_run=True,                    # backtest → DRY_RUN=True
        base_order_notional=base_order_notional,
        max_position_notional=max_position_notional,
        max_leverage=max_leverage,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        trailing_pct=trailing_pct,
        use_atr_sltp=use_atr_sltp,
        atr_sl_mult=atr_sl_mult,
        atr_tp_mult=atr_tp_mult,
        whale_risk_boost=whale_risk_boost,
    )

    # --------------------------------------------------------------
    # CSV export: kapanan işlemleri yakala + equity curve satırları
    # --------------------------------------------------------------
    closed_trades: list[Dict[str, Any]] = []
    equity_rows: list[Dict[str, Any]] = []

    if hasattr(trade_executor, "_close_position"):
        try:
            _orig_close_position = trade_executor._close_position  # type: ignore[attr-defined]

            def _close_position_capture(symbol: str, price: float, reason: str, interval: str):
                closed = _orig_close_position(symbol=symbol, price=price, reason=reason, interval=interval)
                if isinstance(closed, dict):
                    closed["bt_symbol"] = symbol
                    closed["bt_interval"] = interval
                    closed_trades.append(closed)
                return closed

            trade_executor._close_position = _close_position_capture  # type: ignore[method-assign]
            system_logger.info("[BT] TradeExecutor._close_position patch OK (trades.csv aktif).")
        except Exception as e:
            system_logger.warning("[BT] TradeExecutor patch başarısız (trades.csv eksik olabilir): %s", e)
    else:
        system_logger.warning("[BT] TradeExecutor içinde _close_position yok. trades.csv yakalama pasif.")

    # --- Backtest istatistiklerini başlat ---
    start_eq = equity_start_of_day
    bt_stats = BacktestStats(
        starting_equity=start_eq,
        equity=start_eq,
        peak_equity=start_eq,
    )
    system_logger.info(
        "[BT] Stats init | starting_equity=%.2f",
        bt_stats.starting_equity,
    )

    # Feature kolon listesi (main.py ile aynı)
    feature_cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
        "hl_range",
        "oc_change",
        "return_1",
        "return_3",
        "return_5",
        "ma_5",
        "ma_10",
        "ma_20",
        "vol_10",
        "dummy_extra",
    ]

    # Backtest başlangıç index'i (biraz ısınma için)
    warmup = int(os.getenv("BT_WARMUP_BARS", "200"))
    if warmup >= min_len:
        warmup = min_len // 3

    system_logger.info("[BT] Başlangıç index (warmup)=%d, toplam bar=%d", warmup, min_len)

    # ------------------------------------------------------------------
    # Backtest loop: her bar için sinyal + TradeExecutor
    # ------------------------------------------------------------------
    atr_period = int(os.getenv("ATR_PERIOD", "14"))
    long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
    short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))

    n_exec_calls = 0  # execute_decision çağrı sayısı

    for i in range(warmup, min_len - 1):
        try:
            # ----------------------------------------------------------
            # 3.1) Her interval için X_live hazırlanıyor
            # ----------------------------------------------------------
            X_by_interval: Dict[str, pd.DataFrame] = {}
            mtf_whale_raw: Dict[str, pd.DataFrame] = {}

            for itv in MTF_INTERVALS:
                feat_df_itv = feat_by_interval[itv].iloc[: i + 1]
                raw_df_itv = raw_by_interval[itv].iloc[: i + 1]

                cols_itv = [c for c in feature_cols if c in feat_df_itv.columns]
                if not cols_itv:
                    system_logger.warning(
                        "[BT-MTF] Interval=%s için kullanılabilir feature yok, skip.",
                        itv,
                    )
                    continue

                X_by_interval[itv] = feat_df_itv[cols_itv].tail(500)
                mtf_whale_raw[itv] = raw_df_itv.tail(500)

            if main_interval not in X_by_interval:
                system_logger.warning(
                    "[BT] main_interval=%s X_by_interval içinde yok, bar=%d skip.",
                    main_interval,
                    i,
                )
                continue

            # ----------------------------------------------------------
            # 3.2) Single-TF (main interval) hibrit skor
            # ----------------------------------------------------------
            X_main = X_by_interval[main_interval]
            p_arr_single, debug_single = main_model.predict_proba(X_main)
            p_single = float(p_arr_single[-1])

            meta = getattr(main_model, "meta", {}) or {}
            model_conf_factor = float(meta.get("confidence_factor", 1.0) or 1.0)
            best_auc = float(meta.get("best_auc", 0.5) or 0.5)
            best_side = meta.get("best_side", "long")

            # ----------------------------------------------------------
            # 3.3) MTF ensemble
            # ----------------------------------------------------------
            p_used = p_single
            mtf_debug: Optional[Dict[str, Any]] = None
            p_1m = p_5m = p_15m = p_1h = None

            if USE_MTF_ENS:
                try:
                    ensemble_p, mtf_debug = mtf_ensemble.predict_mtf(X_by_interval)
                    p_used = float(ensemble_p)

                    per_int = (
                        mtf_debug.get("per_interval", {})
                        if isinstance(mtf_debug, dict)
                        else {}
                    )

                    def get_p(itv: str) -> Optional[float]:
                        try:
                            v = per_int.get(itv, {}).get("p_last")
                            return float(v) if v is not None else None
                        except Exception:
                            return None

                    p_1m = get_p("1m")
                    p_5m = get_p("5m")
                    p_15m = get_p("15m")
                    p_1h = get_p("1h")

                except Exception as e:
                    system_logger.warning(
                        "[BT-MTF] Ensemble hesaplanırken hata: %s (p_used=p_single)", e
                    )
                    p_used = p_single
                    mtf_debug = None

            # ----------------------------------------------------------
            # 3.4) Whale MTF analizi
            # ----------------------------------------------------------
            whale_meta: Dict[str, Any] = {
                "direction": "none",
                "score": 0.0,
                "per_tf": {},
            }

            if whale_detector is not None:
                try:
                    if hasattr(whale_detector, "analyze_multiple_timeframes"):
                        whale_signals = whale_detector.analyze_multiple_timeframes(
                            mtf_whale_raw
                        )
                        best_tf = None
                        best_score = 0.0

                        for tf, sig in whale_signals.items():
                            whale_meta["per_tf"][tf] = {
                                "direction": sig.direction,
                                "score": sig.score,
                                "reason": sig.reason,
                            }
                            if sig.direction != "none" and sig.score > best_score:
                                best_score = sig.score
                                best_tf = tf

                        if best_tf is not None:
                            best_sig = whale_signals[best_tf]
                            whale_meta.update(
                                {
                                    "direction": best_sig.direction,
                                    "score": best_sig.score,
                                    "best_tf": best_tf,
                                    "best_reason": best_sig.reason,
                                }
                            )
                except Exception as e:
                    system_logger.warning(
                        "[BT-WHALE] MTF whale hesaplanırken hata: %s", e
                    )

            # ----------------------------------------------------------
            # 3.5) ATR (main interval raw)
            # ----------------------------------------------------------
            raw_main = raw_by_interval[main_interval].iloc[: i + 1]
            atr_value = compute_atr_from_klines(
                raw_main.tail(atr_period + 2), period=atr_period
            )

            # ----------------------------------------------------------
            # 3.6) Probs + extra meta
            # ----------------------------------------------------------
            probs = {
                "p_used": p_used,
                "p_single": p_single,
                "p_sgd_mean": float(debug_single.get("p_sgd_mean", 0.0)),
                "p_lstm_mean": float(debug_single.get("p_lstm_mean", 0.5)),
            }

            extra: Dict[str, Any] = {
                "model_confidence_factor": model_conf_factor,
                "best_auc": best_auc,
                "best_side": best_side,
                "mtf_debug": mtf_debug,
                "whale_meta": whale_meta,
                "atr": atr_value,
            }

            system_logger.info(
                "[BT-HYBRID] bar=%d mode=%s n_samples=%d n_features=%d "
                "p_sgd_mean=%.4f p_lstm_mean=%.4f p_hybrid_mean=%.4f "
                "best_auc=%.4f best_side=%s",
                i,
                debug_single.get("mode", "unknown"),
                len(X_main),
                X_main.shape[1],
                float(debug_single.get("p_sgd_mean", 0.0)),
                float(debug_single.get("p_lstm_mean", 0.5)),
                float(debug_single.get("p_hybrid_mean", p_used)),
                best_auc,
                best_side,
            )

            # ----------------------------------------------------------
            # 3.7) Sinyal üretimi
            # ----------------------------------------------------------
            if p_used >= long_thr:
                signal = "long"
            elif p_used <= short_thr:
                signal = "short"
            else:
                signal = "hold"

            # --- Trend filtresi (1h/15m hard veto) ---
            if p_1h is not None and p_15m is not None:
                if signal == "long" and not (p_1h > 0.6 and p_15m > 0.5):
                    system_logger.info(
                        "[BT-TREND_FILTER] LONG -> HOLD (p_1h=%.4f, p_15m=%.4f)",
                        p_1h,
                        p_15m,
                    )
                    signal = "hold"
                elif signal == "short" and not (p_1h < 0.4 and p_15m < 0.5):
                    system_logger.info(
                        "[BT-TREND_FILTER] SHORT -> HOLD (p_1h=%.4f, p_15m=%.4f)",
                        p_1h,
                        p_15m,
                    )
                    signal = "hold"

            # --- 1m mikro filtre ---
            micro_conf_scale = 1.0
            if signal == "long" and isinstance(p_1m, float) and p_1m < 0.30:
                micro_conf_scale = 0.7
            elif signal == "short" and isinstance(p_1m, float) and p_1m > 0.70:
                micro_conf_scale = 0.7

            effective_model_conf = float(model_conf_factor) * micro_conf_scale

            whale_dir = whale_meta.get("direction") if isinstance(whale_meta, dict) else None
            whale_score = whale_meta.get("score") if isinstance(whale_meta, dict) else None

            system_logger.info(
                "[BT-SIGNAL] bar=%d p_used=%.4f long_thr=%.3f short_thr=%.3f "
                "signal=%s model_conf=%.3f effective_conf=%.3f "
                "p_1m=%s p_5m=%s p_15m=%s p_1h=%s whale_dir=%s whale_score=%s",
                i,
                p_used,
                long_thr,
                short_thr,
                signal,
                float(model_conf_factor),
                effective_model_conf,
                f"{p_1m:.4f}" if isinstance(p_1m, float) else "None",
                f"{p_5m:.4f}" if isinstance(p_5m, float) else "None",
                f"{p_15m:.4f}" if isinstance(p_15m, float) else "None",
                f"{p_1h:.4f}" if isinstance(p_1h, float) else "None",
                whale_dir if whale_dir is not None else "None",
                f"{whale_score:.3f}" if isinstance(whale_score, (int, float)) else "None",
            )

            extra["model_confidence_factor"] = effective_model_conf

            # Label diagnostic (sadece log için)
            last_label = build_labels(feat_by_interval[main_interval]).iloc[i]
            system_logger.info(
                "[BT-LABEL] bar=%d label(horizon=1)=%s (1=up,0=down)",
                i,
                last_label,
            )

            # ----------------------------------------------------------
            # 3.8) TradeExecutor ile simülasyon + PnL delta → BacktestStats
            # ----------------------------------------------------------
            prev_pnl = 0.0
            try:
                prev_pnl = float(getattr(risk_manager, "daily_realized_pnl", 0.0) or 0.0)
            except Exception:
                prev_pnl = 0.0

            last_price = float(raw_by_interval[main_interval]["close"].iloc[i])

            await trade_executor.execute_decision(
                signal=signal,
                symbol=symbol,
                price=last_price,
                size=None,
                interval=main_interval,
                training_mode=False,
                hybrid_mode=HYBRID_MODE,
                probs=probs,
                extra=extra,
            )
            n_exec_calls += 1

            try:
                new_pnl = float(getattr(risk_manager, "daily_realized_pnl", prev_pnl) or prev_pnl)
            except Exception:
                new_pnl = prev_pnl

            pnl_delta = new_pnl - prev_pnl
            if pnl_delta != 0.0:
                bt_stats.on_pnl_delta(pnl_delta)
                system_logger.info(
                    "[BT-PNL] bar=%d pnl_delta=%.4f equity=%.2f max_dd=%.2f%% "
                    "closed_trades=%d wins=%d losses=%d",
                    i,
                    pnl_delta,
                    bt_stats.equity,
                    bt_stats.max_drawdown * 100.0,
                    bt_stats.n_trades,
                    bt_stats.n_wins,
                    bt_stats.n_losses,
                )

            # ----------------------------------------------------------
            # 3.9) Equity curve satırı (her bar)
            # ----------------------------------------------------------
            bar_time = _safe_bar_time_iso(raw_by_interval[main_interval], i)

            equity_rows.append(
                {
                    "bar": i,
                    "time": bar_time,
                    "symbol": symbol,
                    "interval": main_interval,
                    "price": last_price,
                    "signal": signal,
                    "p_used": float(p_used),
                    "p_single": float(p_single),
                    "p_1m": p_1m,
                    "p_5m": p_5m,
                    "p_15m": p_15m,
                    "p_1h": p_1h,
                    "whale_dir": whale_dir,
                    "whale_score": whale_score,
                    "atr": float(atr_value),
                    "equity": float(bt_stats.equity),
                    "peak_equity": float(bt_stats.peak_equity),
                    "max_drawdown_pct": float(bt_stats.max_drawdown) * 100.0,
                    "pnl_total": float(bt_stats.equity - bt_stats.starting_equity),
                }
            )

        except Exception as e:
            system_logger.exception("[BT-LOOP-ERROR] bar=%d hata=%s", i, e)

    # ------------------------------------------------------------------
    # Backtest özeti
    # ------------------------------------------------------------------
    system_logger.info(
        "[BT] Backtest tamamlandı. Toplam bar=%d, işlenen bar=%d",
        min_len,
        min_len - warmup,
    )
    system_logger.info(
        "[BT] Toplam execute_decision çağrısı=%d (trade denemesi)",
        n_exec_calls,
    )

    if hasattr(risk_manager, "get_summary"):
        try:
            summary = risk_manager.get_summary()
            system_logger.info("[BT-RISK-SUMMARY] %s", summary)
        except Exception as e:
            system_logger.warning("[BT-RISK-SUMMARY] okunamadı: %s", e)

    bt_summary = bt_stats.summary_dict()
    system_logger.info(
        "[BT-RESULT] starting_equity=%.2f ending_equity=%.2f pnl=%.2f (%.2f%%) "
        "n_trades=%d wins=%d losses=%d winrate=%.2f%% max_dd=%.2f%%",
        bt_summary["starting_equity"],
        bt_summary["ending_equity"],
        bt_summary["pnl"],
        bt_summary["pnl_pct"],
        bt_summary["n_trades"],
        bt_summary["n_wins"],
        bt_summary["n_losses"],
        bt_summary["winrate"],
        bt_summary["max_drawdown_pct"],
    )

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    out_dir = Path(os.getenv("BT_OUT_DIR", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    equity_path = out_dir / f"equity_curve_{symbol}_{main_interval}_{run_tag}.csv"
    trades_path = out_dir / f"trades_{symbol}_{main_interval}_{run_tag}.csv"
    summary_path = out_dir / f"summary_{symbol}_{main_interval}_{run_tag}.csv"

    try:
        pd.DataFrame(equity_rows).to_csv(equity_path, index=False)
        system_logger.info("[BT-CSV] equity_curve yazıldı: %s (rows=%d)", str(equity_path), len(equity_rows))
    except Exception as e:
        system_logger.warning("[BT-CSV] equity_curve yazılamadı: %s", e)

    try:
        pd.DataFrame(closed_trades).to_csv(trades_path, index=False)
        system_logger.info("[BT-CSV] trades yazıldı: %s (trades=%d)", str(trades_path), len(closed_trades))
    except Exception as e:
        system_logger.warning("[BT-CSV] trades yazılamadı: %s", e)

    try:
        pd.DataFrame([bt_summary]).to_csv(summary_path, index=False)
        system_logger.info("[BT-CSV] summary yazıldı: %s", str(summary_path))
    except Exception as e:
        system_logger.warning("[BT-CSV] summary yazılamadı: %s", e)


# ----------------------------------------------------------------------
# Async main + entry point
# ----------------------------------------------------------------------
async def async_main() -> None:
    global system_logger

    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")

    if system_logger is None:
        print("[BT-MAIN] system_logger init edilemedi!")
        return

    await run_backtest()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
