import os
import asyncio
import logging
from typing import Dict, Any, Optional

import pandas as pd

from config.load_env import load_environment_variables
from core.logger import setup_logger
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
from models.hybrid_inference import HybridModel
from core.hybrid_mtf import MultiTimeframeHybridEnsemble
from data.whale_detector import MultiTimeframeWhaleDetector
from data.anomaly_detection import AnomalyDetector

from dataclasses import dataclass, field

@dataclass
class BacktestStats:
    starting_equity: float = 1000.0
    equity: float = 1000.0
    peak_equity: float = 1000.0
    max_drawdown: float = 0.0
    n_trades: int = 0
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

        # Anomali filtresi (şimdilik global uyguluyoruz; ileride sadece geçmişe göre iyileştiririz)
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

    n_trades = 0

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
            # 3.7) Sinyal üretimi (main.py ile aynı mantık)
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
                        "[BT-TREND_FILTER] LONG -> HOLD "
                        "(p_1h=%.4f, p_15m=%.4f)",
                        p_1h,
                        p_15m,
                    )
                    signal = "hold"
                elif signal == "short" and not (p_1h < 0.4 and p_15m < 0.5):
                    system_logger.info(
                        "[BT-TREND_FILTER] SHORT -> HOLD "
                        "(p_1h=%.4f, p_15m=%.4f)",
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
            # 3.8) TradeExecutor ile simülasyon (DRY_RUN backtest)
            # ----------------------------------------------------------
            last_price = float(raw_by_interval[main_interval]["close"].iloc[i])

            await trade_executor.execute_decision(
                signal=signal,
                symbol=symbol,
                price=last_price,
                size=None,
                interval=main_interval,
                training_mode=False,      # backtest'te trade logic aktif
                hybrid_mode=HYBRID_MODE,
                probs=probs,
                extra=extra,
            )

            n_trades += 1

        except Exception as e:
            system_logger.exception("[BT-LOOP-ERROR] bar=%d hata=%s", i, e)

    # ------------------------------------------------------------------
    # Backtest özeti (RiskManager içinde varsa summary fonk. kullan)
    # ------------------------------------------------------------------
    system_logger.info("[BT] Backtest tamamlandı. Toplam bar=%d, işlenen bar=%d", min_len, min_len - warmup)
    system_logger.info("[BT] Toplam trade denemesi (execute_decision çağrısı)=%d", n_trades)

    # RiskManager'da summary / metrics varsa log'la
    if hasattr(risk_manager, "get_summary"):
        try:
            summary = risk_manager.get_summary()
            system_logger.info("[BT-RISK-SUMMARY] %s", summary)
        except Exception as e:
            system_logger.warning("[BT-RISK-SUMMARY] okunamadı: %s", e)


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
