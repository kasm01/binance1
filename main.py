import asyncio
import logging
import os
import signal
import json
import time
from typing import Any, Dict, Optional, List

import pandas as pd
import threading

from config.load_env import load_environment_variables
from config.settings import Settings

from config.credentials import Credentials
from core.logger import setup_logger
from core.binance_client import create_binance_client
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor

from data.anomaly_detection import AnomalyDetector
from data.online_learning import OnlineLearner
from data.whale_detector import MultiTimeframeWhaleDetector

from models.hybrid_inference import HybridModel, HybridMultiTFModel

from tg_bot.telegram_bot import TelegramBot

# WebSocket
from websocket.binance_ws import BinanceWS

from websocket.okx_ws import OKXWS
# NEW: p_buy stabilization + safe proba
from core.prob_stabilizer import ProbStabilizer
from core.model_utils import safe_p_buy

# TEK SÖZLEŞME: schema normalize
from features.schema import normalize_to_schema

# Model path contract
from app_paths import MODELS_DIR

from utils.auc_history import seed_auc_history_if_missing, append_auc_used_once_per_day

# ----------------------------------------------------------------------
# Global config / flags
# ----------------------------------------------------------------------
USE_TESTNET = getattr(Settings, "USE_TESTNET", True)
SYMBOL = getattr(Settings, "SYMBOL", "BTCUSDT")

system_logger: Optional[logging.Logger] = None

LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))

# Default (env parse ile override edilecek)
MTF_INTERVALS_DEFAULT = ["1m", "5m", "15m", "30m", "1h"]

current_ws = None


def get_bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def get_float_env(name: str, default: float) -> float:
    """ENV float okuyucu (hatalarda default döner)."""
    try:
        v = os.getenv(name, None)
        if v is None or str(v).strip() == "":
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


def parse_csv_env_list(name: str, default: List[str]) -> List[str]:
    """
    MTF_INTERVALS=1m,3m,5m,... gibi env'leri parse eder.
    Boş/None ise default döner.
    """
    v = os.getenv(name)
    if v is None:
        return list(default)
    s = str(v).strip()
    if not s:
        return list(default)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else list(default)


# ---- helpers: EMA adaptive alpha ----
def _compute_adaptive_alpha(atr_value: float) -> float:
    a_min = float(os.getenv("EMA_ALPHA_MIN", "0.05"))
    a_max = float(os.getenv("EMA_ALPHA_MAX", "0.45"))
    atr_ref = float(os.getenv("EMA_ATR_REF", "100.0"))

    try:
        atr = float(atr_value)
    except Exception:
        atr = 0.0

    if atr_ref <= 1e-9:
        atr_ref = 100.0

    x = atr / atr_ref
    x = max(0.0, min(2.0, x))

    alpha = a_min + (a_max - a_min) * (x / 2.0)
    alpha = max(a_min, min(a_max, alpha))
    return float(alpha)


# ----------------------------------------------------------------------
# NEW: Whale short-layer weights veto/boost + renorm + trend veto
# ----------------------------------------------------------------------
def _apply_whale_to_short_weights(
    mtf_debug: dict,
    whale_dir: str,
    whale_score: float,
    base_side: str,
    veto_thr: float = 0.70,
    boost_thr: float = 0.60,
    boost_max: float = 1.25,
):
    if not isinstance(mtf_debug, dict):
        return mtf_debug, {"whale_applied": False}

    intervals = mtf_debug.get("intervals_used") or []
    w_raw = mtf_debug.get("weights_raw") or []
    if not intervals or not w_raw or len(intervals) != len(w_raw):
        return mtf_debug, {"whale_applied": False, "reason": "no_weights"}

    short_set = {"1m", "3m", "5m"}

    if whale_dir not in ("long", "short") or whale_score <= 0:
        return mtf_debug, {"whale_applied": False}

    aligned = (whale_dir == base_side)
    opp = not aligned

    if opp and whale_score >= veto_thr:
        w2 = []
        for itv, w in zip(intervals, w_raw):
            if itv in short_set:
                w2.append(0.0)
            else:
                w2.append(float(w))
        mtf_debug["weights_raw"] = w2
        return mtf_debug, {
            "whale_applied": True,
            "mode": "VETO_SHORT_LAYERS",
            "whale_score": float(whale_score),
        }

    if aligned and whale_score >= boost_thr:
        mult = min(boost_max, 1.0 + (whale_score - boost_thr))
        w2 = []
        for itv, w in zip(intervals, w_raw):
            if itv in short_set:
                w2.append(float(w) * mult)
            else:
                w2.append(float(w))
        mtf_debug["weights_raw"] = w2
        return mtf_debug, {
            "whale_applied": True,
            "mode": "BOOST_SHORT_LAYERS",
            "mult": float(mult),
            "whale_score": float(whale_score),
        }

    return mtf_debug, {"whale_applied": False}


def _renorm_weights(mtf_debug: dict):
    w = mtf_debug.get("weights_raw") or []
    s = float(sum(w)) if w else 0.0
    if s > 0:
        mtf_debug["weights_norm"] = [float(x) / s for x in w]
    else:
        mtf_debug["weights_norm"] = [0.0 for _ in w]
    return mtf_debug


def _trend_veto(signal_side: str, p_by_itv: dict):
    """
    Eşikler ENV'den okunur, yoksa defaultlar korunur.
    ENV isimleri:
      TREND_VETO_LONG_15M, TREND_VETO_LONG_30M, TREND_VETO_LONG_1H
      TREND_VETO_SHORT_15M, TREND_VETO_SHORT_30M, TREND_VETO_SHORT_1H
    """
    veto_long_15m = get_float_env("TREND_VETO_LONG_15M", 0.48)
    veto_long_30m = get_float_env("TREND_VETO_LONG_30M", 0.48)
    veto_long_1h = get_float_env("TREND_VETO_LONG_1H", 0.47)

    veto_short_15m = get_float_env("TREND_VETO_SHORT_15M", 0.52)
    veto_short_30m = get_float_env("TREND_VETO_SHORT_30M", 0.52)
    veto_short_1h = get_float_env("TREND_VETO_SHORT_1H", 0.53)

    p15 = p_by_itv.get("15m")
    p30 = p_by_itv.get("30m")
    p1h = p_by_itv.get("1h")

    if signal_side == "long":
        if (p15 is not None and p15 < veto_long_15m) or \
           (p30 is not None and p30 < veto_long_30m) or \
           (p1h is not None and p1h < veto_long_1h):
            return True, f"TREND_VETO_LONG p15={p15} p30={p30} p1h={p1h}"
        return False, "OK"

    if signal_side == "short":
        if (p15 is not None and p15 > veto_short_15m) or \
           (p30 is not None and p30 > veto_short_30m) or \
           (p1h is not None and p1h > veto_short_1h):
            return True, f"TREND_VETO_SHORT p15={p15} p30={p30} p1h={p1h}"
        return False, "OK"

    return False, "HOLD"


def _extract_p_by_itv(mtf_debug: dict) -> Dict[str, Optional[float]]:
    p_by_itv: Dict[str, Optional[float]] = {}
    if not isinstance(mtf_debug, dict):
        return p_by_itv

    itvs = mtf_debug.get("intervals_used") or []
    per = mtf_debug.get("per_interval") if isinstance(mtf_debug.get("per_interval"), dict) else {}

    for itv in itvs:
        p_last = None
        try:
            if isinstance(per, dict) and isinstance(per.get(itv), dict):
                p_last = per.get(itv, {}).get("p_last")
            elif isinstance(mtf_debug.get(itv), dict):
                p_last = mtf_debug.get(itv, {}).get("p_last")
        except Exception:
            p_last = None

        if p_last is None:
            p_by_itv[itv] = None
        else:
            try:
                p_by_itv[itv] = float(p_last)
            except Exception:
                p_by_itv[itv] = None

    return p_by_itv


def _recompute_ensemble_from_debug(mtf_debug: dict) -> Optional[float]:
    if not isinstance(mtf_debug, dict):
        return None

    intervals = mtf_debug.get("intervals_used") or []
    w_norm = mtf_debug.get("weights_norm") or []
    if not intervals or not w_norm or len(intervals) != len(w_norm):
        return None

    p_by_itv = _extract_p_by_itv(mtf_debug)

    num = 0.0
    den = 0.0
    for itv, w in zip(intervals, w_norm):
        p = p_by_itv.get(itv)
        if p is None:
            continue
        try:
            wf = float(w)
            pf = float(p)
        except Exception:
            continue
        if wf <= 0:
            continue
        num += wf * pf
        den += wf

    if den <= 1e-12:
        return None

    return float(num / den)


# Global env flag’ler (async_main içinde tekrar güncellenecek)
BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET: Optional[str] = os.getenv("BINANCE_API_SECRET")

HYBRID_MODE: bool = get_bool_env("HYBRID_MODE", True)
TRAINING_MODE: bool = get_bool_env("TRAINING_MODE", False)
USE_MTF_ENS: bool = get_bool_env("USE_MTF_ENS", False)
DRY_RUN: bool = get_bool_env("DRY_RUN", True)


# ----------------------------------------------------------------------
# Basit feature engineering (stabilized)
# ----------------------------------------------------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stabilize edilmiş FE:
      - RAW 12 kolon yoksa ekler
      - numeric coercion + inf/nan cleanup
      - engineered kolonları garanti eder
      - dummy_extra garanti
    """
    df = raw_df.copy()

    # Binance kline columns guarantee
    raw_cols_12 = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for c in raw_cols_12:
        if c not in df.columns:
            df[c] = 0

    # open_time/close_time -> numeric seconds (float)
    for col in ["open_time", "close_time"]:
        if col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    dt = pd.to_datetime(df[col], unit="ms", utc=True)
                else:
                    dt = pd.to_datetime(df[col], utc=True)
                df[col] = (dt.astype("int64") / 1e9).astype(float)
            except Exception:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # coerce numeric
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_like = ["number_of_trades", "ignore"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ensure ignore exists
    if "ignore" not in df.columns:
        df["ignore"] = 0.0

    # engineered (numeric-safe)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    close = df["close"].astype(float)

    df["hl_range"] = (df["high"] - df["low"]).astype(float)
    df["oc_change"] = (df["close"] - df["open"]).astype(float)

    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)

    df["ma_5"] = close.rolling(5).mean()
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()

    # IMPORTANT: pipeline ile uyumlu olacak şekilde mean
    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    # cleanup
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.ffill().bfill().fillna(0.0)

    return df

# ----------------------------------------------------------------------
# ATR hesaplayıcı
# ----------------------------------------------------------------------
def compute_atr_from_klines(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr)


# ----------------------------------------------------------------------
# Binance Kline fetch helper
# ----------------------------------------------------------------------
def _fetch_klines_public_rest(symbol: str, interval: str, limit: int, logger: Optional[logging.Logger]) -> pd.DataFrame:
    import requests

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": int(limit)}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        klines = r.json()
    except Exception as e:
        if logger:
            logger.error("[DATA] Public REST klines fetch hatası: %s", e)
        raise

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)

    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "ignore" in df.columns:
        df["ignore"] = pd.to_numeric(df["ignore"], errors="coerce").fillna(0).astype(int)

    if logger:
        logger.info(
            "[DATA] LIVE(PUBLIC) REST kline çekildi. symbol=%s interval=%s shape=%s",
            symbol, interval, df.shape
        )
    return df


async def fetch_klines(client, symbol: str, interval: str, limit: int, logger: Optional[logging.Logger]) -> pd.DataFrame:
    data_mode = str(os.getenv("DATA_MODE", "OFFLINE")).upper().strip()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]

    def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Tek sözleşme:
        - kolon isimleri Binance kline schema (12) olacak
        - numeric coercion uygulanacak
        - inf/nan temizlenecek
        """
        if df is None or df.empty:
            return df

        # 12 kolon fazlasını kırp (Binance formatı)
        if df.shape[1] > 12:
            df = df.iloc[:, :12].copy()

        # Header yoksa (0..11) -> isimlendir
        if list(df.columns) == list(range(len(df.columns))) and df.shape[1] == 12:
            df = df.copy()
            df.columns = columns

        # Header varsa ama isimler farklıysa yine de mümkün olduğunca düzelt
        if not set(columns).issubset(set(df.columns)) and df.shape[1] == 12:
            df = df.copy()
            df.columns = columns

        # Tip dönüşümü
        for c in float_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # int kolonları gerçekten int yap (güvenli)
        for c in int_cols:
            if c in df.columns:
                try:
                    df[c] = df[c].astype(int)
                except Exception:
                    df[c] = df[c].fillna(0).astype(int)

        # inf/nan temizliği
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.ffill().bfill().fillna(0)

        return df

    # -------------------------
    # OFFLINE
    # -------------------------
    if data_mode != "LIVE":
        csv_path = f"data/offline_cache/{symbol}_{interval}_6m.csv"
        if not os.path.exists(csv_path):
            if logger:
                logger.error("[DATA] OFFLINE mod: CSV bulunamadı: %s", csv_path)
            raise RuntimeError(
                f"Offline kline dosyası yok: {csv_path}. "
                "DATA_MODE=LIVE yapın (public REST ile çeker) veya offline cache oluşturun."
            )

        # low_memory=False -> mixed dtype warning azalır
        df = pd.read_csv(csv_path, header=None, low_memory=False)

        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        df = _normalize_kline_df(df)

        if logger:
            logger.info("[DATA] OFFLINE mod: %s dosyasından kline yüklendi. shape=%s", csv_path, df.shape)
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                logger.warning("[DATA] OFFLINE normalize sonrası kolonlar eksik: cols=%s", list(df.columns))

        return df

    # -------------------------
    # LIVE
    # -------------------------
    last_err: Optional[Exception] = None

    if client is not None:
        try:
            klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=columns)
            df = _normalize_kline_df(df)

            if logger:
                logger.info(
                    "[DATA] LIVE(CLIENT) kline çekildi. symbol=%s interval=%s shape=%s",
                    symbol, interval, df.shape
                )
            return df

        except Exception as e:
            last_err = e
            if logger:
                logger.error("[DATA] LIVE(CLIENT) client.get_klines hatası: %s", e)

    try:
        df = _fetch_klines_public_rest(symbol, interval, limit, logger)
        df = _normalize_kline_df(df)
        return df
    except Exception as e:
        last_err = e
        if logger:
            logger.error("[DATA] LIVE(PUBLIC) REST de başarısız: %s", e)

    raise RuntimeError(f"DATA_MODE=LIVE fakat live fetch başarısız. last_err={last_err!r}")


# ----------------------------------------------------------------------
# Trading objeleri kurulum
# ----------------------------------------------------------------------
def create_trading_objects() -> Dict[str, Any]:
    global system_logger
    global BINANCE_API_KEY, BINANCE_API_SECRET
    global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS, DRY_RUN, USE_TESTNET

    symbol = os.getenv("SYMBOL") or getattr(Settings, "SYMBOL", None) or "BTCUSDT"
    interval = os.getenv("INTERVAL") or getattr(Settings, "INTERVAL", None) or "5m"

    client = create_binance_client(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        testnet=bool(USE_TESTNET),
        logger=system_logger,
        dry_run=bool(DRY_RUN),
    )

    daily_max_loss_usdt = float(os.getenv("DAILY_MAX_LOSS_USDT", "100"))
    daily_max_loss_pct = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.03"))
    max_consecutive_losses = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
    max_open_trades = int(os.getenv("MAX_OPEN_TRADES", "3"))
    equity_start_of_day = float(os.getenv("EQUITY_START_OF_DAY", "1000"))

    risk_manager = RiskManager(
        daily_max_loss_usdt=daily_max_loss_usdt,
        daily_max_loss_pct=daily_max_loss_pct,
        max_consecutive_losses=max_consecutive_losses,
        max_open_trades=max_open_trades,
        equity_start_of_day=equity_start_of_day,
        logger=system_logger,
    )


    tg_bot = None
    try:
        tg_bot = TelegramBot()

        dispatcher = getattr(tg_bot, "dispatcher", None)
        if dispatcher:
            # RiskManager enjeksiyonu (/risk)
            if hasattr(tg_bot, "set_risk_manager"):
                tg_bot.set_risk_manager(risk_manager)
            else:
                setattr(tg_bot, "risk_manager", risk_manager)

            if system_logger:
                system_logger.info("[MAIN] TelegramBot'a RiskManager enjekte edildi (/risk aktif).")

            # Polling'i arka planda başlat (main loop bloklanmasın)
            def _tg_polling_worker():
                try:
                    if system_logger:
                        system_logger.info("[MAIN] Telegram polling worker starting...")
                    tg_bot.start_polling()
                except Exception as e:
                    if system_logger:
                        system_logger.exception("[MAIN] Telegram polling worker crashed: %s", e)

            threading.Thread(
                target=_tg_polling_worker,
                name="telegram-polling",
                daemon=True,
            ).start()

        else:
            if system_logger:
                system_logger.warning(
                    "[MAIN] Telegram dispatcher yok. TELEGRAM_BOT_TOKEN ayarlı değilse komutlar çalışmaz."
                )

    except Exception as e:
        tg_bot = None
        if system_logger:
            system_logger.exception("[MAIN] TelegramBot init/enjeksiyon hata: %s", e)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", "bot:positions")

    # ---- PG env (robust) ----
    enable_pg = str(os.getenv("ENABLE_PG_POS_LOG", "0")).strip()
    enable_pg_flag = enable_pg.lower() not in ("0", "false", "no", "off", "")
    pg_dsn = (os.getenv("PG_DSN") or "").strip() if enable_pg_flag else None

    if system_logger:
        system_logger.info("[POS][ENV] ENABLE_PG_POS_LOG=%s enable_pg_flag=%s", enable_pg, enable_pg_flag)
        system_logger.info("[POS][ENV] PG_DSN=%s", (pg_dsn if pg_dsn else "(empty)"))

    if enable_pg_flag and not pg_dsn:
        if system_logger:
            system_logger.warning("[POS] ENABLE_PG_POS_LOG=1 ama PG_DSN boş. .env process'e yüklenmiyor olabilir.")

    position_manager = PositionManager(
        redis_url=redis_url,
        redis_key_prefix=redis_key_prefix,
        logger=system_logger,
        enable_pg=enable_pg_flag,
        pg_dsn=pg_dsn,
    )

    # MODELS_DIR contract
    hybrid_model = HybridModel(model_dir=MODELS_DIR, interval=interval, logger=system_logger)
    try:
        if hasattr(hybrid_model, "use_lstm_hybrid"):
            hybrid_model.use_lstm_hybrid = bool(HYBRID_MODE)
    except Exception:
        pass

    # mtf_intervals: env'den parse edelim (MTF_INTERVALS=1m,3m,5m,15m,30m,1h)
    mtf_intervals = parse_csv_env_list("MTF_INTERVALS", MTF_INTERVALS_DEFAULT)

    mtf_ensemble = None
    if USE_MTF_ENS:
        try:
            mtf_ensemble = HybridMultiTFModel(
                model_dir=MODELS_DIR,
                intervals=mtf_intervals,
                logger=system_logger,
            )

            if system_logger:
                system_logger.info("[MAIN] MTF ensemble aktif: intervals=%s", list(mtf_intervals))

            # --- AUC HISTORY BOOTSTRAP + DAILY APPEND (kalıcı) ---
            # Not: seed -> meta'ya bir kere yazar, daily append -> günde 1 kez yazar.
            try:
                seed_auc_history_if_missing(intervals=list(mtf_intervals), logger=system_logger)
                append_auc_used_once_per_day(intervals=list(mtf_intervals), logger=system_logger)
            except Exception as e:
                if system_logger:
                    system_logger.warning("[AUC-HIST] seed/daily append hata: %s", e)

        except Exception as e:
            mtf_ensemble = None
            if system_logger:
                system_logger.warning("[MAIN] HybridMultiTFModel init hata, MTF kapandı: %s", e)

    whale_detector = None
    try:
        whale_detector = MultiTimeframeWhaleDetector()
        if system_logger:
            system_logger.info("[WHALE] MultiTimeframeWhaleDetector init OK.")
    except Exception as e:
        whale_detector = None
        if system_logger:
            system_logger.warning("[WHALE] MultiTimeframeWhaleDetector init hata: %s", e)

    base_order_notional = float(os.getenv("BASE_ORDER_NOTIONAL", "50"))
    max_position_notional = float(os.getenv("MAX_POSITION_NOTIONAL", "500"))
    max_leverage = float(os.getenv("MAX_LEVERAGE", "3"))

    sl_pct = float(os.getenv("SL_PCT", "0.01"))
    tp_pct = float(os.getenv("TP_PCT", "0.02"))
    trailing_pct = float(os.getenv("TRAILING_PCT", "0.01"))

    use_atr_sltp = os.getenv("USE_ATR_SLTP", "true").lower() == "true"
    atr_sl_mult = float(os.getenv("ATR_SL_MULT", "1.5"))
    atr_tp_mult = float(os.getenv("ATR_TP_MULT", "3.0"))

    whale_risk_boost = float(os.getenv("WHALE_RISK_BOOST", "2.0"))

    trade_executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        position_manager=position_manager,
        logger=system_logger,
        dry_run=bool(DRY_RUN),
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
            tg_bot=tg_bot,
)


    # Telegram bot_data injections (commands rely on these)
    try:
        if tg_bot is not None and getattr(tg_bot, "dispatcher", None):
            tg_bot.dispatcher.bot_data["risk_manager"] = risk_manager  # type: ignore
            tg_bot.dispatcher.bot_data["position_manager"] = position_manager  # type: ignore
            tg_bot.dispatcher.bot_data["trade_executor"] = trade_executor  # type: ignore
            tg_bot.dispatcher.bot_data["symbol"] = symbol  # type: ignore
            tg_bot.dispatcher.bot_data["interval"] = interval  # type: ignore
            if system_logger:
                system_logger.info("[MAIN] Telegram bot_data injected: risk_manager/position_manager/trade_executor/symbol/interval")
    except Exception as e:
        if system_logger:
            system_logger.warning("[MAIN] Telegram bot_data inject error: %s", e)

    online_model = OnlineLearner(model_dir=MODELS_DIR, base_model_name="online_model", interval=interval)

    return {
        "symbol": symbol,
        "interval": interval,
        "client": client,
        "risk_manager": risk_manager,
        "position_manager": position_manager,
        "trade_executor": trade_executor,
        "hybrid_model": hybrid_model,
        "mtf_ensemble": mtf_ensemble,
        "mtf_intervals": mtf_intervals,
        "whale_detector": whale_detector,
        "tg_bot": tg_bot,
        "online_model": online_model,
    }


def _normalize_signal(sig: Any) -> str:
    s = str(sig).strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return "hold"


# ----------------------------------------------------------------------
# Ana trading loop
# ----------------------------------------------------------------------
async def bot_loop(objs: Dict[str, Any], prob_stab: ProbStabilizer) -> None:
    global system_logger

    tg_bot = objs.get("tg_bot")  # TelegramBot instance (opsiyonel)
    prev_signal_side: Optional[str] = None
    prev_notif_ts: float = 0.0

    client = objs["client"]
    trade_executor = objs["trade_executor"]
    whale_detector = objs.get("whale_detector")
    hybrid_model = objs["hybrid_model"]
    mtf_ensemble = objs.get("mtf_ensemble")

    symbol = objs.get("symbol", os.getenv("SYMBOL", getattr(Settings, "SYMBOL", "BTCUSDT")))
    interval = objs.get("interval", os.getenv("INTERVAL", "5m"))
    data_limit = int(os.getenv("DATA_LIMIT", "500"))

    mtf_intervals: List[str] = objs.get("mtf_intervals") or parse_csv_env_list("MTF_INTERVALS", MTF_INTERVALS_DEFAULT)

    # Debug throttling
    mtf_debug_every = int(os.getenv("MTF_DEBUG_EVERY", "1"))  # 1 => her loop, 5 => 5'te bir
    loop_i = 0

    if system_logger:
        system_logger.info(
            "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s, USE_MTF_ENS=%s)",
            symbol, interval, TRAINING_MODE, HYBRID_MODE, USE_MTF_ENS
        )

    anomaly_detector = AnomalyDetector(logger=system_logger)

    # ------------------------------------------------------------------
    # TEK SÖZLEŞME: meta schema loader + normalize
    # ------------------------------------------------------------------

    _schema_cache: Dict[str, Optional[list[str]]] = {}

    def _load_schema_from_disk(itv: str) -> Optional[list[str]]:
        if itv in _schema_cache:
            return _schema_cache[itv]

        try:
            p = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            sch = d.get("feature_schema")
            if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
                _schema_cache[itv] = sch
                return sch
        except Exception:
            pass

        _schema_cache[itv] = None
        return None

    def _fallback_schema_from_model() -> Optional[list[str]]:
        meta = getattr(hybrid_model, "meta", {}) or {}
        sch = meta.get("feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        return None

    def _schema_for(itv: str) -> Optional[list[str]]:
        return _load_schema_from_disk(itv) or _fallback_schema_from_model()

    def _normalize_feat_df(feat_df: pd.DataFrame, itv: str) -> pd.DataFrame:
        sch = _schema_for(itv)
        if not sch:
            if system_logger:
                system_logger.warning(
                    "[SCHEMA] No feature_schema for interval=%s (meta missing). Using raw features.",
                    itv,
                )
            return feat_df

        def _log_missing(missing_cols):
            if system_logger and missing_cols:
                system_logger.warning(
                    "[SCHEMA] interval=%s missing cols (filled=0): %s",
                    itv, missing_cols
                )

        # tek kilit: alias + missing fill + order + numeric cleanup
        return normalize_to_schema(feat_df, sch, log_missing=_log_missing)

    while True:
        loop_i += 1
        try:
            # 1) main interval raw
            raw_df = await fetch_klines(
                client=client,
                symbol=symbol,
                interval=interval,
                limit=data_limit,
                logger=system_logger,
            )

            # 2) features (engineer)
            feat_df = build_features(raw_df)

            # 3) schema normalize (TEK KİLİT)
            feat_df = _normalize_feat_df(feat_df, interval)

            # 4) anomaly filter (aynı schema ile)
            sch_main = _schema_for(interval)
            feat_df = anomaly_detector.filter_anomalies(feat_df, schema=sch_main)

            # 5) model input: DAİMA schema-sıralı DF
            X_live = feat_df.tail(500)

            # 6) Single-TF hibrit skor
            p_arr_single, debug_single = hybrid_model.predict_proba(X_live)
            p_single = float(p_arr_single[-1])

            meta = getattr(hybrid_model, "meta", {}) or {}
            model_conf_factor = float(meta.get("confidence_factor", 1.0) or 1.0)
            best_auc = float(meta.get("best_auc", 0.5) or 0.5)
            best_side = meta.get("best_side", "long")

            # 7) MTF verileri
            mtf_feats: Dict[str, pd.DataFrame] = {interval: feat_df}
            mtf_whale_raw: Dict[str, pd.DataFrame] = {interval: raw_df}

            if USE_MTF_ENS and mtf_ensemble is not None:
                for itv in mtf_intervals:
                    if itv == interval:
                        continue
                    try:
                        raw_df_itv = await fetch_klines(
                            client=client,
                            symbol=symbol,
                            interval=itv,
                            limit=data_limit,
                            logger=system_logger,
                        )

                        feat_df_itv = build_features(raw_df_itv)

                        # schema normalize (TEK KİLİT)
                        feat_df_itv = _normalize_feat_df(feat_df_itv, itv)

                        # anomaly filter (itv schema ile)
                        sch_itv = _schema_for(itv)
                        feat_df_itv = anomaly_detector.filter_anomalies(feat_df_itv, schema=sch_itv)

                        mtf_feats[itv] = feat_df_itv
                        mtf_whale_raw[itv] = raw_df_itv

                    except Exception as e:
                        if system_logger:
                            system_logger.warning("[MTF] %s interval'i hazırlanırken hata: %s", itv, e)

            # 8) MTF ensemble skoru
            p_used = p_single
            mtf_debug: Optional[Dict[str, Any]] = None

            if USE_MTF_ENS and mtf_ensemble is not None:
                try:
                    X_by_interval: Dict[str, pd.DataFrame] = {}
                    for itv, df_itv in mtf_feats.items():
                        sch_itv = _schema_for(itv)
                        if not sch_itv:
                            continue
                        X_by_interval[itv] = normalize_to_schema(df_itv, sch_itv).tail(500)

                    if X_by_interval:
                        p_ens, mtf_debug = mtf_ensemble.predict_proba_multi(
                            X_dict=X_by_interval,
                            standardize_auc_key="auc_used",
                            standardize_overwrite=False,
                        )
                        p_used = float(p_ens)

                        # --- MTF DEBUG SUMMARY (compact) --- (doğru yer: tahmin sonrası)
                        try:
                            if (mtf_debug_every <= 1) or (loop_i % mtf_debug_every == 0):
                                if USE_MTF_ENS and (mtf_ensemble is not None) and isinstance(mtf_debug, dict):
                                    per = mtf_debug.get("per_interval", {}) or {}
                                    wnorm = mtf_debug.get("weights_norm", []) or []
                                    used = mtf_debug.get("intervals_used", []) or []

                                    rows = []
                                    for itv2 in used:
                                        d = per.get(itv2, {}) or {}
                                        p_last = d.get("p_last", None)
                                        auc_used = d.get("auc_used", None)
                                        if (p_last is not None) and (auc_used is not None):
                                            rows.append(f"{itv2}:p={float(p_last):.4f},auc={float(auc_used):.4f}")
                                        else:
                                            rows.append(f"{itv2}:p={p_last},auc={auc_used}")

                                    if system_logger:
                                        system_logger.info(
                                            "[MTF] ensemble_p=%.4f | weights_norm=%s | %s",
                                            float(mtf_debug.get("ensemble_p", p_used if p_used is not None else 0.5)),
                                            str(wnorm),
                                            " | ".join(rows),
                                        )
                        except Exception as _e:
                            if system_logger:
                                system_logger.debug("[MTF] debug summary failed: %s", _e)

                except Exception as e:
                    if system_logger:
                        system_logger.warning("[MTF] Ensemble hesaplanırken hata: %s", e)
                    p_used = p_single
                    mtf_debug = None

            # 9) Whale meta
            whale_meta: Dict[str, Any] = {"direction": "none", "score": 0.0, "per_tf": {}}
            whale_dir = "none"
            whale_score = 0.0

            if whale_detector is not None:
                try:
                    if hasattr(whale_detector, "analyze_multiple_timeframes"):
                        whale_signals = whale_detector.analyze_multiple_timeframes(mtf_whale_raw)
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
                    elif hasattr(whale_detector, "from_klines"):
                        ws = whale_detector.from_klines(raw_df)
                        whale_meta.update(
                            {
                                "direction": ws.direction,
                                "score": ws.score,
                                "reason": ws.reason,
                                "meta": ws.meta,
                            }
                        )

                    whale_dir = str(whale_meta.get("direction", "none") or "none")
                    whale_score = float(whale_meta.get("score", 0.0) or 0.0)

                except Exception as e:
                    if system_logger:
                        system_logger.warning("[WHALE] MTF whale hesaplanırken hata: %s", e)

            # 10) ATR
            atr_period = int(os.getenv("ATR_PERIOD", "14"))
            atr_value = compute_atr_from_klines(raw_df, period=atr_period)

            # probs + extra
            probs: Dict[str, Any] = {
                "p_used": p_used,
                "p_single": p_single,
                "p_sgd_mean": float(debug_single.get("p_sgd_mean", 0.0)) if isinstance(debug_single, dict) else 0.0,
                "p_lstm_mean": float(debug_single.get("p_lstm_mean", 0.5)) if isinstance(debug_single, dict) else 0.5,
            }

            extra: Dict[str, Any] = {
                "model_confidence_factor": model_conf_factor,
                "best_auc": best_auc,
                "best_side": best_side,
                "mtf_debug": mtf_debug,
                "whale_meta": whale_meta,
                "atr": atr_value,
                "p_buy_source": (debug_single.get("mode") if isinstance(debug_single, dict) else "unknown"),
                "whale_dir": whale_dir,
                "whale_score": float(whale_score),
            }

            # Signal decision
            signal_side: Optional[str] = None
            use_ema_signal = get_bool_env("USE_PBUY_STABILIZER_SIGNAL", False)
            ema_whale_only = get_bool_env("EMA_WHALE_ONLY", False)
            ema_whale_thr = float(os.getenv("EMA_WHALE_THR", "0.50"))

            try:
                alpha_dyn = _compute_adaptive_alpha(atr_value)
                if hasattr(prob_stab, "set_alpha"):
                    prob_stab.set_alpha(alpha_dyn)
                else:
                    prob_stab.alpha = float(alpha_dyn)

                whale_on = (whale_dir != "none") and (whale_score >= ema_whale_thr)
                extra["ema_whale_on"] = bool(whale_on)

                p_buy_raw = float(p_used)
                p_buy_ema = prob_stab.update(p_buy_raw)
                sig_from_ema = _normalize_signal(prob_stab.signal(p_buy_ema))

                probs["p_buy_raw"] = float(p_buy_raw)
                probs["p_buy_ema"] = float(p_buy_ema)

                extra["p_buy_raw"] = float(p_buy_raw)
                extra["p_buy_ema"] = float(p_buy_ema)
                extra["ema_alpha"] = float(getattr(prob_stab, "alpha", alpha_dyn))
                extra["ema_whale_only"] = bool(ema_whale_only)
                extra["ema_whale_thr"] = float(ema_whale_thr)

                if use_ema_signal and (not ema_whale_only or whale_on):
                    signal_side = sig_from_ema

            except Exception as e:
                if system_logger:
                    system_logger.warning("[ONLINE] EMA block hata: %s", e)
                signal_side = None

            if signal_side is None:
                long_thr = float(os.getenv("LONG_THRESHOLD", "0.60"))
                short_thr = float(os.getenv("SHORT_THRESHOLD", "0.40"))
                if p_used >= long_thr:
                    signal_side = "long"
                elif p_used <= short_thr:
                    signal_side = "short"
                else:
                    signal_side = "hold"
                extra["signal_source"] = "HYBRID"
            else:
                extra["signal_source"] = "EMA"

            # Trend veto (ENV configurable)
            try:
                p_by_itv = _extract_p_by_itv(mtf_debug) if isinstance(mtf_debug, dict) else {}
                vetoed, veto_reason = _trend_veto(signal_side, p_by_itv)
                if vetoed:
                    signal_side = "hold"
                    extra.setdefault("veto_flags", []).append(veto_reason)
            except Exception as e:
                if system_logger:
                    system_logger.warning("[VETO] trend_veto error: %s", e)

            # Whale weights veto/boost -> renorm -> recompute ensemble
            try:
                if USE_MTF_ENS and isinstance(mtf_debug, dict) and signal_side in ("long", "short"):
                    mtf_debug, whale_w_dbg = _apply_whale_to_short_weights(
                        mtf_debug=mtf_debug,
                        whale_dir=whale_dir,
                        whale_score=float(whale_score),
                        base_side=str(signal_side),
                        veto_thr=float(os.getenv("WHALE_SHORT_VETO_THR", "0.70")),
                        boost_thr=float(os.getenv("WHALE_SHORT_BOOST_THR", "0.60")),
                        boost_max=float(os.getenv("WHALE_SHORT_BOOST_MAX", "1.25")),
                    )

                    if isinstance(whale_w_dbg, dict) and whale_w_dbg.get("whale_applied"):
                        extra["whale_weight_adjust"] = whale_w_dbg
                        mtf_debug = _renorm_weights(mtf_debug)
                        new_p = _recompute_ensemble_from_debug(mtf_debug)
                        if new_p is not None:
                            mtf_debug["ensemble_p"] = float(new_p)
                            p_used = float(new_p)
                            probs["p_used"] = float(p_used)
                    extra["mtf_debug"] = mtf_debug
            except Exception as e:
                if system_logger:
                    system_logger.warning("[WHALE][WEIGHT] apply/recompute error: %s", e)

            # Journal extras
            try:
                if USE_MTF_ENS and isinstance(mtf_debug, dict) and (mtf_debug.get("ensemble_p") is not None):
                    extra["ensemble_p"] = float(mtf_debug.get("ensemble_p"))
                    extra["p_buy_source"] = "ensemble_p"
                    extra["signal_source"] = "MTF"
            except Exception:
                pass

            last_price = float(raw_df["close"].iloc[-1])

            # ----------------------------
            # Telegram: status_snapshot + signal notifications
            # ----------------------------
            try:
                enable_tg_snapshot = str(os.getenv("TELEGRAM_ENABLE_SNAPSHOT", "1")).strip().lower() not in ("0", "false", "no", "off", "")
                enable_tg_notify = str(os.getenv("TELEGRAM_NOTIFY_SIGNALS", "1")).strip().lower() not in ("0", "false", "no", "off", "")
                notify_cooldown_s = float(os.getenv("TELEGRAM_NOTIFY_COOLDOWN_S", "10"))

                if tg_bot is not None and getattr(tg_bot, "dispatcher", None) and enable_tg_snapshot:
                    # AUC kısa sözlük
                    aucs: Dict[str, Any] = {}
                    try:
                        if isinstance(mtf_debug, dict):
                            per = mtf_debug.get("per_interval", {}) or {}
                            for itv2, d in per.items():
                                if isinstance(d, dict) and ("auc_used" in d):
                                    aucs[str(itv2)] = d.get("auc_used")
                    except Exception:
                        aucs = {}

                    # Snapshot (commands.py /status bunu okuyacak)
                    tg_bot.dispatcher.bot_data["status_snapshot"] = {
                        "symbol": symbol,
                        "signal": str(signal_side).upper() if signal_side else "N/A",
                        "ensemble_p": float(p_used) if p_used is not None else None,
                        "intervals": list(mtf_intervals) if isinstance(mtf_intervals, list) else [],
                        "aucs": aucs,
                        "last_price": float(last_price),
                        "why": str(extra.get("signal_source", "")) if isinstance(extra, dict) else "",
                    }

                # Sinyal değişim bildirimi (rate-limited)
                now_ts = time.time()
                if enable_tg_notify and tg_bot is not None:
                    sig_now = str(signal_side or "hold").lower()
                    sig_prev = str(prev_signal_side or "hold").lower()

                    if sig_now != sig_prev and (now_ts - prev_notif_ts) >= notify_cooldown_s:
                        emoji = {"long": "✅", "short": "🟣", "hold": "⏸"}.get(sig_now, "❔")
                        src = (extra.get("signal_source") if isinstance(extra, dict) else None) or "?"
                        msg = (
                            f"{emoji} *Signal Update*\n"
                            f"• *Symbol:* `{symbol}`\n"
                            f"• *From → To:* `{sig_prev.upper()}` → `{sig_now.upper()}`\n"
                            f"• *p_used:* `{float(p_used):.4f}`\n"
                            f"• *Price:* `{float(last_price):.4f}`\n"
                            f"• *Source:* `{src}`"
                        )
                        # TelegramBot.send_message CHAT_ID ister (TELEGRAM_CHAT_ID)
                        tg_bot.send_message(msg)
                        prev_notif_ts = now_ts
                        prev_signal_side = sig_now

                    # ilk turda prev'i set et (bildirim spam olmasın)
                    if prev_signal_side is None:
                        prev_signal_side = sig_now

            except Exception as _tg_e:
                if system_logger:
                    system_logger.debug("[TG] snapshot/notify block hata: %s", _tg_e)

            await trade_executor.execute_decision(
                signal=signal_side,
                symbol=symbol,
                price=last_price,
                size=None,
                interval=interval,
                training_mode=TRAINING_MODE,
                hybrid_mode=HYBRID_MODE,
                probs=probs,
                extra=extra,
            )

        except Exception as e:
            if system_logger:
                system_logger.exception("[LOOP ERROR] %s", e)
            else:
                print("[LOOP ERROR]", e)

        await asyncio.sleep(LOOP_SLEEP_SECONDS)


# ----------------------------------------------------------------------
# Async main
# ----------------------------------------------------------------------
async def async_main() -> None:
    global system_logger
    global BINANCE_API_KEY, BINANCE_API_SECRET
    global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS, DRY_RUN

    load_environment_variables()
    setup_logger()
    system_logger = logging.getLogger("system")


    # Credentials (env presence) - standard log
    try:
        Credentials.log_missing(prefix='[ENV]')
    except Exception:
        pass
    # --- ENV FORCE LOAD (.env -> os.environ) ---
    # Bazı projelerde load_environment_variables() .env'yi Settings'e alıp
    # process env'e basmıyor olabilir. Bu blok PG_DSN/ENABLE_PG_POS_LOG gibi
    # kritik değişkenleri garanti eder.
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (os.getenv(k) is None):
                    os.environ[k] = v
    except Exception:
        pass

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

    HYBRID_MODE = get_bool_env("HYBRID_MODE", HYBRID_MODE)
    TRAINING_MODE = get_bool_env("TRAINING_MODE", TRAINING_MODE)
    USE_MTF_ENS = get_bool_env("USE_MTF_ENS", USE_MTF_ENS)
    DRY_RUN = get_bool_env("DRY_RUN", DRY_RUN)

    if system_logger:
        system_logger.info("[MAIN] TRAINING_MODE=%s", TRAINING_MODE)
        system_logger.info("[MAIN] HYBRID_MODE=%s", HYBRID_MODE)
        system_logger.info("[MAIN] USE_MTF_ENS=%s", USE_MTF_ENS)
        system_logger.info("[MAIN] DRY_RUN=%s", DRY_RUN)

        # Bu loglar, env artık okunuyor mu diye kanıt
        system_logger.info("[ENV] ENABLE_PG_POS_LOG=%s", os.getenv("ENABLE_PG_POS_LOG"))
        system_logger.info("[ENV] PG_DSN=%s", os.getenv("PG_DSN"))
        system_logger.info("[ENV] MTF_INTERVALS=%s", os.getenv("MTF_INTERVALS"))

    prob_stab = ProbStabilizer(
        alpha=float(os.getenv("PBUY_EMA_ALPHA", "0.20")),
        buy_thr=float(os.getenv("BUY_THRESHOLD", "0.60")),
        sell_thr=float(os.getenv("SELL_THRESHOLD", "0.40")),
    )

    enable_ws = get_bool_env("ENABLE_WS", False)
    if enable_ws:
        symbol = os.getenv("SYMBOL", getattr(Settings, "SYMBOL", "BTCUSDT"))
        ws = BinanceWS(symbol=symbol)
        ws.run_background()
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=true -> websocket started. symbol=%s", symbol)
    else:
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=false -> websocket disabled.")

    trading_objects = create_trading_objects()

    try:
        await bot_loop(trading_objects, prob_stab)
    finally:
        pass


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, loop.stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(async_main())
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()


if __name__ == "__main__":
    main()
