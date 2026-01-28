
def _mask_env_presence(keys: list[str]) -> dict:
    import os
    out = {}
    for k in keys:
        v = os.getenv(k)
        out[k] = "SET" if (v is not None and str(v).strip() != "") else "MISSING"
    return out

def _log_secret_env_presence(logger):
    # Değerleri ASLA basmıyoruz. Sadece SET/MISSING.
    keys = [
        # Telegram
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_CHAT_IDS",        # Binance / OKX
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "OKX_API_KEY",
        "OKX_API_SECRET",
        "OKX_PASSPHRASE",

        # Infra
        "REDIS_PASSWORD",
        "PG_DSN",

        # Web3 / Data APIs
        "ETH_API_KEY",
        "ALCHEMY_ETH_API_KEY",
        "INFURA_API_KEY",
        "POLYGON_API_KEY",
        "ARBI_API_KEY",
        "THE_GRAPH_API_KEY",
        "GRAPH_API_KEY",
        "COINGLASS_API_KEY",
        "BSCSCAN_API_KEY",
        "ETHERSCAN_API_KEY",
        "CRYPTOQUANT_API_KEY",
        "COINMARKETCAP_API_KEY",
        "SANTIMENT_API_KEY",
    ]
    st = _mask_env_presence(keys)
    # tek satır, okunur, güvenli
    logger.info("[ENV][SECRETS] %s", st)

import asyncio
import logging
import os

# TensorFlow / XLA C++ loglarını azalt (TF import edilmeden ÖNCE olmalı)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import signal
import json
import time
import random
from typing import Any, Dict, Optional, List

import pandas as pd
import threading
from functools import lru_cache

from config.load_env import load_environment_variables
from config.settings import Settings

from config.credentials import Credentials
from core.logger import setup_logger
from core.binance_client import create_binance_client
from core.position_manager import PositionManager
from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor

from core.market_meta_builder import MarketMetaBuilder
from data.anomaly_detection import AnomalyDetector
from data.whale_detector import MultiTimeframeWhaleDetector

from models.hybrid_inference import HybridMultiTFModel
from models.model_registry import ModelRegistry

from tg_bot.telegram_bot import TelegramBot

# WebSocket
from websocket.binance_ws import BinanceWS
from websocket.okx_ws import OKXWS
from websocket.binance_depth_ws import BinanceDepthWS

# p_buy stabilization + safe proba
from core.prob_stabilizer import ProbStabilizer

# TEK SÖZLEŞME: schema normalize
from features.schema import normalize_to_schema

# Model path contract
from app_paths import MODELS_DIR

from utils.auc_history import seed_auc_history_if_missing, append_auc_used_once_per_hour

# Optional: project light scanner (varsa)
try:
    from scanner.light_scanner import LightScanner  # type: ignore
except Exception:
    LightScanner = None  # type: ignore


# ----------------------------------------------------------------------
# Global config / flags
# ----------------------------------------------------------------------
USE_TESTNET = getattr(Settings, "USE_TESTNET", True)
SYMBOL = getattr(Settings, "SYMBOL", "BTCUSDT")

system_logger: Optional[logging.Logger] = None

LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "60"))

# Default (env parse ile override edilecek)
MTF_INTERVALS_DEFAULT = ["1m", "3m", "5m", "15m", "30m", "1h"]


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


def get_int_env(name: str, default: int) -> int:
    try:
        v = os.getenv(name, None)
        if v is None or str(v).strip() == "":
            return int(default)
        return int(float(str(v).strip()))
    except Exception:
        return int(default)


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


def parse_symbols_env() -> List[str]:
    default_symbol = os.getenv("SYMBOL") or getattr(Settings, "SYMBOL", "BTCUSDT")
    syms = parse_csv_env_list("SYMBOLS", [default_symbol])
    out: List[str] = []
    for s in syms:
        s2 = str(s).strip().upper()
        if s2:
            out.append(s2)
    return out or [str(default_symbol).strip().upper()]


def _sleep_jitter(base_s: float, jitter_ratio: float = 0.15) -> float:
    """
    base_s üzerine küçük random jitter ekle (thundering herd azaltır).
    """
    if base_s <= 0:
        return 0.0
    j = base_s * float(jitter_ratio)
    return max(0.0, base_s + random.uniform(-j, j))


# ----------------------------------------------------------------------
# Latency profiling helper
# ----------------------------------------------------------------------
class _StepTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.steps: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.steps[name] = float(now - self.last)
        self.last = now

    def total(self) -> float:
        return float(time.perf_counter() - self.t0)


# ----------------------------------------------------------------------
# Adaptive scan (volatility-based) helpers
# ----------------------------------------------------------------------
def _volatility_score(df: pd.DataFrame) -> float:
    """
    Hızlı volatilite skoru (scan için):
      - (high-low)/close rolling mean
      - abs return_1 rolling mean
    """
    try:
        if df is None or df.empty or len(df) < 40:
            return 0.0
        x = df.copy()
        for c in ("high", "low", "close"):
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        if not {"high", "low", "close"}.issubset(x.columns):
            return 0.0
        close = x["close"].astype(float).replace(0.0, pd.NA).ffill().bfill().fillna(1.0)
        hl = (x["high"].astype(float) - x["low"].astype(float)).abs()
        hl_pct = (hl / (close + 1e-9)).rolling(20).mean().iloc[-1]
        r1 = close.pct_change(1).abs().rolling(20).mean().iloc[-1]
        s = float(hl_pct + 0.8 * r1)
        if not (s == s):
            return 0.0
        return max(0.0, min(1.0, s * 10.0))  # normalize-ish
    except Exception:
        return 0.0


def _adaptive_scan_params(base_every: float, base_topk: int, vol_score: float) -> Dict[str, Any]:
    """
    vol_score ~ [0..1]
      - yüksek vol: daha sık scan + daha yüksek topk
      - düşük vol: daha yavaş scan + düşük topk
    """
    every_min = get_float_env("SCAN_EVERY_MIN_SEC", 3.0)
    every_max = get_float_env("SCAN_EVERY_MAX_SEC", 30.0)
    topk_min = get_int_env("SCAN_TOPK_MIN", 2)
    topk_max = get_int_env("SCAN_TOPK_MAX", 12)

    v = max(0.0, min(1.0, float(vol_score)))

    # high vol => smaller every
    every = float(base_every * (1.0 - 0.6 * v))
    every = max(every_min, min(every_max, every))

    topk = int(round(base_topk + (topk_max - base_topk) * v))
    topk = max(topk_min, min(topk_max, topk))

    return {"scan_every": every, "topk": topk, "vol_score": v}


def _quick_vol_regime(df: pd.DataFrame, window: int = 60) -> float:
    """
    Çok hızlı vol rejimi proxy:
      vol_regime ~= mean(TrueRange)/close
    FAST scan için candidates adaptif ayarına yardımcı.
    """
    try:
        if df is None or df.empty or len(df) < max(10, window):
            return 0.0
        d = df.tail(window).copy()
        for c in ("high", "low", "close"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0).astype(float)
        if not {"high", "low", "close"}.issubset(d.columns):
            return 0.0
        high = d["high"].astype(float)
        low = d["low"].astype(float)
        close = d["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = float(tr.mean())
        last_close = float(close.iloc[-1]) if len(close) else 0.0
        if last_close <= 0:
            return 0.0
        return float(atr / (abs(last_close) + 1e-9))
    except Exception:
        return 0.0
def _light_score_from_klines(raw_df: pd.DataFrame) -> float:
    """
    Light scanner score (hızlı):
      - volume spike (z-score)
      - range/price (volatility proxy)
      - kısa momentum
    """
    try:
        if raw_df is None or raw_df.empty or len(raw_df) < 50:
            return 0.0

        df = raw_df.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

        if "volume" not in df.columns or "close" not in df.columns:
            return 0.0

        vol = df["volume"].values
        if len(vol) < 30:
            return 0.0

        v_last = float(vol[-1])
        v_mean = float(pd.Series(vol).rolling(20).mean().iloc[-1])
        v_std = float(pd.Series(vol).rolling(20).std().iloc[-1])
        v_z = (v_last - v_mean) / (v_std + 1e-9)

        if not {"high", "low"}.issubset(df.columns):
            return 0.0

        hl = (df["high"] - df["low"]).values
        hl_mean = float(pd.Series(hl).rolling(20).mean().iloc[-1])

        close = df["close"].values
        if len(close) < 6:
            return 0.0
        r1 = (float(close[-1]) / (float(close[-2]) + 1e-9) - 1.0)
        r5 = (float(close[-1]) / (float(close[-6]) + 1e-9) - 1.0)

        score = 0.0
        score += max(0.0, v_z) * 2.0
        score += abs(r1) * 50.0
        score += abs(r5) * 20.0
        score += (hl_mean / (float(close[-1]) + 1e-9)) * 100.0
        return float(max(0.0, score))
    except Exception:
        return 0.0


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
    df = raw_df.copy()

    raw_cols_12 = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
    ]
    for c in raw_cols_12:
        if c not in df.columns:
            df[c] = 0

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

    float_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
    ]
    int_like = ["number_of_trades", "ignore"]

    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ignore" not in df.columns:
        df["ignore"] = 0.0

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

    df["vol_10"] = df["volume"].astype(float).rolling(10).mean()

    if "dummy_extra" not in df.columns:
        df["dummy_extra"] = 0.0

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.ffill().bfill().fillna(0.0)

    return df


def compute_atr_from_klines(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    if not {"high", "low", "close"}.issubset(df.columns):
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
# OFFLINE LRU cache helpers (raw + normalized_tail)
# ----------------------------------------------------------------------
_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]
_KLINE_FLOAT_COLS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume",
]
_KLINE_INT_COLS = ["open_time", "close_time", "number_of_trades", "ignore"]


def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    12 kolon kline DF normalize:
      - header yoksa kolonları set eder
      - numeric cast + inf/nan temizliği
    """
    if df is None or df.empty:
        return df

    if df.shape[1] > 12:
        df = df.iloc[:, :12].copy()

    # header yoksa
    if list(df.columns) == list(range(len(df.columns))) and df.shape[1] == 12:
        df = df.copy()
        df.columns = _KLINE_COLUMNS

    # farklı header ama 12 kolon ise yine set et
    if not set(_KLINE_COLUMNS).issubset(set(df.columns)) and df.shape[1] == 12:
        df = df.copy()
        df.columns = _KLINE_COLUMNS

    for c in _KLINE_FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in _KLINE_INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in _KLINE_INT_COLS:
        if c in df.columns:
            try:
                df[c] = df[c].astype(int)
            except Exception:
                df[c] = df[c].fillna(0).astype(int)

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.ffill().bfill().fillna(0)

    return df


@lru_cache(maxsize=128)
def _read_offline_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    """
    Raw CSV cache.
    Key: (path, mtime)
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, header=None, low_memory=False)


@lru_cache(maxsize=256)
def _offline_klines_tail_cached(path: str, mtime: float, limit: int) -> pd.DataFrame:
    """
    Normalized + tail(limit) cache.
    Key: (path, mtime, limit)
    """
    df_raw = _read_offline_csv_cached(path, mtime)
    df = _normalize_kline_df(df_raw.copy())

    lim = int(limit)
    if lim > 0 and len(df) > lim:
        df = df.tail(lim).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


# ----------------------------------------------------------------------
# LIVE fallback: Public REST
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

    df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
    df = _normalize_kline_df(df)

    if logger:
        logger.info("[DATA] LIVE(PUBLIC) REST kline çekildi. symbol=%s interval=%s shape=%s", symbol, interval, df.shape)
    return df


# ----------------------------------------------------------------------
# fetch_klines (OFFLINE cached + LIVE client/rest)
# ----------------------------------------------------------------------
async def fetch_klines(client, symbol: str, interval: str, limit: int, logger: Optional[logging.Logger]) -> pd.DataFrame:
    data_mode = str(os.getenv("DATA_MODE", "OFFLINE")).upper().strip()

    # -------------------------
    # OFFLINE (LRU cached)
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

        try:
            mtime = float(os.path.getmtime(csv_path))
        except Exception:
            mtime = 0.0

        # ✅ normalize + tail(limit) cached
        df = _offline_klines_tail_cached(csv_path, mtime, int(limit)).copy()

        if logger:
            logger.info("[DATA] OFFLINE mod: %s dosyasından kline yüklendi(CACHED_TAIL). shape=%s", csv_path, df.shape)
            if not {"open", "high", "low", "close"}.issubset(df.columns):
                logger.warning("[DATA] OFFLINE normalize sonrası kolonlar eksik: cols=%s", list(df.columns))

        return df

    # -------------------------
    # LIVE (client -> fallback rest)
    # -------------------------
    last_err: Optional[Exception] = None

    if client is not None:
        try:
            klines = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
            df = _normalize_kline_df(df)

            if logger:
                logger.info("[DATA] LIVE(CLIENT) kline çekildi. symbol=%s interval=%s shape=%s", symbol, interval, df.shape)
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
# Shutdown manager (ws stop + tg stop + http close + pm/te close)
# ----------------------------------------------------------------------
class ShutdownManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._done = False

        self._tg_bot: Optional[TelegramBot] = None
        self._binance_ws: Optional[BinanceWS] = None
        self._okx_ws: Optional[OKXWS] = None
        self._client: Any = None
        self._position_manager: Any = None
        self._trade_executor: Any = None

    def register(
        self,
        *,
        tg_bot: Optional[TelegramBot] = None,
        binance_ws: Optional[BinanceWS] = None,
        okx_ws: Optional[OKXWS] = None,
        depth_ws: Any = None,
        client: Any = None,
        position_manager: Any = None,
        trade_executor: Any = None,
    ) -> None:
        with self._lock:
            if tg_bot is not None:
                self._tg_bot = tg_bot
            if binance_ws is not None:
                self._binance_ws = binance_ws
            if okx_ws is not None:
                self._okx_ws = okx_ws
            if depth_ws is not None:
                self._depth_ws = depth_ws
            if client is not None:
                self._client = client
            if position_manager is not None:
                self._position_manager = position_manager
            if trade_executor is not None:
                self._trade_executor = trade_executor

    async def shutdown(self, reason: str = "unknown") -> None:
        log = system_logger

        # idempotent
        with self._lock:
            if self._done:
                return
            self._done = True

            tg_bot = self._tg_bot
            bws = self._binance_ws
            okx = self._okx_ws
            dws = self._depth_ws
            client = self._client
            pm = self._position_manager
            te = self._trade_executor

        if log:
            log.info("[SHUTDOWN] starting cleanup | reason=%s", reason)

        # 1) WS stop (Binance)
        try:
            if bws is not None and hasattr(bws, "stop"):
                bws.stop(timeout=5.0)  # type: ignore
                if log:
                    log.info("[SHUTDOWN] BinanceWS stopped.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] BinanceWS stop failed: %s", e)

        # 2) WS stop (OKX)
        try:
            if okx is not None and hasattr(okx, "stop"):
                okx.stop(timeout=5.0)  # type: ignore
                if log:
                    log.info("[SHUTDOWN] OKXWS stopped.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] OKXWS stop failed: %s", e)

        # 2.5) Depth WS stop (Binance Futures depth)
        try:
            if dws is not None and hasattr(dws, "stop"):
                dws.stop(timeout=5.0)  # type: ignore
                if log:
                    log.info("[SHUTDOWN] DepthWS stopped.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] DepthWS stop failed: %s", e)

        # 3) Telegram stop
        try:
            if tg_bot is not None and hasattr(tg_bot, "stop_polling"):
                tg_bot.stop_polling()  # type: ignore
                if log:
                    log.info("[SHUTDOWN] Telegram polling stopped.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] Telegram stop failed: %s", e)

        # 4) TradeExecutor finalize (shutdown/close/stop/finalize/flush)
        try:
            if te is not None:
                for m in ("shutdown", "close", "stop", "finalize", "flush"):
                    if hasattr(te, m):
                        out = getattr(te, m)(reason) if m == "shutdown" else getattr(te, m)()
                        if asyncio.iscoroutine(out):
                            await out
                        break
                if log:
                    log.info("[SHUTDOWN] TradeExecutor finalized.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] TradeExecutor finalize failed: %s", e)

        # 5) PositionManager close/shutdown
        try:
            if pm is not None:
                for m in ("shutdown", "close", "stop"):
                    if hasattr(pm, m):
                        out = getattr(pm, m)(reason) if m == "shutdown" else getattr(pm, m)()
                        if asyncio.iscoroutine(out):
                            await out
                        break
                if log:
                    log.info("[SHUTDOWN] PositionManager closed.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] PositionManager close failed: %s", e)

        # 6) HTTP session close (binance client)
        try:
            if client is not None and hasattr(client, "close"):
                maybe = client.close()
                if asyncio.iscoroutine(maybe):
                    await maybe
                if log:
                    log.info("[SHUTDOWN] Binance client closed.")
        except Exception as e:
            if log:
                log.warning("[SHUTDOWN] Binance client close failed: %s", e)

        if log:
            log.info("[SHUTDOWN] cleanup done.")


_shutdown_mgr = ShutdownManager()

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

        try:
            risk_manager.set_telegram_bot(tg_bot)
        except Exception:
            pass

        dispatcher = getattr(tg_bot, "dispatcher", None)
        if dispatcher:
            if hasattr(tg_bot, "set_risk_manager"):
                tg_bot.set_risk_manager(risk_manager)
            else:
                setattr(tg_bot, "risk_manager", risk_manager)

            if system_logger:
                system_logger.info("[MAIN] TelegramBot'a RiskManager enjekte edildi (/risk aktif).")

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
                system_logger.warning("[MAIN] Telegram dispatcher yok. TELEGRAM_BOT_TOKEN ayarlı değilse komutlar çalışmaz.")

    except Exception as e:
        tg_bot = None
        if system_logger:
            system_logger.exception("[MAIN] TelegramBot init/enjeksiyon hata: %s", e)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", "bot:positions")

    enable_pg = str(os.getenv("ENABLE_PG_POS_LOG", "0")).strip()
    enable_pg_flag = enable_pg.lower() not in ("0", "false", "no", "off", "")
    pg_dsn = (os.getenv("PG_DSN") or "").strip() if enable_pg_flag else None

    if system_logger:
        system_logger.info("[POS][ENV] ENABLE_PG_POS_LOG=%s enable_pg_flag=%s", enable_pg, enable_pg_flag)

        pg = (pg_dsn or "").strip()
        masked = pg
        if pg and "://" in pg and "@" in pg:
            try:
                import re
                masked = re.sub(r":([^:@/]+)@", r":***@", pg)
            except Exception:
                masked = "***"

        system_logger.info("[POS][ENV] PG_DSN=%s", masked if masked else "(empty)")

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

    registry = ModelRegistry(model_dir=MODELS_DIR)

    mtf_intervals = parse_csv_env_list("MTF_INTERVALS", MTF_INTERVALS_DEFAULT)


    # --- Market meta (depth-based) ---
    depth_ws = None
    market_meta_builder = None

    try:
        enable_depth_ws = get_bool_env("ENABLE_DEPTH_WS", True)
        if enable_depth_ws:
            market_meta_builder = MarketMetaBuilder(
                spread_z_window=get_int_env("SPREAD_Z_WINDOW", 120),
                spread_shock_z_thr=get_float_env("SPREAD_SHOCK_Z_THR", 2.5),
                obi_levels=get_int_env("OBI_LEVELS", 5),
                liq_use_notional=get_bool_env("LIQ_USE_NOTIONAL", True),
            )
            depth_ws = BinanceDepthWS(symbol=symbol, builder=market_meta_builder, tfs=list(mtf_intervals))
            depth_ws.run_background()
            if system_logger:
                system_logger.info("[DEPTHWS] enabled | symbol=%s tfs=%s", symbol, list(mtf_intervals))
        else:
            if system_logger:
                system_logger.info("[DEPTHWS] disabled (ENABLE_DEPTH_WS!=1)")
    except Exception as e:
        depth_ws = None
        market_meta_builder = None
        if system_logger:
            system_logger.warning("[DEPTHWS] init failed: %s", e)

    # Fallback: DepthWS kapalıysa veya init fail olursa, whale/mtf tarafı meta üretebilsin
    if market_meta_builder is None:
        market_meta_builder = MarketMetaBuilder()

    hybrid_model = registry.get_hybrid(interval, model_dir=MODELS_DIR, logger=system_logger)

    models_by_interval = {
        itv: registry.get_hybrid(itv, model_dir=MODELS_DIR, logger=system_logger)
        for itv in mtf_intervals
    }

    mtf_ensemble = None
    if USE_MTF_ENS:
        try:
            mtf_ensemble = HybridMultiTFModel(
                model_dir=MODELS_DIR,
                intervals=mtf_intervals,
                logger=system_logger,
                models_by_interval=models_by_interval,
            )

            if system_logger:
                system_logger.info("[MAIN] MTF ensemble aktif: intervals=%s", list(mtf_intervals))

            try:
                seed_auc_history_if_missing(intervals=list(mtf_intervals), logger=system_logger)

                # AUC history runtime append (default OFF)
                if get_bool_env("AUC_RUNTIME_APPEND", False):
                    append_auc_used_once_per_hour(intervals=list(mtf_intervals), logger=system_logger)

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

    okx_ws = None

    # Telegram bot_data inject + OKXWS create
    try:
        if tg_bot is not None and getattr(tg_bot, "dispatcher", None):
            tg_bot.dispatcher.bot_data["risk_manager"] = risk_manager  # type: ignore
            tg_bot.dispatcher.bot_data["position_manager"] = position_manager  # type: ignore
            tg_bot.dispatcher.bot_data["trade_executor"] = trade_executor  # type: ignore
            tg_bot.dispatcher.bot_data["symbol"] = symbol  # type: ignore
            tg_bot.dispatcher.bot_data["interval"] = interval  # type: ignore

            if system_logger:
                try:
                    _okx_en = str(os.getenv("OKX_WS_ENABLE", "0")).strip().lower() in ("1", "true", "yes", "on")
                except Exception:
                    _okx_en = False

                if _okx_en:
                    try:
                        okx_ws = OKXWS(logger_=system_logger)
                        okx_ws.run_background()
                        system_logger.info(
                            "[OKXWS] enabled: instId=%s channel=%s",
                            getattr(okx_ws, "inst_id", None),
                            getattr(okx_ws, "channel", None),
                        )
                    except Exception as e:
                        okx_ws = None
                        system_logger.warning("[OKXWS] init/start failed: %s", e)
                else:
                    system_logger.info("[OKXWS] disabled (OKX_WS_ENABLE!=1)")

                try:
                    tg_bot.dispatcher.bot_data["okx_ws"] = okx_ws  # type: ignore
                except Exception:
                    pass

                system_logger.info("[MAIN] Telegram bot_data injected: risk_manager/position_manager/trade_executor/symbol/interval")
    except Exception as e:
        if system_logger:
            system_logger.warning("[MAIN] Telegram bot_data inject error: %s", e)

    # shutdown manager register (tg + okx + client + pm + te)
    _shutdown_mgr.register(
        tg_bot=tg_bot,
        okx_ws=okx_ws,
        depth_ws=depth_ws,
        client=client,
        position_manager=position_manager,
        trade_executor=trade_executor,
    )

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
        "registry": registry,
        "models_by_interval": models_by_interval,
        "okx_ws": okx_ws,
        "market_meta_builder": market_meta_builder,
    }
def _normalize_signal(sig: Any) -> str:
    s = str(sig).strip().lower()
    if s in ("buy", "long"):
        return "long"
    if s in ("sell", "short"):
        return "short"
    return "hold"


# ----------------------------------------------------------------------
# Heavy engine
# ----------------------------------------------------------------------
class HeavyEngine:
    def __init__(self, objs: Dict[str, Any]):
        self.objs = objs
        self.client = objs["client"]
        self.trade_executor = objs["trade_executor"]
        self.hybrid_model = objs["hybrid_model"]
        self.mtf_ensemble = objs.get("mtf_ensemble")
        self.whale_detector = objs.get("whale_detector")
        self.tg_bot = objs.get("tg_bot")
        self.okx_ws = objs.get("okx_ws")
        self.market_meta_builder = objs.get("market_meta_builder")
        self.interval = objs.get("interval", os.getenv("INTERVAL", "5m"))
        self.data_limit = int(os.getenv("DATA_LIMIT", "500"))
        self.mtf_intervals: List[str] = objs.get("mtf_intervals") or parse_csv_env_list("MTF_INTERVALS", MTF_INTERVALS_DEFAULT)

        self.anomaly_detector = AnomalyDetector(logger=system_logger)

        self._schema_cache: Dict[str, Optional[List[str]]] = {}
        self._prob_stab_by_symbol: Dict[str, ProbStabilizer] = {}
        self._prev_signal: Dict[str, str] = {}
        self._prev_notif_ts: Dict[str, float] = {}

    def _get_prob_stab(self, symbol: str) -> ProbStabilizer:
        if symbol in self._prob_stab_by_symbol:
            return self._prob_stab_by_symbol[symbol]

        ps = ProbStabilizer(
            alpha=float(os.getenv("PBUY_EMA_ALPHA", "0.20")),
            buy_thr=float(os.getenv("BUY_THRESHOLD", "0.60")),
            sell_thr=float(os.getenv("SELL_THRESHOLD", "0.40")),
        )
        self._prob_stab_by_symbol[symbol] = ps
        return ps

    def _load_schema_from_disk(self, itv: str) -> Optional[List[str]]:
        if itv in self._schema_cache:
            return self._schema_cache[itv]

        try:
            p = os.path.join(MODELS_DIR, f"model_meta_{itv}.json")
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f) or {}
            sch = d.get("feature_schema")
            if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
                self._schema_cache[itv] = sch
                return sch
        except Exception:
            pass

        self._schema_cache[itv] = None
        return None

    def _fallback_schema_from_model(self) -> Optional[List[str]]:
        meta = getattr(self.hybrid_model, "meta", {}) or {}
        sch = meta.get("feature_schema")
        if isinstance(sch, list) and sch and all(isinstance(x, str) for x in sch):
            return sch
        return None

    def _schema_for(self, itv: str) -> Optional[List[str]]:
        return self._load_schema_from_disk(itv) or self._fallback_schema_from_model()

    def _normalize_feat_df(self, feat_df: pd.DataFrame, itv: str) -> pd.DataFrame:
        sch = self._schema_for(itv)
        if not sch:
            if system_logger:
                system_logger.warning("[SCHEMA] No feature_schema for interval=%s (meta missing). Using raw features.", itv)
            return feat_df

        def _log_missing(missing_cols):
            if system_logger and missing_cols:
                system_logger.warning("[SCHEMA] interval=%s missing cols (filled=0): %s", itv, missing_cols)

        return normalize_to_schema(feat_df, sch, log_missing=_log_missing)

    async def run_once(self, symbol: str) -> Dict[str, Any]:
        global HYBRID_MODE, TRAINING_MODE, USE_MTF_ENS

        interval = self.interval
        timer = _StepTimer()

        raw_df = await fetch_klines(
            client=self.client,
            symbol=symbol,
            interval=interval,
            limit=self.data_limit,
            logger=system_logger,
        )
        timer.mark("fetch_klines")

        feat_df = build_features(raw_df)
        feat_df = self._normalize_feat_df(feat_df, interval)
        timer.mark("features+schema")

        sch_main = self._schema_for(interval)
        feat_df = self.anomaly_detector.filter_anomalies(
            feat_df,
            schema=sch_main,
            context=f"heavy:{symbol}:{interval}",
        )
        timer.mark("anomaly_filter")

        X_live = feat_df.tail(500)

        p_arr_single, debug_single = self.hybrid_model.predict_proba(X_live)
        p_single = float(p_arr_single[-1])
        timer.mark("predict_single")

        meta = getattr(self.hybrid_model, "meta", {}) or {}
        model_conf_factor = float(meta.get("confidence_factor", 1.0) or 1.0)
        best_auc = float(meta.get("best_auc", 0.5) or 0.5)
        best_side = meta.get("best_side", "long")

        mtf_feats: Dict[str, pd.DataFrame] = {interval: feat_df}
        mtf_whale_raw: Dict[str, pd.DataFrame] = {interval: raw_df}

        if USE_MTF_ENS and self.mtf_ensemble is not None:
            for itv in self.mtf_intervals:
                if itv == interval:
                    continue
                try:
                    raw_df_itv = await fetch_klines(
                        client=self.client,
                        symbol=symbol,
                        interval=itv,
                        limit=self.data_limit,
                        logger=system_logger,
                    )

                    feat_df_itv = build_features(raw_df_itv)
                    feat_df_itv = self._normalize_feat_df(feat_df_itv, itv)

                    sch_itv = self._schema_for(itv)
                    feat_df_itv = self.anomaly_detector.filter_anomalies(
                        feat_df_itv,
                        schema=sch_itv,
                        context=f"heavy:{symbol}:{itv}",
                    )

                    mtf_feats[itv] = feat_df_itv
                    mtf_whale_raw[itv] = raw_df_itv

                except Exception as e:
                    if system_logger:
                        system_logger.warning("[MTF] %s interval'i hazırlanırken hata: %s", itv, e)

        timer.mark("mtf_prepare")

        p_used = p_single
        mtf_debug: Optional[Dict[str, Any]] = None

        if USE_MTF_ENS and self.mtf_ensemble is not None:
            try:
                X_by_interval: Dict[str, pd.DataFrame] = {}
                for itv, df_itv in mtf_feats.items():
                    sch_itv = self._schema_for(itv)
                    if not sch_itv:
                        continue
                    X_by_interval[itv] = normalize_to_schema(df_itv, sch_itv).tail(500)

                if X_by_interval:
                    p_ens, mtf_debug = self.mtf_ensemble.predict_proba_multi(
                        X_dict=X_by_interval,
                        standardize_auc_key="auc_used",
                        standardize_overwrite=False,
                    )
                    p_used = float(p_ens)

            except Exception as e:
                if system_logger:
                    system_logger.warning("[MTF] Ensemble hesaplanırken hata: %s", e)
                p_used = p_single
                mtf_debug = None

        timer.mark("mtf_ensemble")

        # Whale: main.py sadece çağırır (asıl logic güçlendirme data/whale_detector.py içinde)
        whale_meta: Dict[str, Any] = {"direction": "none", "score": 0.0, "per_tf": {}}
        whale_dir = "none"
        whale_score = 0.0

        if self.whale_detector is not None:
            try:
                if hasattr(self.whale_detector, "analyze_multiple_timeframes"):
                    market_meta_by_tf = None

                    okx_dfs = None

                    if self.market_meta_builder is not None:

                        try:

                            market_meta_by_tf = self.market_meta_builder.build_meta_by_tf(symbol, list(mtf_whale_raw.keys()))

                        except Exception:

                            market_meta_by_tf = None

                    # OKX df'leri opsiyonel: burada sadece placeholder (WS/REST ile doldurulacak)

                    # okx_dfs = {tf: df} gibi verilirse cross-exchange confirm çalışır.

                    whale_signals = self.whale_detector.analyze_multiple_timeframes(
                        mtf_whale_raw,
                        okx_dfs=okx_dfs,
                        market_meta_by_tf=market_meta_by_tf,
                        symbol=symbol,
                    )
                    best_tf = None
                    best_score = 0.0

                    for tf, sig in whale_signals.items():
                        whale_meta["per_tf"][tf] = {"direction": sig.direction, "score": sig.score, "reason": sig.reason}
                        if sig.direction != "none" and sig.score > best_score:
                            best_score = sig.score
                            best_tf = tf

                    if best_tf is not None:
                        best_sig = whale_signals[best_tf]
                        whale_meta.update(
                            {"direction": best_sig.direction, "score": best_sig.score, "best_tf": best_tf, "best_reason": best_sig.reason}
                        )

                elif hasattr(self.whale_detector, "from_klines"):
                    ws = self.whale_detector.from_klines(raw_df)
                    whale_meta.update({"direction": ws.direction, "score": ws.score, "reason": ws.reason, "meta": ws.meta})

                whale_dir = str(whale_meta.get("direction", "none") or "none")
                whale_score = float(whale_meta.get("score", 0.0) or 0.0)

            except Exception as e:
                if system_logger:
                    system_logger.warning("[WHALE] MTF whale hesaplanırken hata: %s", e)

        timer.mark("whale")

        atr_period = int(os.getenv("ATR_PERIOD", "14"))
        atr_value = compute_atr_from_klines(raw_df, period=atr_period)
        timer.mark("atr")

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
            "latency_steps": dict(timer.steps),
        }

        signal_side: Optional[str] = None
        use_ema_signal = get_bool_env("USE_PBUY_STABILIZER_SIGNAL", False)
        ema_whale_only = get_bool_env("EMA_WHALE_ONLY", False)
        ema_whale_thr = float(os.getenv("EMA_WHALE_THR", "0.50"))

        prob_stab = self._get_prob_stab(symbol)

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

        last_price = 0.0
        try:
            if "close" in raw_df.columns and len(raw_df) > 0:
                last_price = float(pd.to_numeric(raw_df["close"].iloc[-1], errors="coerce"))
                if not (last_price == last_price):
                    last_price = 0.0
        except Exception:
            last_price = 0.0

        await self.trade_executor.execute_decision(
            signal=signal_side,
            symbol=symbol,
            price=float(last_price),
            size=None,
            interval=interval,
            training_mode=TRAINING_MODE,
            hybrid_mode=HYBRID_MODE,
            probs=probs,
            extra=extra,
        )

        extra["latency_total_sec"] = float(timer.total())
        if system_logger and get_bool_env("LOG_LATENCY", False):
            system_logger.info(
                "[LATENCY] %s total=%.4fs steps=%s",
                symbol, float(extra["latency_total_sec"]), {k: round(v, 4) for k, v in timer.steps.items()}
            )

        return {
            "symbol": symbol,
            "signal": signal_side,
            "p_used": float(p_used),
            "p_single": float(p_single),
            "price": float(last_price),
            "extra": extra,
        }
# ----------------------------------------------------------------------
# Non-scan mode: tek symbol heavy loop
# ----------------------------------------------------------------------
async def bot_loop_single_symbol(engine: HeavyEngine, symbol: str) -> None:
    if system_logger:
        system_logger.info(
            "[MAIN] Bot loop started for %s (%s, TRAINING_MODE=%s, HYBRID_MODE=%s, USE_MTF_ENS=%s)",
            symbol, engine.interval, TRAINING_MODE, HYBRID_MODE, USE_MTF_ENS
        )

    while True:
        try:
            await engine.run_once(symbol)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if system_logger:
                system_logger.exception("[LOOP ERROR] %s", e)
            else:
                print("[LOOP ERROR]", e)

        await asyncio.sleep(_sleep_jitter(float(LOOP_SLEEP_SECONDS), 0.10))


# ----------------------------------------------------------------------
# Scanner → Candidates → Heavy Full MTF (sadece adaylar)
# ----------------------------------------------------------------------
async def scanner_loop(engine: HeavyEngine) -> None:
    symbols = parse_symbols_env()

    scan_interval = os.getenv("SCAN_INTERVAL", "1m").strip()
    scan_limit = int(os.getenv("SCAN_LIMIT", "200"))
    scan_every = float(os.getenv("SCAN_EVERY_SEC", "15"))
    topk = int(os.getenv("SCAN_TOPK", "3"))
    conc = int(os.getenv("SCAN_CONCURRENCY", "4"))

    # Backward compatible flags (eski isimler)
    legacy_fast_mode = get_bool_env("SCAN_FAST_MODE", False)

    # Yeni öneri: SCAN_FAST + SCAN_AGGRESSIVE
    scan_fast = get_bool_env("SCAN_FAST", legacy_fast_mode)
    scan_aggressive = get_bool_env("SCAN_AGGRESSIVE", False)

    # Warm cache (opsiyonel)
    warm_cache = get_bool_env("SCAN_FAST_WARM_CACHE", False)

    # Volatility-based adaptive (opsiyonel)
    adaptive = get_bool_env("SCAN_ADAPTIVE", False) or scan_fast

    # FAST tuning env
    fast_min_every = float(os.getenv("SCAN_FAST_MIN_EVERY_SEC", "5"))
    fast_max_every = float(os.getenv("SCAN_FAST_MAX_EVERY_SEC", "25"))
    fast_ref_vol = float(os.getenv("SCAN_FAST_REF_VOL", "0.004"))
    fast_topk_min = int(os.getenv("SCAN_FAST_TOPK_MIN", "2"))
    fast_topk_max = int(os.getenv("SCAN_FAST_TOPK_MAX", "5"))

    # AGGRESSIVE preset (hız için clamp/boost)
    if scan_aggressive:
        scan_limit = int(min(scan_limit, int(os.getenv("SCAN_AGGR_LIMIT", "120"))))
        conc = int(max(conc, int(os.getenv("SCAN_AGGR_CONC", "8"))))
        topk = int(max(topk, int(os.getenv("SCAN_AGGR_TOPK", "5"))))
        scan_every = float(min(scan_every, float(os.getenv("SCAN_AGGR_EVERY_SEC", "7"))))

    err_backoff_base = float(os.getenv("SCAN_ERR_BACKOFF_SEC", "2.0"))
    err_backoff_max = float(os.getenv("SCAN_ERR_BACKOFF_MAX_SEC", "30.0"))
    scan_failures = 0

    sem = asyncio.Semaphore(max(1, conc))

    light_scanner = None
    if LightScanner is not None:
        try:
            light_scanner = LightScanner()  # type: ignore
        except Exception:
            light_scanner = None

    async def _scan_one(sym: str) -> Dict[str, Any]:
        async with sem:
            try:
                df = await fetch_klines(
                    client=engine.client,
                    symbol=sym,
                    interval=scan_interval,
                    limit=scan_limit,
                    logger=system_logger,
                )

                if light_scanner is not None and hasattr(light_scanner, "score"):
                    try:
                        out = light_scanner.score(df)  # type: ignore
                        if isinstance(out, dict):
                            score = float(out.get("score", 0.0) or 0.0)
                            reason = out.get("reason")
                        else:
                            score = float(out)
                            reason = None
                    except Exception:
                        score = _light_score_from_klines(df)
                        reason = None
                else:
                    score = _light_score_from_klines(df)
                    reason = None

                vol_reg = _quick_vol_regime(df, window=min(80, max(30, int(scan_limit))))

                last_px = None
                try:
                    if "close" in df.columns and len(df) > 0:
                        last_px = float(pd.to_numeric(df["close"].iloc[-1], errors="coerce"))
                except Exception:
                    last_px = None

                return {"symbol": sym, "score": float(score), "reason": reason, "last": last_px, "vol_reg": float(vol_reg)}
            except Exception as e:
                if system_logger:
                    system_logger.debug("[SCAN] %s failed: %s", sym, e)
                return {"symbol": sym, "score": 0.0, "reason": None, "last": None, "vol_reg": 0.0, "err": str(e)}

    if system_logger:
        system_logger.info(
            "[SCAN] enabled | symbols=%d interval=%s topk=%d every=%.1fs conc=%d fast=%s aggressive=%s adaptive=%s",
            len(symbols), scan_interval, topk, scan_every, conc, scan_fast, scan_aggressive, adaptive
        )

    while True:
        t0 = time.time()
        try:
            results = await asyncio.gather(*[_scan_one(s) for s in symbols], return_exceptions=True)

            rows: List[Dict[str, Any]] = []
            for r in results:
                if isinstance(r, Exception):
                    continue
                if isinstance(r, dict):
                    rows.append(r)

            rows = sorted(rows, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

            # -------------------------
            # FAST adaptive scan config
            # -------------------------
            scan_every_eff = float(scan_every)
            topk_eff = int(topk)
            if scan_fast and rows:
                vol_regs = [float(r.get("vol_reg", 0.0) or 0.0) for r in rows[:min(10, len(rows))]]
                vol_regs = [v for v in vol_regs if v > 0]
                vol_med = float(pd.Series(vol_regs).median()) if vol_regs else 0.0

                # scan_every: vol yükseldikçe daha sık
                if vol_med > 1e-9:
                    scale = float(fast_ref_vol / vol_med)
                else:
                    scale = 1.0
                scale = max(0.4, min(3.0, scale))
                scan_every_eff = float(max(fast_min_every, min(fast_max_every, scan_every * scale)))

                # topk: vol yükseldikçe daha fazla aday
                if vol_med >= fast_ref_vol * 1.2:
                    topk_eff = int(min(fast_topk_max, max(fast_topk_min, topk + 2)))
                elif vol_med <= fast_ref_vol * 0.7:
                    topk_eff = int(max(fast_topk_min, min(topk, fast_topk_max)))
                else:
                    topk_eff = int(max(fast_topk_min, min(topk, fast_topk_max)))

                if system_logger and get_bool_env("LOG_SCAN_FAST", True):
                    system_logger.info(
                        "[SCAN-FAST] vol_med=%.6f ref=%.6f -> every=%.2fs topk=%d",
                        vol_med, fast_ref_vol, scan_every_eff, topk_eff
                    )

            candidates = [r for r in rows[:max(1, topk_eff)] if float(r.get("score", 0.0) or 0.0) > 0.0]
            if not candidates:
                candidates = rows[:1] if rows else []

            cand_syms = [c["symbol"] for c in candidates]

            if system_logger:
                system_logger.info(
                    "[SCAN] candidates=%s | scores(top10)=%s",
                    cand_syms,
                    [(r["symbol"], round(float(r.get("score", 0.0) or 0.0), 2)) for r in rows[:min(10, len(rows))]],
                )

            # Legacy adaptive: probe candidate[0] üzerinden parametre güncelle (opsiyonel)
            if adaptive and cand_syms:
                try:
                    df_probe = await fetch_klines(
                        client=engine.client,
                        symbol=cand_syms[0],
                        interval=scan_interval,
                        limit=min(300, max(120, scan_limit)),
                        logger=None,
                    )
                    v = _volatility_score(df_probe)
                    p = _adaptive_scan_params(scan_every_eff, topk_eff, v)
                    scan_every_eff = float(p["scan_every"])
                    topk_eff = int(p["topk"])
                    if system_logger and get_bool_env("LOG_SCAN_ADAPT", True):
                        system_logger.info(
                            "[SCAN-ADAPT] v=%.3f -> every=%.2fs topk=%d",
                            float(p["vol_score"]), scan_every_eff, topk_eff
                        )
                except Exception:
                    pass

            # Warm cache for candidates (OFFLINE LRU'yu ısıt)
            if scan_fast and warm_cache and cand_syms:
                try:
                    await asyncio.gather(
                        *[
                            fetch_klines(engine.client, s, scan_interval, scan_limit, None)
                            for s in cand_syms
                        ],
                        return_exceptions=True,
                    )
                except Exception:
                    pass

            # Heavy run + latency log
            for sym in cand_syms:
                try:
                    t_heavy0 = time.time()
                    await engine.run_once(sym)
                    t_heavy = time.time() - t_heavy0
                    if system_logger and get_bool_env("LOG_LATENCY", False):
                        system_logger.info("[LATENCY] heavy.run_once symbol=%s dt=%.3fs", sym, t_heavy)
                except Exception as e:
                    if system_logger:
                        system_logger.warning("[HEAVY] %s run_once failed: %s", sym, e)

            dt = time.time() - t0
            if system_logger and get_bool_env("LOG_LATENCY", False):
                system_logger.info("[LATENCY] scan->predict batch dt=%.3fs candidates=%s", dt, cand_syms)

            sleep_s = max(0.0, scan_every_eff - dt)
            await asyncio.sleep(_sleep_jitter(sleep_s, 0.10))

            scan_failures = 0

        except asyncio.CancelledError:
            raise
        except Exception as e:
            scan_failures += 1
            backoff = min(err_backoff_max, err_backoff_base * (2 ** max(0, scan_failures - 1)))
            if system_logger:
                system_logger.exception("[SCAN LOOP ERROR] %s | backoff=%.1fs failures=%d", e, backoff, scan_failures)
            await asyncio.sleep(_sleep_jitter(backoff, 0.20))

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

    try:
        Credentials.refresh_from_env()
    except Exception:
        pass

    try:
        Credentials.log_missing(prefix='[ENV]')
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

    # ENABLE_WS -> BinanceWS create & register for shutdown
    enable_ws = get_bool_env("ENABLE_WS", False)
    binance_ws = None
    if enable_ws:
        sym = os.getenv("SYMBOL", getattr(Settings, "SYMBOL", "BTCUSDT"))
        binance_ws = BinanceWS(symbol=sym)
        binance_ws.run_background()
        _shutdown_mgr.register(binance_ws=binance_ws)
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=true -> websocket started. symbol=%s", sym)
    else:
        if system_logger:
            system_logger.info("[WS] ENABLE_WS=false -> websocket disabled.")

    trading_objects = create_trading_objects()
    engine = HeavyEngine(trading_objects)

    scan_enable = get_bool_env("SCAN_ENABLE", False)
    if scan_enable:
        await scanner_loop(engine)
        return

    symbol = str(trading_objects.get("symbol") or os.getenv("SYMBOL") or getattr(Settings, "SYMBOL", "BTCUSDT")).upper()
    await bot_loop_single_symbol(engine, symbol)


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def _request_shutdown(sig_name: str):
        if system_logger:
            system_logger.info("[MAIN] signal received: %s", sig_name)
        try:
            loop.call_soon_threadsafe(stop_event.set)
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except NotImplementedError:
            pass

    async def _runner():
        main_task = asyncio.create_task(async_main(), name="main-task")
        stop_task = asyncio.create_task(stop_event.wait(), name="stop-wait")

        done, pending = await asyncio.wait(
            {main_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop_task in done and not main_task.done():
            try:
                main_task.cancel()
            except Exception:
                pass

        # cleanup resources (ws/tg/http/pm/te)
        try:
            await _shutdown_mgr.shutdown(reason="signal_or_exit")
        except Exception:
            pass

        for t in pending:
            try:
                t.cancel()
            except Exception:
                pass

        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            pass

        try:
            await asyncio.gather(main_task, return_exceptions=True)
        except Exception:
            pass

    try:
        loop.run_until_complete(_runner())
    finally:
        try:
            pending = asyncio.all_tasks(loop=loop)
        except TypeError:
            pending = asyncio.all_tasks()

        for task in pending:
            task.cancel()

        try:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass

        # Son güvenlik: shutdown bir daha çağrılsa bile idempotent
        try:
            loop.run_until_complete(_shutdown_mgr.shutdown(reason="loop_finally"))
        except Exception:
            pass

        loop.close()


if __name__ == "__main__":
    main()

