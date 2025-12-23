# core/prob_stabilizer.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import math


@dataclass
class ProbStabilizer:
    """
    EMA stabilizer for probabilities (p_buy).
    - supports dynamic alpha update
    - supports z-score clipping on raw p before EMA
    """
    alpha: float = 0.20
    buy_thr: float = 0.60
    sell_thr: float = 0.40

    # z-score clip
    zwin: int = 60          # rolling window for zscore
    zclip: float = 3.0      # clip threshold
    use_zclip: bool = True

    _ema: Optional[float] = None
    _raw_hist: Deque[float] = None

    def __post_init__(self):
        if self._raw_hist is None:
            self._raw_hist = deque(maxlen=max(5, int(self.zwin)))

        # sanitize
        self.alpha = float(self.alpha)
        self.buy_thr = float(self.buy_thr)
        self.sell_thr = float(self.sell_thr)

    @property
    def ema(self) -> Optional[float]:
        return self._ema

    def set_alpha(self, alpha: float) -> None:
        a = float(alpha)
        if not math.isfinite(a):
            return
        # keep inside (0,1]
        if a <= 0:
            a = 0.01
        if a > 1:
            a = 1.0
        self.alpha = a

    def _zscore_clip(self, x: float) -> float:
        if not self.use_zclip:
            return x
        if len(self._raw_hist) < max(10, int(self.zwin) // 2):
            return x

        vals = list(self._raw_hist)
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, (len(vals) - 1))
        sd = math.sqrt(var) if var > 1e-12 else 0.0
        if sd <= 0:
            return x

        z = (x - mu) / sd
        zc = float(self.zclip)
        if z > zc:
            x = mu + zc * sd
        elif z < -zc:
            x = mu - zc * sd

        # clamp to [0,1]
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0
        return float(x)

    def update(self, p_raw: float) -> float:
        # clamp & store raw
        try:
            x = float(p_raw)
        except Exception:
            x = 0.5


        # --- SATDBG: p_raw -> clamp -> zclip -> ema (ENV: PROBSTAB_DEBUG=1) ---
        try:
            _dbg = str(__import__('os').getenv('PROBSTAB_DEBUG','0')).lower() in ('1','true','yes','on')
        except Exception:
            _dbg = False
        if _dbg:
            try:
                # x daha clamp edilmeden burada (float(p_raw) sonrası) görülecek
                self._dbg_last = {'p_raw': p_raw, 'x_float': x, 'ema_before': self._ema, 'alpha': self.alpha}
            except Exception:
                pass
        if x < 0:
            x = 0.0
        elif x > 1:
            x = 1.0

        # add to raw history first (so stats include recent distribution)
        self._raw_hist.append(x)

        # z-score clip
        x2 = self._zscore_clip(x)
        # --- debug: saturate/zclip gözlem (ENV: PROBSTAB_DEBUG=1) ---
        try:
            import os
            _dbg = str(os.getenv('PROBSTAB_DEBUG','0')).lower() in ('1','true','yes','on')
        except Exception:
            _dbg = False
        if _dbg:
            try:
                # x: clamp sonrası, x2: zclip sonrası, ema: önce/sonra
                _ema_prev = self._ema
                self._last_dbg = (float(p_raw), float(x), float(x2), None if _ema_prev is None else float(_ema_prev))
            except Exception:
                pass

        if _dbg:
            try:
                if isinstance(getattr(self,'_dbg_last',None), dict):
                    self._dbg_last.update({'x_clamped': x, 'x_zclip': x2})
            except Exception:
                pass

        if self._ema is None:
            self._ema = float(x2)
            return self._ema

        a = float(self.alpha)
        self._ema = a * float(x2) + (1.0 - a) * float(self._ema)
        if _dbg:
            try:
                if isinstance(getattr(self,'_dbg_last',None), dict):
                    self._dbg_last.update({'ema_after': self._ema})
            except Exception:
                pass
        return float(self._ema)

    def signal(self, p_ema: float) -> str:
        try:
            p = float(p_ema)
        except Exception:
            p = 0.5

        if p >= float(self.buy_thr):
            return "BUY"
        if p <= float(self.sell_thr):
            return "SELL"
        return "HOLD"
