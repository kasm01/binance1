class ProbStabilizer:
    """
    p_buy EMA + hysteresis (buy/sell thresholds)
    """
    def __init__(self, alpha: float = 0.20, buy_thr: float = 0.60, sell_thr: float = 0.40):
        self.alpha = float(alpha)
        self.buy_thr = float(buy_thr)
        self.sell_thr = float(sell_thr)
        self._ema = None

    def update(self, p_buy: float) -> float:
        p = float(p_buy)
        if self._ema is None:
            self._ema = p
        else:
            self._ema = self.alpha * p + (1.0 - self.alpha) * self._ema
        return float(self._ema)

    def signal(self, p_buy_ema: float) -> str:
        p = float(p_buy_ema)
        if p >= self.buy_thr:
            return "BUY"
        if p <= self.sell_thr:
            return "SELL"
        return "HOLD"
