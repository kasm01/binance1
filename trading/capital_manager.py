class CapitalManager:
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.available_capital = total_capital

    def allocate(self, risk_pct):
        capital_to_use = self.available_capital * risk_pct
        self.available_capital -= capital_to_use
        return capital_to_use

    def release(self, amount):
        self.available_capital += amount
