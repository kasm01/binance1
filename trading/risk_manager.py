class RiskManager:
    def __init__(self, max_risk_per_trade=0.02):
        self.max_risk = max_risk_per_trade

    def check_risk(self, capital, position_qty):
        risk = position_qty / capital
        if risk > self.max_risk:
            return False
        return True
