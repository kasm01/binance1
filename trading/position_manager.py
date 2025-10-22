from core.logger import system_logger

class PositionManager:
    def __init__(self):
        self.active_positions = {}

    def open_position(self, symbol, qty, side, price):
        pos_id = f"{symbol}_{len(self.active_positions)+1}"
        self.active_positions[pos_id] = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "entry_price": price
        }
        system_logger.info(f"Position opened: {pos_id}")
        return pos_id

    def close_position(self, pos_id, exit_price):
        if pos_id in self.active_positions:
            pos = self.active_positions.pop(pos_id)
            pnl = (exit_price - pos['entry_price']) * pos['qty']
            system_logger.info(f"Position {pos_id} closed. PnL: {pnl}")
            return pnl
        return None
