from trading.order_executor import OrderExecutor
from utils.risk_management import stop_loss_take_profit

class PositionManager:
    def __init__(self):
        self.executor = OrderExecutor()

    def open_position(self, symbol, side, amount, entry_price, stop_loss_pct, take_profit_pct):
        order = self.executor.create_order(symbol, side, amount)
        stop_loss, take_profit = stop_loss_take_profit(entry_price, stop_loss_pct, take_profit_pct)
        # Stop-loss ve take-profit emirleri için örnek (Binance spotta OCO ile yapılabilir)
        # self.executor.create_oco_order(symbol, side, amount, take_profit, stop_loss)
        return order, stop_loss, take_profit

    def close_position(self, symbol, amount):
        # Pozisyonu kapatmak için ters işlem
        side = 'sell'  # long pozisyon için
        return self.executor.create_order(symbol, side, amount) 