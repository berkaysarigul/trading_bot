import ccxt
from config.api_config import BINANCE_API_KEY, BINANCE_API_SECRET

class OrderExecutor:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })

    def create_order(self, symbol, side, amount, price=None, order_type='market'):
        try:
            if order_type == 'market':
                order = self.exchange.create_market_order(symbol, side, amount)
            else:
                order = self.exchange.create_limit_order(symbol, side, amount, price)
            return order
        except Exception as e:
            print(f"Order error: {e}")
            return None

    def cancel_order(self, symbol, order_id):
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            print(f"Cancel error: {e}")
            return None

    def fetch_order_status(self, symbol, order_id):
        try:
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            print(f"Status error: {e}")
            return None 