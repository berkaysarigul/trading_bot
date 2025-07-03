from trading.position_manager import PositionManager
import time

class LiveTrader:
    def __init__(self, agent, symbols, interval=60, paper_mode=True):
        self.position_manager = PositionManager()
        self.agent = agent
        self.symbols = symbols
        self.interval = interval  # saniye
        self.paper_mode = paper_mode
        self.paper_balance = 10000  # Başlangıç bakiyesi (paper trading için)
        self.paper_positions = {symbol: 0 for symbol in symbols}

    def run(self, get_observation_fn):
        while True:
            try:
                obs = get_observation_fn()
                actions, _, _ = self.agent.act(obs)
                for i, symbol in enumerate(self.symbols):
                    action = actions[i]
                    if self.paper_mode:
                        # Paper trading: sadece bakiyeyi ve pozisyonu güncelle
                        price = obs['close'] if isinstance(obs, dict) and 'close' in obs else 1.0
                        if action > 0.1:
                            amount = abs(action)
                            self.paper_positions[symbol] += amount
                            self.paper_balance -= amount * price
                        elif action < -0.1:
                            amount = abs(action)
                            self.paper_positions[symbol] -= amount
                            self.paper_balance += amount * price
                    else:
                        if action > 0.1:
                            self.position_manager.open_position(symbol, 'buy', abs(action), obs['close'], 0.02, 0.04)
                        elif action < -0.1:
                            self.position_manager.close_position(symbol, abs(action))
                print(f"[Paper Mode] Balance: {self.paper_balance}, Positions: {self.paper_positions}") if self.paper_mode else None
                time.sleep(self.interval)
            except Exception as e:
                print(f"Live trading error: {e}")
                time.sleep(10) 