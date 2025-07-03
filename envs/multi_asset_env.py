import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class MultiAssetTradingEnv(gym.Env):
    def __init__(self, df_dict, initial_balance=100000, commission_rate=0.001, max_position_size=0.95, lookback_window_size=10):
        super().__init__()
        self.symbols = list(df_dict.keys())
        self.n_assets = len(self.symbols)
        self.dfs = df_dict
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.max_position_size = max_position_size
        self.lookback_window_size = lookback_window_size
        self.current_step = 0
        self.balance = initial_balance
        self.positions = np.zeros(self.n_assets, dtype=np.float32)  # Her varlık için pozisyon oranı (0-1)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.scalers = {}
        self._prepare_data()
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)
        obs_dim = self.n_assets * 4 * self.lookback_window_size + 1 + self.n_assets  # 4 teknik gösterge x lookback x varlık + bakiye + pozisyonlar
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def _prepare_data(self):
        # Her sembol için sadece gerekli teknik göstergeleri ve scaler'ı hazırla
        self.obs_cols = {}
        for i, symbol in enumerate(self.symbols):
            df = self.dfs[symbol]
            cols = [f'Close_{symbol}', f'MA10_{symbol}', f'EMA20_{symbol}', f'MACD_{symbol}']
            self.obs_cols[symbol] = cols
            self.scalers[symbol] = StandardScaler().fit(df[cols])
        # Ortak tarihleri bul
        self.dates = sorted(list(set.intersection(*[set(df.index) for df in self.dfs.values()])))
        for symbol in self.symbols:
            self.dfs[symbol] = self.dfs[symbol].loc[self.dates]
        self.n_steps = len(self.dates)

    def reset(self, seed=None, options=None):
        self.current_step = self.lookback_window_size
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        return self._next_observation(), {}

    def _get_portfolio_value(self):
        total = self.balance
        for i, symbol in enumerate(self.symbols):
            price = float(self.dfs[symbol].iloc[self.current_step][f'Close_{symbol}'])
            total += self.positions[i] * price
        return total

    def _next_observation(self):
        obs = []
        for i, symbol in enumerate(self.symbols):
            df = self.dfs[symbol]
            cols = self.obs_cols[symbol]
            # Lookback window
            window = df.iloc[self.current_step - self.lookback_window_size:self.current_step][cols].values
            window_scaled = self.scalers[symbol].transform(window)
            obs.extend(window_scaled.flatten())
        obs.append(self.balance)
        obs.extend(self.positions)
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        prev_value = self._get_portfolio_value()
        info = {}
        # Aksiyonları normalize et (toplam mutlak değer 1'i geçmesin)
        norm_actions = actions / (np.sum(np.abs(actions)) + 1e-8)
        prices = np.array([float(self.dfs[symbol].iloc[self.current_step][f'Close_{symbol}']) for symbol in self.symbols])
        # Her varlık için pozisyonu güncelle
        for i, symbol in enumerate(self.symbols):
            desired_position = norm_actions[i] * self.max_position_size  # -max..+max arası
            current_position = self.positions[i]
            price = prices[i]
            # Satış
            if desired_position < current_position:
                sell_amount = current_position - desired_position
                proceeds = sell_amount * price * (1 - self.commission_rate)
                self.balance += proceeds
                self.positions[i] -= sell_amount
                if self.positions[i] < 1e-6:
                    self.entry_prices[i] = 0.0
            # Alış
            elif desired_position > current_position:
                buy_amount = desired_position - current_position
                cost = buy_amount * price * (1 + self.commission_rate)
                if self.balance >= cost:
                    self.balance -= cost
                    self.positions[i] += buy_amount
                    if self.entry_prices[i] == 0.0:
                        self.entry_prices[i] = price
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        new_value = self._get_portfolio_value()
        reward = (new_value - prev_value) / self.initial_balance
        obs = self._next_observation()
        info['portfolio_value'] = new_value
        info['reward'] = reward
        return obs, reward, done, False, info 