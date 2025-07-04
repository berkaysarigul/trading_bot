# multi_asset_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class MultiAssetTradingEnv(gym.Env):
    def __init__(self, dfs, lookback_window_size=10, initial_balance=10000):
        super(MultiAssetTradingEnv, self).__init__()
        self.symbols = list(dfs.keys())
        self.dfs = dfs
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance

        self.current_step = None
        self.balance = None
        self.positions = None
        self.prices = None

        # Observation: for each symbol: [Close, MA, EMA, MACD] * lookback
        self.obs_cols = {}
        for symbol in self.symbols:
            self.obs_cols[symbol] = [f'Close_{symbol}', f'MA10_{symbol}', f'EMA20_{symbol}', f'MACD_{symbol}']

        obs_len = len(self.obs_cols[self.symbols[0]]) * len(self.symbols) * self.lookback_window_size
        obs_len += len(self.symbols)  # positions
        obs_len += 1  # balance

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.symbols),), dtype=np.float32)

        self.scaler = StandardScaler()
        self._prepare_data()
        self.reset()

    def _prepare_data(self):
        common_dates = set(self.dfs[self.symbols[0]].index)
        for df in self.dfs.values():
            common_dates &= set(df.index)
        self.dates = sorted(list(common_dates))
        assert len(self.dates) > self.lookback_window_size, "Not enough data for lookback window!"

    def reset(self, *, seed=None, options=None):
        try:
            self.current_step = self.lookback_window_size
            self.balance = self.initial_balance
            self.positions = np.zeros(len(self.symbols))
            obs = self._next_observation()
            info = {'portfolio_value': self._get_portfolio_value(self._get_prices())}
            print(f"[RESET] current_step: {self.current_step}, balance: {self.balance}, positions: {self.positions}")
            print(f"[RESET] Observation shape: {obs.shape}, dtype: {obs.dtype}")
            return obs, info
        except Exception as e:
            print(f"[RESET][EXCEPTION] {e}")
            raise

    def step(self, actions):
        try:
            prices = self._get_prices()
            prev_value = self._get_portfolio_value(prices)

            actions = np.clip(actions, -1, 1)

            for idx, action in enumerate(actions):
                if action > 0:
                    buy_amount = self.balance * action
                    self.balance -= buy_amount
                    self.positions[idx] += buy_amount / prices[idx]
                elif action < 0:
                    sell_amount = self.positions[idx] * abs(action)
                    self.balance += sell_amount * prices[idx]
                    self.positions[idx] -= sell_amount

            self.current_step += 1
            done = self.current_step >= len(self.dates) - 1
            terminated = done
            truncated = False
            prices = self._get_prices()
            current_value = self._get_portfolio_value(prices)
            reward = current_value - prev_value

            info = {'portfolio_value': current_value}

            obs = self._next_observation()
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"[STEP][WARNING] Observation contains NaN or Inf! {obs}")
            if np.isnan(reward) or np.isinf(reward):
                print(f"[STEP][WARNING] Reward is NaN or Inf! reward: {reward}")
            print(f"[STEP] current_step: {self.current_step}, done: {done}, reward: {reward}")
            print(f"[STEP] Observation shape: {obs.shape}, dtype: {obs.dtype}")
            print(f"[STEP] positions: {self.positions}, balance: {self.balance}")
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print(f"[STEP][EXCEPTION] {e}")
            raise

    def _get_prices(self):
        return np.array([
            self.dfs[symbol].loc[self.dates[self.current_step], f'Close_{symbol}']
            for symbol in self.symbols
        ])

    def _next_observation(self):
        frames = []
        for symbol in self.symbols:
            df = self.dfs[symbol]
            obs = df.iloc[self.current_step - self.lookback_window_size + 1: self.current_step + 1]
            frames.append(obs[self.obs_cols[symbol]].values)
        obs = np.concatenate(frames).flatten()
        obs = np.append(obs, self.positions)
        obs = np.append(obs, self.balance)
        obs = obs.astype(np.float32)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"[_NEXT_OBS][WARNING] Observation contains NaN or Inf! {obs}")
        return obs

    def _get_portfolio_value(self, prices):
        return self.balance + np.sum(self.positions * prices)
