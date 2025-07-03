import gym
import numpy as np
import pandas as pd
from utils.risk_management import position_sizing, stop_loss_take_profit
from utils.performance_metrics import sharpe_ratio

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, commission=0.001, risk_per_trade=0.01):
        super().__init__()
        self.data = data  # dict: {symbol: {tf: df}}
        self.symbols = list(data.keys())
        self.timeframes = list(next(iter(data.values())).keys())
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.portfolio = {symbol: 0 for symbol in self.symbols}
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.history = []
        self.done = False
        self.max_steps = min([len(data[s][self.timeframes[0]]) for s in self.symbols])
        # action_space: continuous [-1, 1] for each asset (buy/sell/hold)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.symbols),), dtype=np.float32)
        # observation_space: tüm coinler ve teknik indikatörler
        obs_dim = sum([data[s][self.timeframes[0]].shape[1] for s in self.symbols])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = {symbol: 0 for symbol in self.symbols}
        self.history = []
        self.done = False
        return self._get_observation()

    def step(self, action):
        # Pozisyon büyüklüğü ve emir uygulama
        prices = {s: self.data[s][self.timeframes[0]].iloc[self.current_step]['close'] for s in self.symbols}
        prev_value = self._get_portfolio_value(prices)
        for i, symbol in enumerate(self.symbols):
            pos_size = position_sizing(self.balance, self.risk_per_trade, 0.02)  # örnek
            trade_amount = pos_size * float(action[i])
            cost = abs(trade_amount) * self.commission
            self.balance -= cost
            self.portfolio[symbol] += trade_amount / prices[symbol]
        self.current_step += 1
        prices = {s: self.data[s][self.timeframes[0]].iloc[self.current_step]['close'] for s in self.symbols}
        curr_value = self._get_portfolio_value(prices)
        portfolio_return = (curr_value - prev_value) / (prev_value + 1e-8)
        self.history.append(portfolio_return)
        # Reward bileşenleri
        sharpe = sharpe_ratio(np.array(self.history)) if len(self.history) > 2 else 0
        max_drawdown = self._max_drawdown()
        transaction_cost_penalty = -sum([abs(float(a)) for a in action]) * self.commission
        volatility_bonus = -np.std(self.history[-10:]) if len(self.history) > 10 else 0
        reward = (
            0.6 * portfolio_return +
            0.2 * sharpe +
            0.1 * max_drawdown +
            0.05 * transaction_cost_penalty +
            0.05 * volatility_bonus
        )
        if self.current_step >= self.max_steps - 1:
            self.done = True
        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        obs = []
        for s in self.symbols:
            obs.extend(self.data[s][self.timeframes[0]].iloc[self.current_step].values)
        return np.array(obs, dtype=np.float32)

    def _get_portfolio_value(self, prices):
        value = self.balance
        for s in self.symbols:
            value += self.portfolio[s] * prices[s]
        return value

    def _max_drawdown(self):
        if not self.history:
            return 0
        cum_returns = np.cumprod([1 + r for r in self.history])
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / (peak + 1e-8)
        return np.min(drawdown) 