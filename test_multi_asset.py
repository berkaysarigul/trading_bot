# test_multi_asset.py
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import yfinance as yf
from envs.multi_asset_env import MultiAssetTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pandas as pd
from utils.data_utils import prepare_multi_asset_data

MODEL_DIR = "./models/"
vecnorm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
dfs = prepare_multi_asset_data(symbols, "2023-01-01", "2023-12-31")

env = DummyVecEnv([lambda: MultiAssetTradingEnv(dfs)])
if os.path.exists(vecnorm_path):
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

model = PPO.load(os.path.join("data/models", "ppo_multi_asset_model"))

obs = env.reset()
portfolio_values = []
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(info[0]['portfolio_value'])
    if done[0]:
        break

pd.DataFrame({'PortfolioValue': portfolio_values}).to_csv("data/portfolio_log.csv", index=False)
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.savefig("data/portfolio_balance.png")
plt.close()