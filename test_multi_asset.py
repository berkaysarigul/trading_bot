import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.multi_asset_env import MultiAssetTradingEnv
from utils.data_utils import prepare_multi_asset_data

SEMBOLLER = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
START_DATE = "2024-01-02"
END_DATE = "2024-07-01"
LOOKBACK = 10
MODEL_DIR = "./data/models/"
LOG_DIR = "./data/logs/"
PORTFOLIO_CSV = "./data/portfolio_log.csv"
BALANCE_PNG = "./data/portfolio_balance.png"

os.makedirs("./data", exist_ok=True)

def main():
    df_dict = prepare_multi_asset_data(SEMBOLLER, START_DATE, END_DATE)
    env = DummyVecEnv([lambda: MultiAssetTradingEnv(df_dict, lookback_window_size=LOOKBACK)])
    # VecNormalize yükle
    vecnorm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
    model_path = os.path.join(MODEL_DIR, "final_multiasset_model.zip")
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    done = False
    portfolio_values = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info[0]["portfolio_value"])
        if done[0]:
            break
    # CSV log
    pd.DataFrame({"portfolio_value": portfolio_values}).to_csv(PORTFOLIO_CSV, index=False)
    # Çizim
    plt.figure(figsize=(10,5))
    plt.plot(portfolio_values)
    plt.title("Portföy Büyüklüğü")
    plt.xlabel("Adım")
    plt.ylabel("Portföy Değeri")
    plt.tight_layout()
    plt.savefig(BALANCE_PNG)
    print(f"Test tamamlandı. Log: {PORTFOLIO_CSV}, Grafik: {BALANCE_PNG}")

if __name__ == "__main__":
    main() 