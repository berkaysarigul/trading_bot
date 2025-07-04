import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from envs.env_wrapper import make_env

def create_dummy_dfs():
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    data = {
        "Close_BTC": np.random.rand(100) * 50000 + 10000,
        "MA10_BTC": np.random.rand(100) * 50000 + 10000,
        "EMA20_BTC": np.random.rand(100) * 50000 + 10000,
        "MACD_BTC": np.random.randn(100),
    }
    df = pd.DataFrame(data, index=dates)
    dfs = {"BTC": df}
    return dfs

def test_environment():
    dfs = create_dummy_dfs()
    env = make_env(dfs)
    obs, info = env.reset()
    print("Ortam başarıyla resetlendi. Gözlem şekli:", np.shape(obs))
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step sonrası gözlem şekli: {np.shape(obs)}, Reward: {reward}, Terminated: {terminated}, Info: {info}")
    assert obs is not None, "Gözlem None döndü!"
    assert isinstance(reward, (float, int, np.floating, np.integer)), "Reward tipi yanlış!"
    assert isinstance(terminated, (bool, np.bool_)), "Terminated tipi yanlış!"
    print("Environment testleri başarıyla geçti!")

if __name__ == "__main__":
    test_environment() 