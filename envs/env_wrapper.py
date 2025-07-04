from envs.multi_asset_env import MultiAssetTradingEnv

def make_env(dfs, lookback_window_size=10, initial_balance=10000):
    """
    RL ortamını başlatır. mode parametresi ile train/test/backtest gibi farklı modlar desteklenebilir.
    """
    env = MultiAssetTradingEnv(dfs, lookback_window_size, initial_balance)
    return env 