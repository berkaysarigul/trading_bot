import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from envs.multi_asset_env import MultiAssetTradingEnv
from utils.data_utils import prepare_multi_asset_data

SEMBOLLER = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"
LOOKBACK = 10
LOG_DIR = "./data/logs/"
MODEL_DIR = "./data/models/"
N_ENVS = 4
TOTAL_TIMESTEPS = 500_000
TRAIN_SPLIT_RATIO = 0.8
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(df_dict, **env_kwargs):
    def _init():
        env = MultiAssetTradingEnv(df_dict, **env_kwargs)
        return env
    return _init

def main():
    df_dict = prepare_multi_asset_data(SEMBOLLER, START_DATE, END_DATE)
    dates = sorted(list(set.intersection(*[set(df.index) for df in df_dict.values()])))
    train_size = int(len(dates) * TRAIN_SPLIT_RATIO)
    train_dates = dates[:train_size]
    val_dates = dates[train_size:]
    train_df_dict = {sym: df.loc[train_dates] for sym, df in df_dict.items()}
    val_df_dict = {sym: df.loc[val_dates] for sym, df in df_dict.items()}
    # Vektör ortamlar
    train_env = SubprocVecEnv([make_env(train_df_dict, lookback_window_size=LOOKBACK) for _ in range(N_ENVS)])
    val_env = SubprocVecEnv([make_env(val_df_dict, lookback_window_size=LOOKBACK) for _ in range(1)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)
    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
        )
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, tb_log_name="PPO_MultiAsset", progress_bar=True)
    model.save(os.path.join(MODEL_DIR, "final_multiasset_model.zip"))
    train_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print("\n✅ Eğitim tamamlandı ve model kaydedildi!")

if __name__ == "__main__":
    main() 