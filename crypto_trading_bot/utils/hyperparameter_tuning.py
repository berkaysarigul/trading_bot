import optuna
import numpy as np
from models.ppo_lstm_model import PPO_LSTM_Model
from environments.crypto_env import CryptoTradingEnv
from utils.backtesting_engine import run_backtest

# Basit bir tuning örneği (dummy agent yerine gerçek agent ile entegre edilebilir)
def objective(trial, env, input_dim, action_dim):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    dropout_p = trial.suggest_float('dropout_p', 0.1, 0.5)
    # Modeli oluştur
    model = PPO_LSTM_Model(input_dim, action_dim, hidden_dim=hidden_dim, dropout_p=dropout_p)
    # Dummy agent örneği (gerçek eğitim burada yapılmalı)
    class DummyAgent:
        def act(self, obs):
            action = np.random.uniform(-1, 1, size=(action_dim,))
            return action, None, None
    agent = DummyAgent()
    rewards, portfolio_values = run_backtest(env, agent, episodes=1)
    returns = np.array(portfolio_values[0])
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
    return sharpe  # Maksimize edilecek metrik

def run_optuna_tuning(env, input_dim, action_dim, n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, env, input_dim, action_dim), n_trials=n_trials)
    print('En iyi hiperparametreler:', study.best_params)
    return study.best_params

# Kullanım örneği:
# env = CryptoTradingEnv(data)
# input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# run_optuna_tuning(env, input_dim, action_dim) 