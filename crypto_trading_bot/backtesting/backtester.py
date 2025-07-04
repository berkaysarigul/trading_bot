import os
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from datetime import datetime

# Ortam ve veri yolları
from crypto_trading_bot.envs.env_wrapper import make_env

MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/best_model.zip')
LOG_PATH = os.getenv('BACKTEST_LOG_PATH', 'backtest_results/')
PORTFOLIO_CSV = os.path.join(LOG_PATH, 'portfolio_log.csv')

os.makedirs(LOG_PATH, exist_ok=True)

def calculate_metrics(df):
    # Toplam kar/zarar
    total_pnl = df['portfolio_value'].iloc[-1] - df['portfolio_value'].iloc[0]
    # Günlük getiriler (varsayım: her step bir gün)
    returns = df['portfolio_value'].pct_change().dropna()
    # Sharpe oranı (risk-free rate = 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    # Maksimum drawdown
    roll_max = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - roll_max) / roll_max
    max_drawdown = drawdown.min()
    # Volatilite
    volatility = returns.std() * np.sqrt(252)
    # İşlem sayısı ve başarı oranı
    trades = df['action'].diff().fillna(0).abs()
    trade_count = int(trades.sum())
    # Başarı oranı için reward > 0 olan adımların oranı (yaklaşık)
    win_rate = (df['reward'] > 0).sum() / len(df)
    # Buy&Hold benchmark
    buyhold_return = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
    # Sonuçları sözlük olarak döndür
    return {
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'buyhold_return': buyhold_return
    }

def run_backtest(model_path=MODEL_PATH, log_path=LOG_PATH):
    env = DummyVecEnv([lambda: make_env(mode='backtest')])
    # policy_kwargs'ı sadeleştir
    policy_kwargs = {
        'net_arch': best_params['net_arch'],
        'activation_fn': best_params['activation_fn']
    }
    model = RecurrentPPO.load(model_path, env=env, policy_kwargs=policy_kwargs)
    obs = env.reset()
    done = False
    portfolio_values = []
    rewards = []
    actions = []
    prices = []
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info[0].get('portfolio_value', np.nan))
        rewards.append(reward[0])
        actions.append(action[0] if hasattr(action, '__getitem__') else action)
        prices.append(info[0].get('price', np.nan))
        steps += 1
        if done[0]:
            break
    df = pd.DataFrame({
        'step': np.arange(len(portfolio_values)),
        'portfolio_value': portfolio_values,
        'reward': rewards,
        'action': actions,
        'price': prices
    })
    df.to_csv(os.path.join(log_path, 'portfolio_log.csv'), index=False)
    # Metrikleri hesapla
    metrics = calculate_metrics(df)
    # Metrikleri CSV'ye kaydet
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(log_path, 'backtest_metrics.csv'), index=False)
    # Sonuçları ekrana yazdır
    print("\n--- Backtest Sonuçları ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    # Portföy değeri grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['portfolio_value'], label='RL Agent')
    # Buy&Hold benchmark
    initial_value = df['portfolio_value'].iloc[0]
    buyhold_curve = initial_value * (df['price'] / df['price'].iloc[0])
    plt.plot(df['step'], buyhold_curve, label='Buy & Hold', linestyle='--')
    plt.title('Backtest Portfolio Value')
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_path, 'portfolio_value.png'))
    plt.close()
    # Günlük getiri dağılımı
    returns = df['portfolio_value'].pct_change().dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(returns, bins=50, alpha=0.7)
    plt.title('Günlük Getiri Dağılımı')
    plt.xlabel('Getiri')
    plt.ylabel('Frekans')
    plt.grid()
    plt.savefig(os.path.join(log_path, 'returns_hist.png'))
    plt.close()
    print(f"\nBacktest tamamlandı. Sonuçlar {log_path} klasörüne kaydedildi.")

if __name__ == "__main__":
    run_backtest() 