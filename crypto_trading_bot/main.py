from data.crypto_data_fetcher import CryptoDataFetcher
from data.technical_indicators import add_technical_indicators
from data.market_data_processor import (
    get_market_caps, apply_market_cap_weights, add_fear_greed_feature,
    add_twitter_sentiment_feature, add_reddit_sentiment_feature, add_news_sentiment_feature,
    filter_highly_correlated_assets
)
from environments.crypto_env import CryptoTradingEnv
from models.ppo_lstm_model import PPO_LSTM_Model
from utils.risk_management import value_at_risk
from utils.backtesting_engine import run_backtest
from utils.performance_metrics import sharpe_ratio, profit_factor
from monitoring.dashboard import show_dashboard
from monitoring.report_generator import generate_report
import numpy as np
import torch

# 1. Veri çekme ve feature engineering
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
timeframes = ["1h"]
fetcher = CryptoDataFetcher()
data = fetcher.fetch_multi_asset_data(symbols, timeframes, limit=500)

# 2. Market cap ağırlıklandırması ve Fear & Greed
market_caps = get_market_caps(symbols)
data = apply_market_cap_weights(data, market_caps)
fng = fetcher.fetch_fear_greed_index()['data'][0]['value']
for s in symbols:
    for tf in timeframes:
        data[s][tf] = add_fear_greed_feature(data[s][tf], fng)
        data[s][tf] = add_twitter_sentiment_feature(data[s][tf], 0.0)  # Dummy
        data[s][tf] = add_reddit_sentiment_feature(data[s][tf], 0.0)   # Dummy
        data[s][tf] = add_news_sentiment_feature(data[s][tf], 0.0)     # Dummy

# 3. Korelasyon analizi ve risk filtreleme
data = filter_highly_correlated_assets(data, threshold=0.9)

# 4. RL ortamı oluşturma
env = CryptoTradingEnv(data)

# 5. Model seçimi (örnek: PPO + LSTM)
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
model = PPO_LSTM_Model(input_dim, action_dim)

# Dummy agent (örnek)
class DummyAgent:
    def act(self, obs):
        action = np.random.uniform(-1, 1, size=(action_dim,))
        return action, None, None
agent = DummyAgent()

# 6. Backtest ve performans metrikleri
rewards, portfolio_values = run_backtest(env, agent, episodes=1)
returns = np.array(portfolio_values[0])
metrics = {
    "Sharpe": sharpe_ratio(returns),
    "Profit Factor": profit_factor(returns),
    "VaR": value_at_risk(returns)
}

# 7. Dashboard ve rapor
import pandas as pd
trade_log = pd.DataFrame({
    'portfolio_value': returns
})
show_dashboard(trade_log, metrics)
generate_report(trade_log, metrics)

# 8. Canlı trade başlatma (dummy agent ile)
# from trading.live_trader import LiveTrader
# live_trader = LiveTrader(agent, symbols)
# live_trader.run(lambda: env._get_observation()) 