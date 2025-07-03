import numpy as np

def run_backtest(env, agent, episodes=1):
    all_rewards = []
    all_portfolio_values = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        rewards = []
        portfolio_values = []
        while not done:
            action, _, _ = agent.act(obs)  # agent act fonksiyonu: (obs) -> (action, value, hidden)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)
            portfolio_values.append(env._get_portfolio_value({s: env.data[s][env.timeframes[0]].iloc[env.current_step]['close'] for s in env.symbols}))
        all_rewards.append(rewards)
        all_portfolio_values.append(portfolio_values)
    return all_rewards, all_portfolio_values

def walk_forward_analysis(env_class, agent_class, data_splits):
    results = []
    for train_data, test_data in data_splits:
        env = env_class(train_data)
        agent = agent_class()
        # agent.train(env)  # Eğitim fonksiyonu burada çağrılır
        test_env = env_class(test_data)
        rewards, portfolio_values = run_backtest(test_env, agent)
        results.append((rewards, portfolio_values))
    return results

def monte_carlo_simulation(env_class, agent_class, data, n_simulations=50, window_size=200):
    results = []
    for i in range(n_simulations):
        start = np.random.randint(0, max(1, len(next(iter(data.values()))[next(iter(next(iter(data.values())).keys()))]) - window_size))
        sim_data = {}
        for symbol, dfs in data.items():
            sim_data[symbol] = {}
            for tf, df in dfs.items():
                sim_data[symbol][tf] = df.iloc[start:start+window_size].reset_index(drop=True)
        env = env_class(sim_data)
        agent = agent_class()
        rewards, portfolio_values = run_backtest(env, agent)
        results.append((rewards, portfolio_values))
    return results

def stress_test(env_class, agent_class, data, market_type='bear', window_size=200):
    # market_type: 'bear', 'bull', 'sideways'
    # Fiyat trendine göre veri seçimi
    results = []
    for symbol, dfs in data.items():
        for tf, df in dfs.items():
            if market_type == 'bear':
                mask = df['close'].diff(window_size) < 0
            elif market_type == 'bull':
                mask = df['close'].diff(window_size) > 0
            else:  # sideways
                mask = abs(df['close'].diff(window_size)) < (0.01 * df['close'])
            idx = np.where(mask)[0]
            for i in idx:
                if i+window_size < len(df):
                    test_data = {symbol: {tf: df.iloc[i:i+window_size].reset_index(drop=True)}}
                    env = env_class(test_data)
                    agent = agent_class()
                    rewards, portfolio_values = run_backtest(env, agent)
                    results.append((rewards, portfolio_values))
    return results 