import numpy as np
from scipy.stats import norm

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate
    return np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252)

def calmar_ratio(annual_return, max_drawdown):
    return annual_return / (abs(max_drawdown) + 1e-8)

def sortino_ratio(returns, risk_free_rate=0.0):
    downside = returns[returns < 0]
    return (np.mean(returns) - risk_free_rate) / (np.std(downside) + 1e-8) * np.sqrt(252)

def win_loss_ratio(returns):
    wins = np.sum(returns > 0)
    losses = np.sum(returns < 0)
    return wins / (losses + 1e-8)

def profit_factor(returns):
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = -np.sum(returns[returns < 0])
    return gross_profit / (gross_loss + 1e-8)

def recovery_factor(net_profit, max_drawdown):
    return net_profit / (abs(max_drawdown) + 1e-8)

def alpha_beta(returns, benchmark_returns):
    cov = np.cov(returns, benchmark_returns)
    beta = cov[0, 1] / (np.var(benchmark_returns) + 1e-8)
    alpha = np.mean(returns) - beta * np.mean(benchmark_returns)
    return alpha, beta

def value_at_risk(returns, confidence_level=0.95):
    """
    Parametrik (standart sapma ve normal dağılım varsayımı ile) Value at Risk (VaR) hesaplar.
    returns: getiri serisi (numpy array veya pandas Series)
    confidence_level: Güven düzeyi (örn. 0.95)
    """
    mean = np.mean(returns)
    std = np.std(returns)
    var = norm.ppf(1 - confidence_level, mean, std)
    return abs(var) 