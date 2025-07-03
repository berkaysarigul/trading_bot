import numpy as np
import pandas as pd

def kelly_position_sizing(balance, win_rate, win_loss_ratio):
    kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0, min(balance * kelly_fraction, balance))

def volatility_position_sizing(balance, volatility, risk_factor=0.01):
    # Volatilite arttıkça pozisyon küçülür
    size = balance * risk_factor / (volatility + 1e-8)
    return min(size, balance)

def position_sizing(balance, risk_per_trade, stop_loss_pct):
    # Basit risk tabanlı sizing (örnek)
    return min(balance * risk_per_trade, balance * stop_loss_pct)

def stop_loss_take_profit(entry_price, stop_loss_pct, take_profit_pct):
    stop_loss = entry_price * (1 - stop_loss_pct)
    take_profit = entry_price * (1 + take_profit_pct)
    return stop_loss, take_profit

def max_drawdown_protection(portfolio_values, max_dd=0.15):
    # Maksimum drawdown %15'i aşarsa pozisyonları küçült
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / (peak + 1e-8)
    if np.min(drawdown) < -max_dd:
        return True  # Pozisyon küçült
    return False

def filter_highly_correlated_assets(df_dict, threshold=0.9):
    # Her bir coin için kapanış fiyatlarını al
    closes = {symbol: dfs[list(dfs.keys())[0]]['close'] for symbol, dfs in df_dict.items()}
    close_df = pd.DataFrame(closes)
    corr = close_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    filtered = {k: v for k, v in df_dict.items() if k not in to_drop}
    return filtered

def value_at_risk(returns, alpha=0.05):
    # Tarihsel VaR
    var = np.percentile(returns, 100 * alpha)
    return abs(var)

def estimate_slippage(order_size, avg_volume, slippage_factor=0.1):
    """
    Emir büyüklüğü ve ortalama hacme göre slippage yüzdesi hesaplar.
    slippage_factor: 0.1 = %10 hacim için %1 slippage gibi
    """
    ratio = order_size / (avg_volume + 1e-8)
    slippage = slippage_factor * ratio
    return min(slippage, 0.05)  # Maksimum %5 slippage

def filter_low_liquidity_assets(df_dict, min_volume=100000):
    filtered = {}
    for symbol, dfs in df_dict.items():
        df = dfs[list(dfs.keys())[0]]
        if df['volume'].mean() >= min_volume:
            filtered[symbol] = dfs
    return filtered 