import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(l) for l in col if l]) for col in df.columns.values]
    if df.empty:
        return None
    df = df.rename(columns={
        'Close': f'Close_{symbol}',
        'Open': f'Open_{symbol}',
        'High': f'High_{symbol}',
        'Low': f'Low_{symbol}',
        'Volume': f'Volume_{symbol}'
    })
    return df

def add_technical_indicators(df, symbol):
    close = df[f'Close_{symbol}']
    df[f'MA10_{symbol}'] = close.rolling(window=10).mean()
    df[f'EMA20_{symbol}'] = close.ewm(span=20, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df[f'MACD_{symbol}'] = ema12 - ema26
    return df

def prepare_multi_asset_data(symbols, start, end):
    df_dict = {}
    for symbol in symbols:
        df = fetch_data(symbol, start, end)
        if df is None or df.empty:
            continue
        df = add_technical_indicators(df, symbol)
        df = df.dropna()
        df_dict[symbol] = df
    # Ortak tarihleri bul ve hizala
    if not df_dict:
        raise ValueError('Hiç veri indirilemedi!')
    common_dates = set.intersection(*[set(df.index) for df in df_dict.values()])
    for symbol in df_dict:
        df_dict[symbol] = df_dict[symbol].loc[list(common_dates)].sort_index()
    return df_dict

def fit_scalers(df_dict, obs_cols):
    scalers = {}
    for symbol, df in df_dict.items():
        scaler = StandardScaler()
        scaler.fit(df[obs_cols[symbol]])
        scalers[symbol] = scaler
    return scalers 