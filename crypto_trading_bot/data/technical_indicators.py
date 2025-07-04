import pandas as pd
import ta
import numpy as np

def add_technical_indicators(df):
    """Teknik göstergeleri ekle"""
    df = df.copy()
    
    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    
    # VWAP (pandas_ta yerine manuel hesaplama)
    df['vwap'] = calculate_vwap(df)
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # ADX
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    # CCI
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    # MFI
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # EMA'lar
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    
    # SMA'lar
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    
    return df

def calculate_vwap(df):
    """VWAP hesapla (pandas_ta olmadan)"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap 