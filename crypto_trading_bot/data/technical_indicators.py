import pandas as pd
import ta
import pandas_ta as pta

def add_technical_indicators(df):
    # RSI
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    # VWAP
    df['vwap'] = pta.vwap(df['high'], df['low'], df['close'], df['volume'])
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
    return df 