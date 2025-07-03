import ccxt
import pandas as pd
from config.api_config import BINANCE_API_KEY, BINANCE_API_SECRET
from data.technical_indicators import add_technical_indicators

class CryptoDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET,
            'enableRateLimit': True,
        })

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=1000):
        data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def fetch_multi_asset_data(self, symbols, timeframes, limit=1000):
        all_data = {}
        for symbol in symbols:
            all_data[symbol] = {}
            for tf in timeframes:
                df = self.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                df = add_technical_indicators(df)
                df = df.dropna().reset_index(drop=True)
                all_data[symbol][tf] = df
        return all_data

    def fetch_fear_greed_index(self):
        import requests
        response = requests.get("https://api.alternative.me/fng/")
        return response.json()

# Kullanım örneği:
# fetcher = CryptoDataFetcher()
# data = fetcher.fetch_multi_asset_data(
#     symbols=["BTC/USDT", "ETH/USDT", "BNB/USDT"],
#     timeframes=["1h", "4h", "1d"],
#     limit=500
# ) 