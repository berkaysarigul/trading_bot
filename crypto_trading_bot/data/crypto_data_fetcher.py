import ccxt
import pandas as pd
from crypto_trading_bot.config.api_config import BINANCE_API_KEY, BINANCE_API_SECRET
from crypto_trading_bot.data.technical_indicators import add_technical_indicators

class CryptoDataFetcher:
    def __init__(self):
        api_key = BINANCE_API_KEY if BINANCE_API_KEY != "YOUR_API_KEY" else None
        api_secret = BINANCE_API_SECRET if BINANCE_API_SECRET != "YOUR_API_SECRET" else None
        if api_key and api_secret:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
            })
        else:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
            })

    def fetch_ohlcv(self, symbol, timeframe='1h', limit=1000, since=None):
        all_data = []
        fetch_limit = 1000  # Binance API tek seferde max 1000 bar döner
        if since is None:
            # En güncel barın timestamp'ini bulmak için bir defa çek
            recent = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            if not recent or len(recent) == 0:
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            end_time = recent[-1][0]
        else:
            end_time = since
        fetched = 0
        while fetched < limit:
            current_limit = min(fetch_limit, limit - fetched)
            # Binance API'da 'since' parametresi başlangıç zamanıdır, bu yüzden geriye doğru çekmek için end_time'dan current_limit*bar arası çek
            start_time = end_time - current_limit * self.exchange.parse_timeframe(timeframe) * 1000
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=current_limit, since=int(start_time))
            if not data or len(data) == 0:
                break
            all_data = data + all_data  # başa ekle, eskiye doğru
            fetched += len(data)
            if len(data) < current_limit:
                break  # Daha fazla veri yok
            end_time = data[0][0] - 1  # bir önceki bloğun başı
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.tail(limit).reset_index(drop=True)

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