import unittest
from crypto_trading_bot.data.crypto_data_fetcher import CryptoDataFetcher

class TestData(unittest.TestCase):
    def test_fetch_ohlcv(self):
        fetcher = CryptoDataFetcher()
        df = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=10)
        self.assertFalse(df.empty)
        self.assertIn('close', df.columns)

if __name__ == '__main__':
    unittest.main() 