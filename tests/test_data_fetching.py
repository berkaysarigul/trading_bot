import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_trading_bot.data.crypto_data_fetcher import CryptoDataFetcher

def test_binance_fetch():
    fetcher = CryptoDataFetcher()
    df = fetcher.fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=10)
    assert df is not None and not df.empty, "Veri çekilemedi veya DataFrame boş!"
    print("Binance veri çekme başarılı!")

if __name__ == "__main__":
    test_binance_fetch() 