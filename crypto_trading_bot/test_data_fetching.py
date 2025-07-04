#!/usr/bin/env python3
"""
Veri çekme sistemini test etmek için basit script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.crypto_data_fetcher import CryptoDataFetcher
from config.api_config import validate_api_keys, load_api_config
import pandas as pd

def test_data_fetching():
    """Veri çekme sistemini test et"""
    print("🔍 Testing data fetching system...")
    
    # 1. API key kontrolü
    print("\n1. API Key Validation:")
    if validate_api_keys():
        print("✅ API keys are valid")
    else:
        print("❌ API keys are not set up!")
        print("   Please create a .env file with your Binance API credentials:")
        print("   BINANCE_API_KEY=your_api_key_here")
        print("   BINANCE_API_SECRET=your_api_secret_here")
        return False
    
    # 2. API config yükle
    print("\n2. Loading API Configuration:")
    config = load_api_config()
    print(f"✅ API config loaded: {list(config.keys())}")
    
    # 3. Data fetcher oluştur
    print("\n3. Creating Data Fetcher:")
    try:
        fetcher = CryptoDataFetcher()
        print("✅ Data fetcher created successfully")
    except Exception as e:
        print(f"❌ Error creating data fetcher: {e}")
        return False
    
    # 4. Tek coin veri çekme testi
    print("\n4. Testing Single Coin Data Fetching:")
    try:
        # BTC/USDT için son 100 veri
        df = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
        print(f"✅ BTC/USDT data fetched: {len(df)} records")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Columns: {list(df.columns)}")
        
        # İlk birkaç satırı göster
        print("\n   Sample data:")
        print(df.head(3).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error fetching BTC/USDT data: {e}")
        return False
    
    # 5. Çoklu coin veri çekme testi
    print("\n5. Testing Multi-Asset Data Fetching:")
    try:
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        timeframes = ["1h", "4h"]
        
        data = fetcher.fetch_multi_asset_data(symbols, timeframes, limit=50)
        
        print(f"✅ Multi-asset data fetched successfully")
        for symbol in symbols:
            for tf in timeframes:
                if symbol in data and tf in data[symbol]:
                    df = data[symbol][tf]
                    print(f"   {symbol} {tf}: {len(df)} records")
        
    except Exception as e:
        print(f"❌ Error fetching multi-asset data: {e}")
        return False
    
    # 6. Fear & Greed Index testi
    print("\n6. Testing Fear & Greed Index:")
    try:
        fng_data = fetcher.fetch_fear_greed_index()
        fng_value = fng_data['data'][0]['value']
        fng_classification = fng_data['data'][0]['value_classification']
        print(f"✅ Fear & Greed Index: {fng_value} ({fng_classification})")
        
    except Exception as e:
        print(f"❌ Error fetching Fear & Greed Index: {e}")
        return False
    
    print("\n🎉 All tests passed! Data fetching system is working correctly.")
    return True

def show_data_sample():
    """Örnek veri göster"""
    print("\n📊 Sample Data Analysis:")
    
    try:
        fetcher = CryptoDataFetcher()
        
        # BTC/USDT için son 1000 veri
        df = fetcher.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=1000)
        
        print(f"BTC/USDT Data Summary:")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"  Volume range: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
        
        # Son 5 günün verisi
        recent_data = df.tail(120)  # 5 gün * 24 saat
        print(f"\nLast 5 days statistics:")
        print(f"  Average price: ${recent_data['close'].mean():.2f}")
        print(f"  Price volatility: {recent_data['close'].std():.2f}")
        print(f"  Average volume: {recent_data['volume'].mean():.2f}")
        
        # Price change
        first_price = recent_data.iloc[0]['close']
        last_price = recent_data.iloc[-1]['close']
        price_change = ((last_price - first_price) / first_price) * 100
        print(f"  Price change: {price_change:.2f}%")
        
    except Exception as e:
        print(f"❌ Error in data analysis: {e}")

if __name__ == "__main__":
    print("🚀 Crypto Trading Bot - Data Fetching Test")
    print("=" * 50)
    
    # Ana test
    success = test_data_fetching()
    
    if success:
        # Ek analiz
        show_data_sample()
        
        print("\n" + "=" * 50)
        print("✅ Data fetching system is ready for training!")
        print("\nNext steps:")
        print("1. Run training pipeline: python main.py --mode train")
        print("2. Or test specific components individually")
    else:
        print("\n❌ Data fetching test failed!")
        print("Please check your API credentials and try again.") 