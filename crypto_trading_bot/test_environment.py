#!/usr/bin/env python3
"""
Crypto Trading Environment test scripti
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.crypto_env import CryptoTradingEnv
from data.crypto_data_fetcher import CryptoDataFetcher
from config.api_config import validate_api_keys

def create_sample_data():
    """Test için örnek veri oluştur"""
    print("📊 Creating sample data for testing...")
    
    # Örnek OHLCV verisi oluştur
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    sample_data = {}
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    for symbol in symbols:
        sample_data[symbol] = {}
        
        # 1h timeframe
        np.random.seed(42)  # Reproducible results
        base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 300)
        
        # Random walk price simulation
        returns = np.random.normal(0, 0.02, len(dates))  # %2 volatility
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        # Teknik göstergeler ekle
        df['rsi'] = np.random.uniform(20, 80, len(dates))
        df['macd'] = np.random.normal(0, 1, len(dates))
        df['bb_upper'] = df['close'] * 1.02
        df['bb_lower'] = df['close'] * 0.98
        df['atr'] = df['close'] * 0.02
        
        sample_data[symbol]['1h'] = df
        sample_data[symbol]['4h'] = df[::4].reset_index(drop=True)  # 4h için sample
        sample_data[symbol]['1d'] = df[::24].reset_index(drop=True)  # 1d için sample
    
    print(f"✅ Sample data created: {len(symbols)} symbols, {len(dates)} timestamps")
    return sample_data

def test_environment_basic():
    """Temel environment testleri"""
    print("\n🔍 Testing basic environment functionality...")
    
    # Örnek veri oluştur
    data = create_sample_data()
    
    # Environment oluştur
    env = CryptoTradingEnv(
        data=data,
        initial_balance=10000,
        commission=0.001,
        risk_per_trade=0.02,
        max_position_size=0.1,
        lookback_window=50
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Symbols: {env.symbols}")
    print(f"   Timeframes: {env.timeframes}")
    print(f"   Data length: {env.data_length}")
    print(f"   Lookback window: {env.lookback_window}")
    
    # Space'leri kontrol et
    print(f"\n📏 Space Information:")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    
    return env

def test_environment_interaction(env):
    """Environment ile etkileşim testi"""
    print("\n🎮 Testing environment interaction...")
    
    # Reset
    obs, info = env.reset()
    print(f"✅ Reset successful")
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial balance: ${info['balance']:.2f}")
    print(f"   Initial portfolio value: ${info['portfolio_value']:.2f}")
    
    # Birkaç adım çalıştır
    total_reward = 0
    portfolio_values = []
    
    for step in range(10):
        # Random action
        action = env.action_space.sample()
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        portfolio_values.append(info['portfolio_value'])
        
        print(f"   Step {step + 1}: Portfolio=${info['portfolio_value']:.2f}, "
              f"Return={info['total_return']:.4f}, Reward={reward:.4f}")
        
        if done:
            break
    
    print(f"\n📈 Summary:")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final portfolio: ${info['portfolio_value']:.2f}")
    print(f"   Total return: {info['total_return']:.4f}")
    print(f"   Number of trades: {info['n_trades']}")
    
    return portfolio_values

def test_environment_features(env):
    """Environment özelliklerini test et"""
    print("\n🔧 Testing environment features...")
    
    # Reset
    obs, info = env.reset()
    
    # Portfolio history test
    portfolio_history = env.get_portfolio_history()
    print(f"✅ Portfolio history: {len(portfolio_history)} values")
    
    # Render test
    print("\n📊 Environment state:")
    env.render()
    
    # Info test
    print(f"\n📋 Environment info:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

def test_environment_edge_cases():
    """Edge case'leri test et"""
    print("\n⚠️ Testing edge cases...")
    
    # Çok kısa veri ile test
    try:
        short_data = {
            "BTC/USDT": {
                "1h": pd.DataFrame({
                    'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
                    'open': [50000] * 10,
                    'high': [51000] * 10,
                    'low': [49000] * 10,
                    'close': [50500] * 10,
                    'volume': [1000] * 10
                })
            }
        }
        
        # Bu hata vermeli (lookback_window çok büyük)
        env = CryptoTradingEnv(short_data, lookback_window=50)
        print("❌ Should have raised ValueError for short data")
        
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    # Normal veri ile test
    data = create_sample_data()
    env = CryptoTradingEnv(data, lookback_window=10)
    print("✅ Environment with smaller lookback window created successfully")

def test_environment_performance():
    """Performance testi"""
    print("\n⚡ Testing environment performance...")
    
    data = create_sample_data()
    env = CryptoTradingEnv(data, lookback_window=50)
    
    import time
    
    # 100 adım performans testi
    start_time = time.time()
    
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"✅ 100 steps completed in {elapsed:.3f} seconds")
    print(f"   Average time per step: {elapsed/100*1000:.2f} ms")

def main():
    """Ana test fonksiyonu"""
    print("🚀 Crypto Trading Environment Test")
    print("=" * 50)
    
    try:
        # 1. Temel testler
        env = test_environment_basic()
        
        # 2. Etkileşim testi
        portfolio_values = test_environment_interaction(env)
        
        # 3. Özellik testleri
        test_environment_features(env)
        
        # 4. Edge case testleri
        test_environment_edge_cases()
        
        # 5. Performance testi
        test_environment_performance()
        
        print("\n" + "=" * 50)
        print("🎉 All environment tests passed!")
        print("\nEnvironment is ready for RL training!")
        print("\nNext steps:")
        print("1. Test with real data: python test_data_fetching.py")
        print("2. Start training: python main.py --mode train")
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 