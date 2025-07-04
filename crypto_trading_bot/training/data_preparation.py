import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.crypto_data_fetcher import CryptoDataFetcher
from data.market_data_processor import get_market_caps, apply_market_cap_weights, add_fear_greed_feature
import warnings
import pickle
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, symbols=None, timeframes=None, days_back=730):
        self.fetcher = CryptoDataFetcher()
        self.symbols = symbols if symbols is not None else [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", 
            "SOL/USDT", "DOT/USDT", "LINK/USDT", "AVAX/USDT"
        ]
        self.timeframes = timeframes if timeframes is not None else ["1h", "4h", "1d"]
        self.days_back = days_back
        
    def fetch_historical_data(self, days_back=None):  # 2 yıl
        """Çoklu coin ve timeframe için geçmiş veri çekme"""
        days_back = days_back if days_back is not None else self.days_back
        print(f"Fetching {days_back} days of historical data for {len(self.symbols)} symbols...")
        
        all_data = {}
        for symbol in self.symbols:
            all_data[symbol] = {}
            for tf in self.timeframes:
                try:
                    # Her timeframe için uygun limit hesapla
                    if tf == "1h":
                        limit = days_back * 24
                    elif tf == "4h":
                        limit = days_back * 6
                    else:  # 1d
                        limit = days_back
                    
                    df = self.fetcher.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                    all_data[symbol][tf] = df
                    print(f"✓ {symbol} {tf}: {len(df)} records")
                except Exception as e:
                    print(f"✗ Error fetching {symbol} {tf}: {e}")
                    continue
        
        return all_data
    
    def add_market_features(self, data):
        """Market cap, Fear & Greed, sentiment gibi market özelliklerini ekle"""
        print("Adding market features...")
        
        # Market cap ağırlıklandırması
        market_caps = get_market_caps(self.symbols)
        data = apply_market_cap_weights(data, market_caps)
        
        # Fear & Greed Index
        try:
            fng_data = self.fetcher.fetch_fear_greed_index()
            fng_value = fng_data['data'][0]['value']
            for symbol in self.symbols:
                for tf in self.timeframes:
                    if symbol in data and tf in data[symbol]:
                        data[symbol][tf] = add_fear_greed_feature(data[symbol][tf], fng_value)
        except Exception as e:
            print(f"Warning: Could not fetch Fear & Greed Index: {e}")
        
        return data
    
    def align_data(self, data):
        """Tüm coinlerin verilerini aynı tarih aralığında hizala"""
        print("Aligning data across all symbols...")
        
        # En kısa veri setinin uzunluğunu bul
        min_length = float('inf')
        for symbol in self.symbols:
            if symbol in data:
                for tf in self.timeframes:
                    if tf in data[symbol]:
                        min_length = min(min_length, len(data[symbol][tf]))
        
        # Tüm veri setlerini aynı uzunlukta kes
        aligned_data = {}
        for symbol in self.symbols:
            if symbol in data:
                aligned_data[symbol] = {}
                for tf in self.timeframes:
                    if tf in data[symbol]:
                        aligned_data[symbol][tf] = data[symbol][tf].tail(min_length).reset_index(drop=True)
        
        print(f"Aligned data length: {min_length}")
        return aligned_data
    
    def create_train_val_test_split(self, data, train_ratio=0.7, val_ratio=0.15):
        """Time series aware train/validation/test split"""
        print("Creating train/validation/test split...")
        
        total_length = len(next(iter(data.values()))[next(iter(next(iter(data.values())).keys()))])
        train_end = int(total_length * train_ratio)
        val_end = int(total_length * (train_ratio + val_ratio))
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        for symbol in self.symbols:
            if symbol in data:
                train_data[symbol] = {}
                val_data[symbol] = {}
                test_data[symbol] = {}
                
                for tf in self.timeframes:
                    if tf in data[symbol]:
                        df = data[symbol][tf]
                        train_data[symbol][tf] = df.iloc[:train_end].reset_index(drop=True)
                        val_data[symbol][tf] = df.iloc[train_end:val_end].reset_index(drop=True)
                        test_data[symbol][tf] = df.iloc[val_end:].reset_index(drop=True)
        
        print(f"Train: {train_end}, Val: {val_end-train_end}, Test: {total_length-val_end}")
        return train_data, val_data, test_data
    
    def prepare_training_data(self, days_back=730):
        """Ana veri hazırlama pipeline'ı"""
        print("Starting data preparation pipeline...")
        
        # 1. Geçmiş veri çek
        data = self.fetch_historical_data(days_back)
        
        # 2. Market özelliklerini ekle
        data = self.add_market_features(data)
        
        # 3. Verileri hizala
        data = self.align_data(data)
        
        # 4. Train/val/test split
        train_data, val_data, test_data = self.create_train_val_test_split(data)
        
        print("Data preparation completed!")
        return train_data, val_data, test_data

    def validate_data(self, data):
        # En az bir sembol ve timeframe'de veri var mı kontrolü
        if not data or not isinstance(data, dict):
            return False
        for symbol in data:
            for tf in data[symbol]:
                df = data[symbol][tf]
                if df is not None and not df.empty:
                    return True
        return False

    def save_data(self, data, path):
        """Veriyi pickle ile kaydeder."""
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, path):
        """Pickle ile kaydedilmiş veriyi yükler."""
        with open(path, 'rb') as f:
            return pickle.load(f)

# Kullanım örneği:
# data_prep = DataPreparation()
# train_data, val_data, test_data = data_prep.prepare_training_data(days_back=730) 