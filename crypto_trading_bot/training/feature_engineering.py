import pandas as pd
import numpy as np
import ta
from data.technical_indicators import add_technical_indicators
import pickle

class AdvancedFeatureEngineering:
    def __init__(self, lookback_window=50, technical_indicators=True, market_regime_features=True, sentiment_features=False, risk_features=True, feature_selection=True):
        self.lookback_window = lookback_window
        self.technical_indicators = technical_indicators
        self.market_regime_features = market_regime_features
        self.sentiment_features = sentiment_features
        self.risk_features = risk_features
        self.feature_selection = feature_selection
        self.feature_columns = []
        
    def add_price_features(self, df):
        """Fiyat tabanlı özellikler"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['realized_volatility'] = df['returns'].rolling(window=20).apply(lambda x: np.sqrt(np.sum(x**2)))
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Price levels
        df['price_position'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        return df
    
    def add_volume_features(self, df):
        """Hacim tabanlı özellikler"""
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volume profile
        df['volume_price_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        df['volume_weighted_avg_price'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
        
        # Money Flow
        df['money_flow_index'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
        
        return df
    
    def add_trend_features(self, df):
        """Trend tabanlı özellikler"""
        # ADX
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['di_plus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_pos()
        df['di_minus'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_neg()
        
        # Parabolic SAR
        df['parabolic_sar'] = ta.trend.PSARIndicator(df['high'], df['low'], df['close']).psar()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        return df
    
    def add_oscillator_features(self, df):
        """Oscillator tabanlı özellikler"""
        # RSI variations
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_21'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        df['rsi_30'] = ta.momentum.RSIIndicator(df['close'], window=30).rsi()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # CCI
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        return df
    
    def add_volatility_features(self, df):
        """Volatilite tabanlı özellikler"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_middle'] = kc.keltner_channel_mband()
        df['kc_lower'] = kc.keltner_channel_lband()
        
        return df
    
    def add_time_features(self, df):
        """Zaman tabanlı özellikler"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_cross_asset_features(self, data_dict):
        """Cross-asset özellikler (korelasyon, relative strength)"""
        # Tüm coinlerin close fiyatlarını al
        close_prices = {}
        for symbol, timeframes in data_dict.items():
            if '1h' in timeframes:  # 1h timeframe'i kullan
                close_prices[symbol] = timeframes['1h']['close']
        close_df = pd.DataFrame(close_prices)
        # Rolling correlation matrix
        corr_window = 24  # 24 saat
        for symbol in close_prices.keys():
            correlations = []
            for i in range(corr_window, len(close_df)):
                corr_matrix = close_df.iloc[i-corr_window:i].corr()
                # Diğer coinlerle korelasyonun ortalaması
                other_corrs = [corr_matrix.loc[symbol, col] for col in corr_matrix.columns if col != symbol]
                correlations.append(np.mean(other_corrs))
            # NaN değerleri doldur
            correlations = [0] * corr_window + correlations
            data_dict[symbol]['1h']['cross_correlation'] = correlations
        # Relative strength
        for symbol in close_prices.keys():
            # BTC'ye göre relative strength
            if 'BTC/USDT' in close_prices:
                btc_returns = close_prices['BTC/USDT'].pct_change()
                symbol_returns = close_prices[symbol].pct_change()
                relative_strength = symbol_returns - btc_returns
                data_dict[symbol]['1h']['relative_strength_btc'] = relative_strength
        # --- EK: Cross-asset feature'larda kalan NaN'ları temizle ---
        for symbol in data_dict:
            if '1h' in data_dict[symbol]:
                df = data_dict[symbol]['1h']
                # Sadece cross-asset feature'ları doldur
                for col in ['cross_correlation', 'relative_strength_btc']:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                data_dict[symbol]['1h'] = df
        return data_dict
    
    def engineer_features(self, data_dict):
        """Ana feature engineering pipeline'ı"""
        print("Starting advanced feature engineering...")
        for symbol in data_dict.keys():
            for timeframe in data_dict[symbol].keys():
                df = data_dict[symbol][timeframe].copy()
                print(f"[FE-ÖNCE] {symbol} {timeframe}: satır={len(df)}, NaN={df.isna().sum().sum()}")
                # Temel teknik göstergeler
                # df = add_technical_indicators(df)  # Eğer varsa
                df = self.add_price_features(df)
                df = self.add_volume_features(df)
                df = self.add_volatility_features(df)
                df = self.add_oscillator_features(df)
                df = self.add_trend_features(df)
                # Cross-asset
                if hasattr(self, 'add_cross_asset_features'):
                    df = self.add_cross_asset_features(df)
                print(f"[FE-SONRA] {symbol} {timeframe}: satır={len(df)}, NaN={df.isna().sum().sum()}")
                # Kolon tiplerini logla
                print(f"[FE-DTYPE] {symbol} {timeframe}: {[f'{col}: {df[col].dtype}' for col in df.columns]}")
                # float olmayan kolonları bul
                non_float_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.floating)]
                # Özellikle timestamp ve fear_greed'i otomatik drop et
                for col in non_float_cols:
                    if col == 'timestamp':
                        df = df.drop(columns=[col])
                    elif col == 'fear_greed':
                        # Eğer float'a çevrilebiliyorsa çevir, yoksa drop et
                        try:
                            df[col] = df[col].astype(np.float32)
                        except Exception:
                            df = df.drop(columns=[col])
                # Kalan float olmayan kolonları tekrar kontrol et
                non_float_cols = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.floating)]
                if non_float_cols:
                    print(f"[FE-ERROR] {symbol} {timeframe}: float olmayan kolonlar: {non_float_cols}")
                    raise TypeError(f"{symbol} {timeframe}: float olmayan kolonlar: {non_float_cols}")
                # NaN temizle (burada değil, zincir sonunda yapacağız)
                data_dict[symbol][timeframe] = df
                print(f"✓ {symbol} {timeframe}: {len(df.columns)} features")
        # Feature engineering zinciri sonunda, tüm semboller ve timeframeler için feature isimleri/sayısı eşit mi kontrol et
        feature_sets = []
        for symbol in data_dict:
            for timeframe in data_dict[symbol]:
                cols = [col for col in data_dict[symbol][timeframe].columns if np.issubdtype(data_dict[symbol][timeframe][col].dtype, np.floating)]
                feature_sets.append(tuple(cols))
        if len(set(feature_sets)) != 1:
            raise ValueError(f"Tüm semboller ve timeframeler için feature isimleri/sayısı eşit olmalı! Feature sets: {feature_sets}")
        # --- EK: Zincir sonunda dropna ve reset_index ---
        for symbol in data_dict:
            for timeframe in data_dict[symbol]:
                df = data_dict[symbol][timeframe]
                nan_before = df.isna().sum().sum()
                if nan_before > 0:
                    print(f"[FE-FINAL-DROPNA] {symbol} {timeframe}: satır={len(df)}, NaN={nan_before}")
                df = df.dropna().reset_index(drop=True)
                nan_after = df.isna().sum().sum()
                print(f"[FE-FINAL] {symbol} {timeframe}: satır={len(df)}, NaN={nan_after}")
                data_dict[symbol][timeframe] = df
        print("Feature engineering completed!")
        return data_dict

    def create_features(self, data_dict):
        """Pipeline ile uyumlu ana feature engineering fonksiyonu."""
        return self.engineer_features(data_dict)

    def select_features(self, feature_data):
        """Şimdilik feature selection yapılmıyor, veriyi olduğu gibi döndürür."""
        return feature_data

    def split_data(self, feature_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Zaman serisi verisini train/val/test olarak böler."""
        train_data = {}
        val_data = {}
        test_data = {}
        for symbol in feature_data:
            train_data[symbol] = {}
            val_data[symbol] = {}
            test_data[symbol] = {}
            for tf in feature_data[symbol]:
                df = feature_data[symbol][tf]
                total_length = len(df)
                train_end = int(total_length * train_ratio)
                val_end = int(total_length * (train_ratio + val_ratio))
                train_data[symbol][tf] = df.iloc[:train_end].reset_index(drop=True)
                val_data[symbol][tf] = df.iloc[train_end:val_end].reset_index(drop=True)
                test_data[symbol][tf] = df.iloc[val_end:].reset_index(drop=True)
                print(f"[SPLIT] {symbol} {tf}: total={total_length}, train={len(train_data[symbol][tf])}, val={len(val_data[symbol][tf])}, test={len(test_data[symbol][tf])}")
                # float32'ye zorla
                try:
                    train_data[symbol][tf] = train_data[symbol][tf].astype(np.float32)
                    val_data[symbol][tf] = val_data[symbol][tf].astype(np.float32)
                    test_data[symbol][tf] = test_data[symbol][tf].astype(np.float32)
                except Exception as e:
                    print(f"[SPLIT-ERROR] {symbol} {tf}: float32'ye çevrilemeyen kolonlar! {e}")
                    raise
                # NaN kontrolü
                for name, dset in zip(['train','val','test'], [train_data[symbol][tf], val_data[symbol][tf], test_data[symbol][tf]]):
                    nan_cols = dset.columns[dset.isna().any()].tolist()
                    if dset.isna().sum().sum() > 0:
                        print(f"[SPLIT-WARN] {symbol} {tf} {name}: NaN bulunan kolonlar: {nan_cols}")
                        raise ValueError(f"[SPLIT-ERROR] {symbol} {tf} {name}: Split sonrası NaN bulundu! Kolonlar: {nan_cols}")
        return train_data, val_data, test_data

    def save_features(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_features(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_feature_names(self):
        """Tüm semboller ve timeframe'ler için unique feature isimlerini döndürür."""
        feature_names = set()
        # Feature engineering sırasında self.feature_columns güncellenmiş olmalı
        if hasattr(self, 'feature_columns') and self.feature_columns:
            for col in self.feature_columns:
                feature_names.add(col)
        else:
            # Eğer feature_columns yoksa, veri üzerinden çıkar
            # (Bu fallback, feature engineering fonksiyonunda feature_columns güncellenmediyse çalışır)
            return []
        return list(feature_names)

# Kullanım örneği:
# feature_eng = AdvancedFeatureEngineering()
# enhanced_data = feature_eng.engineer_features(data_dict) 