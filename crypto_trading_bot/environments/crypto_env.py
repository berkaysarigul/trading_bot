import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
from utils.risk_management import position_sizing, stop_loss_take_profit
from utils.performance_metrics import sharpe_ratio

class CryptoTradingEnv(gym.Env):
    """
    Çoklu varlık kripto trading ortamı
    PPO + LSTM için optimize edilmiş
    """
    
    def __init__(self, 
                 data: Dict[str, Dict[str, pd.DataFrame]],
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 risk_per_trade: float = 0.02,
                 max_position_size: float = 0.1,
                 lookback_window: int = 50,
                 reward_scaling: float = 1.0):
        """
        Args:
            data: {symbol: {timeframe: DataFrame}} formatında veri
            initial_balance: Başlangıç bakiyesi
            commission: İşlem komisyonu (0.001 = %0.1)
            risk_per_trade: Trade başına maksimum risk
            max_position_size: Maksimum pozisyon büyüklüğü
            lookback_window: Geçmiş veri penceresi
            reward_scaling: Reward ölçeklendirme faktörü
        """
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        
        # Veri hazırlığı
        self.symbols = list(data.keys())
        self.timeframes = list(data[self.symbols[0]].keys())
        self.primary_timeframe = self.timeframes[0]  # Ana timeframe
        
        # Veri uzunluğu kontrolü
        self.data_length = len(data[self.symbols[0]][self.primary_timeframe])
        if self.data_length < lookback_window + 1:
            raise ValueError(f"Data length ({self.data_length}) must be > lookback_window + 1 ({lookback_window + 1})")
        
        # State ve action space tanımla
        self._setup_spaces()
        
        # Ortam durumu
        self.reset()
    
    def _setup_spaces(self):
        """Observation ve action space'lerini tanımla"""
        # Tüm semboller ve timeframe'ler için feature sayısı ve isimleri kontrolü
        feature_sets = []
        for symbol in self.symbols:
            df = self.data[symbol][self.primary_timeframe]
            float_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.floating)]
            feature_sets.append(tuple(float_cols))
        if len(set(feature_sets)) != 1:
            raise ValueError(f"Tüm semboller için feature isimleri/sayısı eşit olmalı! Feature sets: {feature_sets}")
        feature_count = len(feature_sets[0])
        obs_shape = (self.lookback_window * len(self.symbols) * feature_count,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Action space: Her symbol için [-1, 1] arası pozisyon ağırlığı
        # -1: Tam short, 0: Nötr, 1: Tam long
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.symbols),),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Ortamı sıfırla"""
        super().reset(seed=seed)
        
        # Trading durumunu sıfırla
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.trade_history = []
        self.daily_returns = []
        self.max_portfolio_value = self.initial_balance
        
        # İlk observation'ı al
        observation = self._get_observation()
        info = self._get_info()
        
        # NaN ve dtype kontrolü
        if np.isnan(observation).any():
            print(f"[ENV-ERROR] reset sonrası observation'da NaN var! Step: {self.current_step}")
            raise ValueError("reset sonrası observation NaN!")
        if not np.issubdtype(observation.dtype, np.floating):
            print(f"[ENV-ERROR] reset sonrası observation dtype float değil! Dtype: {observation.dtype}")
            raise TypeError(f"reset sonrası observation dtype float değil! Dtype: {observation.dtype}")
        if observation.shape != self.observation_space.shape:
            print(f"[ENV-ERROR] reset sonrası observation shape uyumsuz! Beklenen: {self.observation_space.shape}, Gelen: {observation.shape}")
            raise ValueError(f"reset sonrası observation shape uyumsuz! Beklenen: {self.observation_space.shape}, Gelen: {observation.shape}")
        return observation.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Bir adım ilerle"""
        
        # Action'ı normalize et
        action = np.clip(action, -1.0, 1.0)
        
        # Mevcut fiyatları al
        current_prices = self._get_current_prices()
        
        # Pozisyonları güncelle
        self._execute_trades(action, current_prices)
        
        # Bir adım ilerle
        self.current_step += 1
        
        # Yeni observation al
        observation = self._get_observation()
        
        # Reward hesapla
        reward = self._calculate_reward()
        
        # NaN ve dtype kontrolü
        if np.isnan(observation).any():
            print(f"[ENV-ERROR] step sonrası observation'da NaN var! Step: {self.current_step}")
            raise ValueError("step sonrası observation NaN!")
        if not np.issubdtype(observation.dtype, np.floating):
            print(f"[ENV-ERROR] step sonrası observation dtype float değil! Dtype: {observation.dtype}")
            raise TypeError(f"step sonrası observation dtype float değil! Dtype: {observation.dtype}")
        if observation.shape != self.observation_space.shape:
            print(f"[ENV-ERROR] step sonrası observation shape uyumsuz! Beklenen: {self.observation_space.shape}, Gelen: {observation.shape}")
            raise ValueError(f"step sonrası observation shape uyumsuz! Beklenen: {self.observation_space.shape}, Gelen: {observation.shape}")
        if not isinstance(reward, (float, np.floating)):
            print(f"[ENV-ERROR] step sonrası reward float değil! Type: {type(reward)}")
            raise TypeError(f"step sonrası reward float değil! Type: {type(reward)}")
        if np.isnan(reward):
            print(f"[ENV-ERROR] step sonrası reward NaN! Step: {self.current_step}")
            raise ValueError("step sonrası reward NaN!")
        
        # Episode bitti mi?
        done = self.current_step >= self.data_length - 1
        
        # Info
        info = self._get_info()
        
        return observation.astype(np.float32), float(reward), done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Mevcut durumu observation olarak döndür"""
        
        # Lookback window için veri al
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        observation_data = []
        
        for symbol in self.symbols:
            df = self.data[symbol][self.primary_timeframe]
            symbol_data = df.iloc[start_idx:end_idx].copy()
            
            # Timestamp'i çıkar, sadece OHLCV ve teknik göstergeler
            if 'timestamp' in symbol_data.columns:
                symbol_data = symbol_data.drop('timestamp', axis=1)
            
            # Normalize et
            symbol_data = self._normalize_features(symbol_data)
            
            observation_data.append(symbol_data.values)
        
        # Tüm symbol'leri birleştir
        observation = np.concatenate(observation_data, axis=1)
        observation = observation.flatten()  # Tek boyutlu yap
        
        # float32'ye zorla
        try:
            observation = observation.astype(np.float32)
        except Exception as e:
            print(f"[ENV-ERROR] _get_observation: float32'ye çevrilemedi! {e}")
            raise
        if not np.issubdtype(observation.dtype, np.floating):
            print(f"[ENV-ERROR] _get_observation: dtype float değil! Dtype: {observation.dtype}")
            raise TypeError(f"_get_observation: dtype float değil! Dtype: {observation.dtype}")
        return observation
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature'ları normalize et"""
        normalized_df = df.copy()
        
        # OHLCV normalize et (min-max scaling)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in normalized_df.columns:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        # Volume normalize et
        if 'volume' in normalized_df.columns:
            volume_mean = normalized_df['volume'].mean()
            volume_std = normalized_df['volume'].std()
            if volume_std > 0:
                normalized_df['volume'] = (normalized_df['volume'] - volume_mean) / volume_std
        
        # Teknik göstergeler zaten normalize edilmiş olmalı
        return normalized_df
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Mevcut fiyatları al"""
        prices = {}
        for symbol in self.symbols:
            df = self.data[symbol][self.primary_timeframe]
            prices[symbol] = df.iloc[self.current_step]['close']
        return prices
    
    def _execute_trades(self, action: np.ndarray, current_prices: Dict[str, float]):
        """Trade'leri gerçekleştir"""
        
        for i, symbol in enumerate(self.symbols):
            target_position = action[i]
            current_price = current_prices[symbol]
            
            # Mevcut pozisyon değeri
            current_position_value = self.positions[symbol] * current_price
            
            # Hedef pozisyon değeri
            target_position_value = target_position * self.portfolio_value * self.max_position_size
            
            # Pozisyon değişimi
            position_change = target_position_value - current_position_value
            
            if abs(position_change) > 0:  # Trade yapılacak
                # Komisyon hesapla
                commission_cost = abs(position_change) * self.commission
                
                # Bakiye güncelle
                self.balance -= position_change + commission_cost
                
                # Pozisyon güncelle
                self.positions[symbol] = target_position_value / current_price
                
                # Trade kaydı
                self.trade_history.append({
                    'step': self.current_step,
                    'symbol': symbol,
                    'action': target_position,
                    'price': current_price,
                    'position_change': position_change,
                    'commission': commission_cost
                })
        
        # Portfolio değerini güncelle
        self._update_portfolio_value(current_prices)
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Portfolio değerini güncelle"""
        portfolio_value = self.balance
        
        for symbol in self.symbols:
            position_value = self.positions[symbol] * current_prices[symbol]
            portfolio_value += position_value
        
        # Daily return hesapla
        if self.portfolio_value > 0:
            daily_return = (portfolio_value - self.portfolio_value) / self.portfolio_value
            self.daily_returns.append(daily_return)
        
        self.portfolio_value = portfolio_value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
    
    def _calculate_reward(self) -> float:
        """Reward hesapla"""
        
        if len(self.daily_returns) == 0:
            return 0.0
        
        # Son return
        current_return = self.daily_returns[-1]
        
        # Sharpe ratio benzeri reward (risk-ayarlı)
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            sharpe_like = current_return / (returns_array.std() + 1e-8)
        else:
            sharpe_like = current_return
        
        # Drawdown penalty
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        drawdown_penalty = -drawdown * 2.0
        
        # Risk penalty (aşırı pozisyon için)
        position_risk = sum(abs(pos) for pos in self.positions.values()) / len(self.symbols)
        risk_penalty = -position_risk * 0.1
        
        # Toplam reward
        reward = (sharpe_like + drawdown_penalty + risk_penalty) * self.reward_scaling
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """Environment bilgilerini döndür"""
        current_prices = self._get_current_prices()
        
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'positions': self.positions.copy(),
            'current_prices': current_prices,
            'total_return': (self.portfolio_value - self.initial_balance) / self.initial_balance,
            'max_drawdown': (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value,
            'n_trades': len(self.trade_history)
        }
        
        return info
    
    def _get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Verilen fiyatlarla portfolio değerini hesapla"""
        portfolio_value = self.balance
        
        for symbol in self.symbols:
            position_value = self.positions[symbol] * prices[symbol]
            portfolio_value += position_value
        
        return portfolio_value
    
    def get_portfolio_history(self) -> List[float]:
        """Portfolio değer geçmişini döndür"""
        portfolio_history = []
        
        for step in range(self.lookback_window, self.data_length):
            prices = {}
            for symbol in self.symbols:
                df = self.data[symbol][self.primary_timeframe]
                prices[symbol] = df.iloc[step]['close']
            
            portfolio_value = self._get_portfolio_value(prices)
            portfolio_history.append(portfolio_value)
        
        return portfolio_history
    
    def render(self):
        """Ortamı görselleştir (opsiyonel)"""
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Total Return: {((self.portfolio_value - self.initial_balance) / self.initial_balance) * 100:.2f}%")
        print(f"Positions: {self.positions}")
        print(f"Balance: ${self.balance:.2f}")

# Kullanım örneği:
# env = CryptoTradingEnv(data, initial_balance=10000)
# obs, info = env.reset()
# action = env.action_space.sample()
# obs, reward, done, truncated, info = env.step(action) 