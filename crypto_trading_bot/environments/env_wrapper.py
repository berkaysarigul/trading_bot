import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from .crypto_env import CryptoTradingEnv

class CryptoEnvWrapper(gym.Wrapper):
    """
    CryptoTradingEnv için Stable-Baselines3 wrapper'ı
    Observation ve action space'leri normalize eder
    """
    
    def __init__(self, env: CryptoTradingEnv, normalize_obs: bool = True, normalize_reward: bool = True):
        super().__init__(env)
        
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        
        # Observation normalization için
        self.obs_rms = None
        if normalize_obs:
            self._setup_obs_normalization()
        
        # Reward normalization için
        self.reward_rms = None
        if normalize_reward:
            self._setup_reward_normalization()
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _setup_obs_normalization(self):
        """Observation normalization ayarla"""
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        
        obs_shape = self.observation_space.shape
        self.obs_rms = RunningMeanStd(shape=obs_shape)
    
    def _setup_reward_normalization(self):
        """Reward normalization ayarla"""
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        
        self.reward_rms = RunningMeanStd(shape=())
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset ve observation normalize et"""
        obs, info = self.env.reset(**kwargs)
        
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        # Episode tracking
        self.episode_count += 1
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step ve reward normalize et"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Normalization
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        if self.normalize_reward:
            reward = self._normalize_reward(reward)
        
        # Episode sonu tracking
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Info'ya episode bilgilerini ekle
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_episode_length,
                'episode_count': self.episode_count
            }
        
        return obs, reward, done, truncated, info
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Observation'ı normalize et"""
        if self.obs_rms is not None:
            self.obs_rms.update(obs)
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return obs
    
    def _normalize_reward(self, reward: float) -> float:
        """Reward'ı normalize et"""
        if self.reward_rms is not None:
            self.reward_rms.update(np.array([reward]))
            reward = reward / np.sqrt(self.reward_rms.var + 1e-8)
        return reward
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Episode istatistiklerini döndür"""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'total_episodes': len(self.episode_rewards)
        }

class CryptoEnvMonitor(gym.Wrapper):
    """
    CryptoTradingEnv için monitoring wrapper'ı
    Detaylı trading metriklerini takip eder
    """
    
    def __init__(self, env: CryptoTradingEnv):
        super().__init__(env)
        
        # Trading metrikleri
        self.trading_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }
        
        # Portfolio tracking
        self.portfolio_history = []
        self.drawdown_history = []
        self.peak_portfolio = 0.0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset ve metrikleri sıfırla"""
        obs, info = self.env.reset(**kwargs)
        
        # Metrikleri sıfırla
        self.trading_metrics = {k: 0.0 if isinstance(v, float) else 0 for k, v in self.trading_metrics.items()}
        self.portfolio_history = []
        self.drawdown_history = []
        self.peak_portfolio = self.initial_balance
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step ve metrikleri güncelle"""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Portfolio tracking
        portfolio_value = info['portfolio_value']
        self.portfolio_history.append(portfolio_value)
        
        # Peak ve drawdown tracking
        if portfolio_value > self.peak_portfolio:
            self.peak_portfolio = portfolio_value
        
        drawdown = (self.peak_portfolio - portfolio_value) / self.peak_portfolio
        self.drawdown_history.append(drawdown)
        
        # Trade metriklerini güncelle
        self._update_trading_metrics(info)
        
        # Info'ya metrikleri ekle
        info.update(self.trading_metrics)
        info['current_drawdown'] = drawdown
        info['peak_portfolio'] = self.peak_portfolio
        
        return obs, reward, done, truncated, info
    
    def _update_trading_metrics(self, info: Dict[str, Any]):
        """Trading metriklerini güncelle"""
        # Yeni trade var mı?
        if info['n_trades'] > self.trading_metrics['total_trades']:
            # Yeni trade yapıldı
            self.trading_metrics['total_trades'] = info['n_trades']
            
            # Son trade'ın sonucunu hesapla
            if len(self.portfolio_history) >= 2:
                trade_return = (self.portfolio_history[-1] - self.portfolio_history[-2]) / self.portfolio_history[-2]
                
                if trade_return > 0:
                    # Kazanan trade
                    self.trading_metrics['winning_trades'] += 1
                    self.trading_metrics['total_profit'] += trade_return
                    self.trading_metrics['max_profit'] = max(self.trading_metrics['max_profit'], trade_return)
                    self.trading_metrics['consecutive_wins'] += 1
                    self.trading_metrics['consecutive_losses'] = 0
                    self.trading_metrics['max_consecutive_wins'] = max(
                        self.trading_metrics['max_consecutive_wins'], 
                        self.trading_metrics['consecutive_wins']
                    )
                else:
                    # Kaybeden trade
                    self.trading_metrics['losing_trades'] += 1
                    self.trading_metrics['total_loss'] += abs(trade_return)
                    self.trading_metrics['max_loss'] = max(self.trading_metrics['max_loss'], abs(trade_return))
                    self.trading_metrics['consecutive_losses'] += 1
                    self.trading_metrics['consecutive_wins'] = 0
                    self.trading_metrics['max_consecutive_losses'] = max(
                        self.trading_metrics['max_consecutive_losses'], 
                        self.trading_metrics['consecutive_losses']
                    )
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """Trading istatistiklerini döndür"""
        stats = self.trading_metrics.copy()
        
        # Ek metrikler
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            stats['profit_factor'] = (stats['total_profit'] / stats['total_loss']) if stats['total_loss'] > 0 else float('inf')
        else:
            stats['win_rate'] = 0.0
            stats['profit_factor'] = 0.0
        
        # Portfolio metrikleri
        if self.portfolio_history:
            stats['final_portfolio'] = self.portfolio_history[-1]
            stats['total_return'] = (self.portfolio_history[-1] - self.initial_balance) / self.initial_balance
            stats['max_drawdown'] = max(self.drawdown_history) if self.drawdown_history else 0.0
        
        return stats

def create_crypto_env(data: Dict[str, Dict[str, pd.DataFrame]], 
                     initial_balance: float = 10000,
                     commission: float = 0.001,
                     risk_per_trade: float = 0.02,
                     max_position_size: float = 0.1,
                     lookback_window: int = 50,
                     normalize_obs: bool = True,
                     normalize_reward: bool = True,
                     add_monitoring: bool = True) -> gym.Env:
    """
    Crypto trading environment'ı oluştur
    
    Args:
        data: Trading verisi
        initial_balance: Başlangıç bakiyesi
        commission: Komisyon oranı
        risk_per_trade: Trade başına risk
        max_position_size: Maksimum pozisyon büyüklüğü
        lookback_window: Geçmiş veri penceresi
        normalize_obs: Observation normalization
        normalize_reward: Reward normalization
        add_monitoring: Monitoring wrapper ekle
    
    Returns:
        Wrapped environment
    """
    
    # Base environment
    env = CryptoTradingEnv(
        data=data,
        initial_balance=initial_balance,
        commission=commission,
        risk_per_trade=risk_per_trade,
        max_position_size=max_position_size,
        lookback_window=lookback_window
    )
    
    # Monitoring wrapper
    if add_monitoring:
        env = CryptoEnvMonitor(env)
    
    # Normalization wrapper
    env = CryptoEnvWrapper(env, normalize_obs=normalize_obs, normalize_reward=normalize_reward)
    
    return env

# Kullanım örneği:
# env = create_crypto_env(data, initial_balance=10000, normalize_obs=True)
# obs, info = env.reset()
# action = env.action_space.sample()
# obs, reward, done, truncated, info = env.step(action) 