import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from environments.crypto_env import CryptoTradingEnv
from utils.performance_metrics import (
    sharpe_ratio, calmar_ratio, sortino_ratio, win_loss_ratio,
    profit_factor, recovery_factor, alpha_beta, value_at_risk
)
from utils.backtesting_engine import monte_carlo_simulation, stress_test
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ModelEvaluator:
    def __init__(self, model_path, test_data):
        self.model_path = model_path
        self.test_data = test_data
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            # policy_kwargs'ı sadeleştir
            policy_kwargs = {
                'net_arch': self.best_params['net_arch'],
                'activation_fn': self.best_params['activation_fn']
            }
            self.model = RecurrentPPO.load(self.model_path, policy_kwargs=policy_kwargs)
            print(f"Model loaded successfully from: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_test_env(self, data):
        """Test ortamı oluştur"""
        env = CryptoTradingEnv(
            data=data,
            initial_balance=10000,
            commission=0.001,
            risk_per_trade=0.02
        )
        return env
    
    def run_single_episode(self, env, deterministic=True):
        """Tek episode çalıştır"""
        obs = env.reset()
        done = False
        rewards = []
        portfolio_values = []
        actions_taken = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            rewards.append(reward)
            portfolio_values.append(env._get_portfolio_value({
                s: env.data[s][env.timeframes[0]].iloc[env.current_step]['close'] 
                for s in env.symbols
            }))
            actions_taken.append(action)
        
        return {
            'rewards': rewards,
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] if len(portfolio_values) > 1 else 0
        }
    
    def walk_forward_analysis(self, window_size=100, step_size=50):
        """Walk-forward analysis ile model performansını değerlendir"""
        print("Running walk-forward analysis...")
        
        results = []
        total_length = len(next(iter(self.test_data.values()))[next(iter(next(iter(self.test_data.values())).keys()))])
        
        for start_idx in range(0, total_length - window_size, step_size):
            end_idx = start_idx + window_size
            
            # Test verisi kesiti
            test_slice = {}
            for symbol in self.test_data.keys():
                test_slice[symbol] = {}
                for tf in self.test_data[symbol].keys():
                    test_slice[symbol][tf] = self.test_data[symbol][tf].iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Test ortamı oluştur
            test_env = self.create_test_env(test_slice)
            
            # Episode çalıştır
            episode_result = self.run_single_episode(test_env)
            
            results.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'total_return': episode_result['total_return'],
                'avg_reward': np.mean(episode_result['rewards']),
                'volatility': np.std(episode_result['portfolio_values']) if len(episode_result['portfolio_values']) > 1 else 0
            })
        
        # Sonuçları analiz et
        returns = [r['total_return'] for r in results]
        rewards = [r['avg_reward'] for r in results]
        
        walk_forward_results = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': sharpe_ratio(np.array(returns)),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'n_windows': len(results)
        }
        
        self.results['walk_forward'] = walk_forward_results
        print(f"Walk-forward analysis completed: {len(results)} windows")
        return walk_forward_results
    
    def stress_testing(self, market_conditions=['bear', 'bull', 'sideways']):
        """Farklı piyasa koşullarında stress test"""
        print("Running stress testing...")
        
        stress_results = {}
        
        for condition in market_conditions:
            print(f"Testing {condition} market condition...")
            
            # Stress test fonksiyonunu kullan
            test_results = stress_test(
                env_class=CryptoTradingEnv,
                agent_class=lambda: self.model,
                data=self.test_data,
                market_type=condition,
                window_size=200
            )
            
            if test_results:
                returns = []
                for result in test_results:
                    if result[1]:  # portfolio_values
                        portfolio_values = result[1][0]
                        if len(portfolio_values) > 1:
                            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                            returns.append(total_return)
                
                if returns:
                    stress_results[condition] = {
                        'mean_return': np.mean(returns),
                        'std_return': np.std(returns),
                        'sharpe_ratio': sharpe_ratio(np.array(returns)),
                        'win_rate': len([r for r in returns if r > 0]) / len(returns),
                        'n_tests': len(returns)
                    }
        
        self.results['stress_test'] = stress_results
        print("Stress testing completed")
        return stress_results
    
    def monte_carlo_simulation(self, n_simulations=100, window_size=200):
        """Monte Carlo simulation"""
        print(f"Running Monte Carlo simulation with {n_simulations} simulations...")
        
        # Monte Carlo simulation fonksiyonunu kullan
        mc_results = monte_carlo_simulation(
            env_class=CryptoTradingEnv,
            agent_class=lambda: self.model,
            data=self.test_data,
            n_simulations=n_simulations,
            window_size=window_size
        )
        
        # Sonuçları analiz et
        returns = []
        for result in mc_results:
            if result[1]:  # portfolio_values
                portfolio_values = result[1][0]
                if len(portfolio_values) > 1:
                    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                    returns.append(total_return)
        
        if returns:
            mc_analysis = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe_ratio': sharpe_ratio(np.array(returns)),
                'var_95': value_at_risk(np.array(returns), alpha=0.05),
                'var_99': value_at_ratio(np.array(returns), alpha=0.01),
                'win_rate': len([r for r in returns if r > 0]) / len(returns),
                'n_simulations': len(returns)
            }
            
            self.results['monte_carlo'] = mc_analysis
            print("Monte Carlo simulation completed")
            return mc_analysis
        
        return None
    
    def calculate_detailed_metrics(self, portfolio_values, returns):
        """Detaylı performans metrikleri hesapla"""
        if len(returns) < 2:
            return {}
        
        # Temel metrikler
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = total_return * (252 / len(returns))
        
        # Risk metrikleri
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        
        # Drawdown hesapla
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        calmar = calmar_ratio(annualized_return, max_drawdown)
        
        # Trading metrikleri
        win_rate = win_loss_ratio(returns)
        profit_fact = profit_factor(returns)
        recovery_fact = recovery_factor(total_return, abs(max_drawdown))
        
        # VaR
        var_95 = value_at_risk(returns, alpha=0.05)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_fact,
            'recovery_factor': recovery_fact,
            'var_95': var_95
        }
    
    def comprehensive_evaluation(self, n_episodes=50):
        """Kapsamlı model değerlendirmesi"""
        print("Starting comprehensive model evaluation...")
        
        if not self.load_model():
            return None
        
        # 1. Çoklu episode test
        test_env = self.create_test_env(self.test_data)
        episode_results = []
        
        for episode in range(n_episodes):
            result = self.run_single_episode(test_env, deterministic=True)
            episode_results.append(result)
        
        # 2. Detaylı metrikler
        all_returns = []
        all_portfolio_values = []
        
        for result in episode_results:
            if len(result['portfolio_values']) > 1:
                returns = np.diff(result['portfolio_values']) / result['portfolio_values'][:-1]
                all_returns.extend(returns)
                all_portfolio_values.extend(result['portfolio_values'])
        
        detailed_metrics = self.calculate_detailed_metrics(
            np.array(all_portfolio_values), 
            np.array(all_returns)
        )
        
        # 3. Walk-forward analysis
        walk_forward_results = self.walk_forward_analysis()
        
        # 4. Stress testing
        stress_results = self.stress_testing()
        
        # 5. Monte Carlo simulation
        mc_results = self.monte_carlo_simulation()
        
        # Tüm sonuçları birleştir
        comprehensive_results = {
            'detailed_metrics': detailed_metrics,
            'walk_forward': walk_forward_results,
            'stress_test': stress_results,
            'monte_carlo': mc_results,
            'n_episodes': n_episodes,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        self.results = comprehensive_results
        
        # Sonuçları yazdır
        self.print_evaluation_summary(comprehensive_results)
        
        return comprehensive_results
    
    def print_evaluation_summary(self, results):
        """Değerlendirme özetini yazdır"""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        if 'detailed_metrics' in results:
            metrics = results['detailed_metrics']
            print(f"Total Return: {metrics.get('total_return', 0):.4f}")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.4f}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
            print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.4f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.4f}")
            print(f"VaR (95%): {metrics.get('var_95', 0):.4f}")
        
        if 'walk_forward' in results:
            wf = results['walk_forward']
            print(f"\nWalk-Forward Analysis:")
            print(f"  Mean Return: {wf.get('mean_return', 0):.4f}")
            print(f"  Sharpe Ratio: {wf.get('sharpe_ratio', 0):.4f}")
            print(f"  Win Rate: {wf.get('win_rate', 0):.4f}")
        
        if 'stress_test' in results:
            print(f"\nStress Testing:")
            for condition, metrics in results['stress_test'].items():
                print(f"  {condition.capitalize()}: Return={metrics.get('mean_return', 0):.4f}, Sharpe={metrics.get('sharpe_ratio', 0):.4f}")
        
        print("="*60)
    
    def save_results(self, output_path="evaluation_results.json"):
        """Sonuçları JSON olarak kaydet"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to: {output_path}")

# Kullanım örneği:
# evaluator = ModelEvaluator("models/best_model", test_data)
# results = evaluator.comprehensive_evaluation(n_episodes=50)
# evaluator.save_results() 