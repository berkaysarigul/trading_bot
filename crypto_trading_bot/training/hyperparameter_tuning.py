import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from environments.crypto_env import CryptoTradingEnv
from training.rl_trainer import RLTrainer
from utils.performance_metrics import sharpe_ratio
import os
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial
from sb3_contrib import RecurrentPPO

class HyperparameterTuner:
    def __init__(self, train_data, val_data, n_trials=100, study_name="ppo_optimization"):
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.study_name = study_name
        self.best_params = None
        self.best_score = -np.inf
        
        # Optuna study oluştur
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=None,  # SQLite kullanmak için: f"sqlite:///{study_name}.db"
            load_if_exists=True
        )
        
        # Sonuçları kaydetmek için
        self.results_dir = "hyperparameter_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def objective(self, trial):
        """Optuna objective function - tek trial için hyperparameter optimization"""
        # --- DATA UZUNLUĞU KONTROLÜ ---
        # Tüm semboller ve timeframeler için minimum veri uzunluğunu bul
        min_data_length = min([len(df['1h']) for df in self.train_data.values()])
        # Search space'i veri uzunluğuna göre kısıtla
        max_n_steps = min(1024, max(32, min_data_length // 2))
        max_batch_size = min(256, max(16, min_data_length // 4))
        max_lookback = min(100, max(10, min_data_length // 8))
        # Hyperparameter space tanımla (sadece desteklenen parametreler)
        trial_params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_int('n_steps', 32, max_n_steps),
            'batch_size': trial.suggest_int('batch_size', 16, max_batch_size),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 1.0),
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.01, 0.05),
            'commission': trial.suggest_float('commission', 0.0005, 0.002),
            'lookback_window': trial.suggest_int('lookback_window', 10, max_lookback),
        }
        # --- PARAMETRELERİN UYGUNLUĞUNU KONTROL ET ---
        if trial_params['n_steps'] + trial_params['lookback_window'] + 1 > min_data_length:
            raise optuna.TrialPruned()
        if trial_params['batch_size'] > trial_params['n_steps']:
            raise optuna.TrialPruned()
        try:
            # Model parametreleri
            allowed_params = [
                'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm',
            ]
            model_params = {k: v for k, v in trial_params.items() if k in allowed_params}
            # Ortam parametrelerini ayır
            env_params = {
                'commission': trial_params['commission'],
                'risk_per_trade': trial_params['risk_per_trade'],
                'lookback_window': trial_params['lookback_window']
            }
            # Ortamları oluştur
            train_env = self.create_env(self.train_data, env_params)
            val_env = self.create_env(self.val_data, env_params)
            # Model oluştur ve eğit
            model = RecurrentPPO(
                "MlpLstmPolicy",
                train_env,
                **model_params
            )
            # Kısa eğitim (hızlı değerlendirme için)
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path=None,
                log_path=None,
                eval_freq=5000,
                deterministic=True,
                render=False
            )
            model.learn(
                total_timesteps=50000,  # Kısa eğitim
                callback=eval_callback,
                progress_bar=False
            )
            # Modeli değerlendir
            score = self.evaluate_model(model, val_env)
            trial.set_user_attr('score', score)
            # Sadece desteklenen parametreleri kaydet
            trial.set_user_attr('params', trial_params)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return -np.inf
    
    def create_env(self, data, env_params):
        """Ortam oluştur"""
        # Veri uzunluklarını ve anahtarları logla
        print("Env'e giden data:", {k: {tf: len(v[tf]) for tf in v} for k, v in data.items()})
        lookback_window = env_params.get('lookback_window', 50)
        env = CryptoTradingEnv(
            data=data,
            initial_balance=10000,
            commission=env_params['commission'],
            risk_per_trade=env_params['risk_per_trade'],
            lookback_window=lookback_window
        )
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        return vec_env
    
    def evaluate_model(self, model, val_env, n_episodes=5):
        """Modeli değerlendir ve score döndür"""
        returns = []
        
        for _ in range(n_episodes):
            obs = val_env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)
                episode_rewards.append(reward)
            
            # Episode return hesapla
            if len(episode_rewards) > 0:
                total_return = np.sum(episode_rewards)
                returns.append(total_return)
        
        if len(returns) > 0:
            # Sharpe ratio hesapla (risk-free rate = 0)
            sharpe = sharpe_ratio(np.array(returns))
            return sharpe if not np.isnan(sharpe) else -np.inf
        else:
            return -np.inf
    
    def optimize_parallel(self, n_jobs=-1):
        """Parallel hyperparameter optimization"""
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        print(f"Starting parallel hyperparameter optimization with {n_jobs} jobs...")
        
        # Parallel optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # En iyi sonuçları kaydet
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"Optimization completed!")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Sonuçları kaydet
        self.save_results()
        
        return self.best_params, self.best_score
    
    def optimize_sequential(self):
        """Sequential hyperparameter optimization"""
        print("Starting sequential hyperparameter optimization...")
        
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # En iyi sonuçları kaydet
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        print(f"Optimization completed!")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Sonuçları kaydet
        self.save_results()
        
        return self.best_params, self.best_score
    
    def save_results(self):
        """Optimization sonuçlarını kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # En iyi parametreleri kaydet
        best_params_path = os.path.join(self.results_dir, f"best_params_{timestamp}.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_trials': self.n_trials,
                'timestamp': timestamp
            }, f, indent=2)
        
        # Tüm trial sonuçlarını kaydet
        trials_path = os.path.join(self.results_dir, f"trials_{timestamp}.json")
        trials_data = []
        
        for trial in self.study.trials:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'state': trial.state.name
            })
        
        with open(trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        print(f"Results saved to {self.results_dir}")
    
    def plot_optimization_history(self):
        """Optimization geçmişini görselleştir"""
        try:
            import matplotlib.pyplot as plt
            
            # Optimization history
            plt.figure(figsize=(12, 8))
            
            # Objective values
            plt.subplot(2, 2, 1)
            plt.plot([trial.value for trial in self.study.trials if trial.value is not None])
            plt.title('Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.grid(True)
            
            # Parameter importance
            plt.subplot(2, 2, 2)
            importance = optuna.importance.get_param_importances(self.study)
            if importance:
                params = list(importance.keys())
                values = list(importance.values())
                plt.barh(params, values)
                plt.title('Parameter Importance')
                plt.xlabel('Importance')
            
            # Learning rate distribution
            plt.subplot(2, 2, 3)
            lr_values = [trial.params.get('learning_rate', 0) for trial in self.study.trials]
            plt.hist(lr_values, bins=20, alpha=0.7)
            plt.title('Learning Rate Distribution')
            plt.xlabel('Learning Rate')
            plt.ylabel('Frequency')
            
            # Score distribution
            plt.subplot(2, 2, 4)
            scores = [trial.value for trial in self.study.trials if trial.value is not None]
            plt.hist(scores, bins=20, alpha=0.7)
            plt.title('Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            
            # Kaydet
            plot_path = os.path.join(self.results_dir, f"optimization_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Optimization plots saved to: {plot_path}")
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def get_best_model_params(self):
        """En iyi parametreleri RLTrainer formatında döndür"""
        if not self.best_params:
            return None
        
        # RLTrainer formatına çevir
        model_params = {
            'learning_rate': self.best_params['learning_rate'],
            'n_steps': self.best_params['n_steps'],
            'batch_size': self.best_params['batch_size'],
            'n_epochs': self.best_params['n_epochs'],
            'gamma': self.best_params['gamma'],
            'gae_lambda': self.best_params['gae_lambda'],
            'clip_range': self.best_params['clip_range'],
            'ent_coef': self.best_params['ent_coef'],
            'vf_coef': self.best_params['vf_coef'],
            'max_grad_norm': self.best_params['max_grad_norm'],
        }
        
        return model_params
    
    def print_optimization_summary(self):
        """Optimization özetini yazdır"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Study Name: {self.study_name}")
        print(f"Number of Trials: {len(self.study.trials)}")
        print(f"Best Score: {self.best_score:.4f}")
        print(f"Best Trial Number: {self.study.best_trial.number}")
        print("\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print("="*60)

# Kullanım örneği:
# tuner = HyperparameterTuner(train_data, val_data, n_trials=50)
# best_params, best_score = tuner.optimize_parallel(n_jobs=4)
# tuner.plot_optimization_history()
# tuner.print_optimization_summary() 