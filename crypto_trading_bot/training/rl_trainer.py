import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from environments.crypto_env import CryptoTradingEnv
from models.ppo_lstm_model import PPO_LSTM_Model
import os
import json
from datetime import datetime
from sb3_contrib import RecurrentPPO

class RLTrainer:
    def __init__(self, train_data, val_data, model_save_path="models/"):
        self.train_data = train_data
        self.val_data = val_data
        self.model_save_path = model_save_path
        self.best_model_path = None
        self.training_history = []
        
        # Model parametreleri
        self.model_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'sde_sample_freq': -1,
            'target_kl': None,
            'tensorboard_log': './logs/',
            'policy_kwargs': {
                'net_arch': [dict(pi=[128, 128], vf=[128, 128])],
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 128,
                'n_lstm_layers': 2,
                'dropout': 0.1
            }
        }
        
        # Eğitim parametreleri
        self.training_params = {
            'total_timesteps': 1000000,  # 1M steps
            'eval_freq': 10000,
            'save_freq': 50000,
            'patience': 20,  # Early stopping patience
            'min_improvement': 0.01
        }
        
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs('./logs/', exist_ok=True)
        
    def create_env(self, data, is_training=True):
        """Eğitim veya validasyon ortamı oluştur"""
        env = CryptoTradingEnv(
            data=data,
            initial_balance=10000,
            commission=0.001,
            risk_per_trade=0.02
        )
        env = Monitor(env)
        return env
    
    def create_vec_env(self, data, is_training=True):
        """Vectorized environment oluştur (daha hızlı eğitim için)"""
        env = self.create_env(data, is_training)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        return vec_env
    
    def create_callbacks(self, eval_env):
        """Eğitim callback'lerini oluştur"""
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_save_path,
            log_path=self.model_save_path,
            eval_freq=self.training_params['eval_freq'],
            deterministic=True,
            render=False
        )
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.training_params['save_freq'],
            save_path=self.model_save_path,
            name_prefix="ppo_crypto"
        )
        
        # Early stopping callback
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=self.training_params['patience'],
            min_evals=self.training_params['patience'] // 2,
            verbose=1
        )
        
        return [eval_callback, checkpoint_callback, stop_callback]
    
    def curriculum_learning(self, model, train_env, val_env):
        """Curriculum learning: Basit stratejilerden karmaşık stratejilere"""
        print("Starting curriculum learning...")
        
        # Curriculum aşamaları
        curriculum_stages = [
            {'risk_per_trade': 0.01, 'commission': 0.0005},  # Düşük risk, düşük komisyon
            {'risk_per_trade': 0.02, 'commission': 0.001},   # Orta risk, orta komisyon
            {'risk_per_trade': 0.03, 'commission': 0.002},   # Yüksek risk, yüksek komisyon
        ]
        
        for stage_idx, stage_params in enumerate(curriculum_stages):
            print(f"Curriculum stage {stage_idx + 1}/{len(curriculum_stages)}")
            print(f"Parameters: {stage_params}")
            
            # Ortamları güncelle
            train_env.envs[0].risk_per_trade = stage_params['risk_per_trade']
            train_env.envs[0].commission = stage_params['commission']
            val_env.envs[0].risk_per_trade = stage_params['risk_per_trade']
            val_env.envs[0].commission = stage_params['commission']
            
            # Bu aşamada eğitim
            timesteps_per_stage = self.training_params['total_timesteps'] // len(curriculum_stages)
            model.learn(
                total_timesteps=timesteps_per_stage,
                callback=self.create_callbacks(val_env),
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Aşama sonuçlarını kaydet
            self.save_training_history(stage_idx, stage_params)
    
    def train_model(self, use_curriculum=True):
        """Ana eğitim fonksiyonu"""
        print("Starting RL model training...")
        
        # Ortamları oluştur
        train_env = self.create_vec_env(self.train_data, is_training=True)
        val_env = self.create_vec_env(self.val_data, is_training=False)
        
        # Model oluştur
        policy_kwargs = {
            'net_arch': self.model_params['policy_kwargs']['net_arch'],
            'activation_fn': self.model_params['policy_kwargs']['activation_fn']
        }
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=self.model_params['learning_rate'],
            n_steps=self.model_params['n_steps'],
            batch_size=self.model_params['batch_size'],
            n_epochs=self.model_params['n_epochs'],
            gamma=self.model_params['gamma'],
            gae_lambda=self.model_params['gae_lambda'],
            clip_range=self.model_params['clip_range'],
            clip_range_vf=self.model_params['clip_range_vf'],
            ent_coef=self.model_params['ent_coef'],
            vf_coef=self.model_params['vf_coef'],
            max_grad_norm=self.model_params['max_grad_norm'],
            use_sde=self.model_params['use_sde'],
            sde_sample_freq=self.model_params['sde_sample_freq'],
            target_kl=self.model_params['target_kl']
        )
        
        # Eğitim
        if use_curriculum:
            self.curriculum_learning(model, train_env, val_env)
        else:
            model.learn(
                total_timesteps=self.training_params['total_timesteps'],
                callback=self.create_callbacks(val_env),
                progress_bar=True
            )
        
        # En iyi modeli kaydet
        best_model_path = os.path.join(self.model_save_path, "best_model")
        model.save(best_model_path)
        self.best_model_path = best_model_path
        
        # Ortam normalizasyonunu kaydet
        train_env.save(os.path.join(self.model_save_path, "vec_normalize.pkl"))
        
        print(f"Training completed! Best model saved at: {best_model_path}")
        return model
    
    def save_training_history(self, stage_idx, stage_params):
        """Eğitim geçmişini kaydet"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage_idx,
            'stage_params': stage_params,
            'model_params': self.model_params,
            'training_params': self.training_params
        }
        
        self.training_history.append(history_entry)
        
        # JSON olarak kaydet
        history_path = os.path.join(self.model_save_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_best_model(self):
        """En iyi modeli yükle"""
        if self.best_model_path and os.path.exists(self.best_model_path + ".zip"):
            model = RecurrentPPO.load(self.best_model_path)
            print(f"Loaded best model from: {self.best_model_path}")
            return model
        else:
            print("No best model found!")
            return None
    
    def evaluate_model(self, model, test_data):
        """Modeli test verisi üzerinde değerlendir"""
        print("Evaluating model on test data...")
        
        test_env = self.create_env(test_data, is_training=False)
        
        # Test episode'ları çalıştır
        n_episodes = 10
        episode_rewards = []
        episode_returns = []
        
        for episode in range(n_episodes):
            obs = test_env.reset()
            done = False
            total_reward = 0
            portfolio_values = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
                portfolio_values.append(test_env._get_portfolio_value({
                    s: test_env.data[s][test_env.timeframes[0]].iloc[test_env.current_step]['close'] 
                    for s in test_env.symbols
                }))
            
            episode_rewards.append(total_reward)
            if len(portfolio_values) > 1:
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                episode_returns.append(total_return)
        
        # Sonuçları hesapla
        avg_reward = np.mean(episode_rewards)
        avg_return = np.mean(episode_returns)
        std_reward = np.std(episode_rewards)
        
        results = {
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'std_reward': std_reward,
            'n_episodes': n_episodes
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average Return: {avg_return:.4f}")
        print(f"  Reward Std: {std_reward:.4f}")
        
        return results

# Kullanım örneği:
# trainer = RLTrainer(train_data, val_data)
# model = trainer.train_model(use_curriculum=True)
# results = trainer.evaluate_model(model, test_data) 