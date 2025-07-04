import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from data.crypto_data_fetcher import CryptoDataFetcher
from data.market_data_processor import get_market_caps, apply_market_cap_weights
from training.data_preparation import DataPreparation
from training.feature_engineering import AdvancedFeatureEngineering
from training.rl_trainer import RLTrainer
from training.model_evaluator import ModelEvaluator
from training.hyperparameter_tuning import HyperparameterTuner

class TrainingPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Training pipeline'ı başlat
        
        Args:
            config: Pipeline konfigürasyonu
        """
        self.config = config
        self.setup_logging()
        self.results = {}
        
        # Dizinleri oluştur
        self.create_directories()
        
        # Pipeline aşamaları
        self.data_prep = None
        self.feature_eng = None
        self.trainer = None
        self.evaluator = None
        self.tuner = None
        
    def setup_logging(self):
        """Logging ayarlarını yapılandır"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_pipeline_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training pipeline initialized")
    
    def create_directories(self):
        """Gerekli dizinleri oluştur"""
        directories = [
            "models",
            "data/processed",
            "logs",
            "results",
            "hyperparameter_results"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_data_preparation(self) -> bool:
        """Veri hazırlama aşamasını çalıştır"""
        self.logger.info("Starting data preparation phase...")
        
        try:
            data_config = self.config.get('data_preparation', {})
            
            self.data_prep = DataPreparation(
                symbols=data_config.get('symbols', None),
                timeframes=data_config.get('timeframes', None),
                days_back=data_config.get('days_back', 210)  # 210 gün ≈ 7 ay
            )
            
            # Veri çek ve hazırla
            train_data, val_data, test_data = self.data_prep.prepare_training_data(days_back=self.data_prep.days_back)
            processed_data = {**train_data, **val_data, **test_data}  # Tüm verileri birleştir
            
            # Veri kalitesi kontrolü
            if not self.data_prep.validate_data(processed_data):
                self.logger.error("Data validation failed")
                return False
            
            # Veriyi kaydet
            self.data_prep.save_data(processed_data, 'data/processed/processed_data.pkl')
            
            self.results['data_preparation'] = {
                'status': 'success',
                'data_shape': {symbol: {tf: df.shape for tf, df in data.items()} 
                             for symbol, data in processed_data.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Data preparation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            self.results['data_preparation'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_feature_engineering(self) -> bool:
        """Feature engineering aşamasını çalıştır"""
        self.logger.info("Starting feature engineering phase...")
        
        try:
            if not self.data_prep:
                self.logger.error("Data preparation must be run first")
                return False
            
            # İşlenmiş veriyi yükle
            processed_data = self.data_prep.load_data('data/processed/processed_data.pkl')
            
            feature_config = self.config.get('feature_engineering', {})
            
            self.feature_eng = AdvancedFeatureEngineering(
                lookback_window=feature_config.get('lookback_window', 50),
                technical_indicators=feature_config.get('technical_indicators', True),
                market_regime_features=feature_config.get('market_regime_features', True),
                sentiment_features=feature_config.get('sentiment_features', False),
                risk_features=feature_config.get('risk_features', True)
            )
            
            # Feature'ları oluştur
            feature_data = self.feature_eng.create_features(processed_data)
            
            # Feature selection
            if feature_config.get('feature_selection', True):
                feature_data = self.feature_eng.select_features(feature_data)
            
            # Veriyi train/val/test olarak böl
            train_data, val_data, test_data = self.feature_eng.split_data(feature_data)
            
            # Feature'ları kaydet
            self.feature_eng.save_features(train_data, 'data/processed/train_data.pkl')
            self.feature_eng.save_features(val_data, 'data/processed/val_data.pkl')
            self.feature_eng.save_features(test_data, 'data/processed/test_data.pkl')
            
            self.results['feature_engineering'] = {
                'status': 'success',
                'feature_count': len(self.feature_eng.get_feature_names()),
                'train_shape': {symbol: {tf: df.shape for tf, df in data.items()} 
                              for symbol, data in train_data.items()},
                'val_shape': {symbol: {tf: df.shape for tf, df in data.items()} 
                            for symbol, data in val_data.items()},
                'test_shape': {symbol: {tf: df.shape for tf, df in data.items()} 
                             for symbol, data in test_data.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            # Test verisini sakla
            self.test_data = test_data
            
            self.logger.info("Feature engineering completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            self.results['feature_engineering'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_hyperparameter_tuning(self) -> bool:
        """Hyperparameter tuning aşamasını çalıştır"""
        self.logger.info("Starting hyperparameter tuning phase...")
        
        try:
            if not self.feature_eng:
                self.logger.error("Feature engineering must be run first")
                return False
            
            # Veriyi yükle
            train_data = self.feature_eng.load_features('data/processed/train_data.pkl')
            val_data = self.feature_eng.load_features('data/processed/val_data.pkl')
            
            tuning_config = self.config.get('hyperparameter_tuning', {})
            
            self.tuner = HyperparameterTuner(
                train_data=train_data,
                val_data=val_data,
                n_trials=tuning_config.get('n_trials', 50),
                study_name=tuning_config.get('study_name', 'ppo_optimization')
            )
            
            # Optimization çalıştır
            if tuning_config.get('parallel', True):
                best_params, best_score = self.tuner.optimize_parallel(
                    n_jobs=tuning_config.get('n_jobs', -1)
                )
            else:
                best_params, best_score = self.tuner.optimize_sequential()
            
            # Sonuçları görselleştir
            if tuning_config.get('plot_results', True):
                self.tuner.plot_optimization_history()
            
            # Özet yazdır
            self.tuner.print_optimization_summary()
            
            self.results['hyperparameter_tuning'] = {
                'status': 'success',
                'best_score': best_score,
                'best_params': best_params,
                'n_trials': tuning_config.get('n_trials', 50),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Hyperparameter tuning completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            self.results['hyperparameter_tuning'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_model_training(self) -> bool:
        """Model eğitimi aşamasını çalıştır"""
        self.logger.info("Starting model training phase...")
        
        try:
            if not self.feature_eng:
                self.logger.error("Feature engineering must be run first")
                return False
            
            # Veriyi yükle
            train_data = self.feature_eng.load_features('data/processed/train_data.pkl')
            val_data = self.feature_eng.load_features('data/processed/val_data.pkl')
            
            training_config = self.config.get('model_training', {})
            
            # Eğer hyperparameter tuning yapıldıysa, en iyi parametreleri kullan
            if self.tuner and self.tuner.best_params:
                model_params = self.tuner.get_best_model_params()
                self.logger.info("Using optimized hyperparameters")
            else:
                model_params = training_config.get('model_params', {})
                self.logger.info("Using default hyperparameters")
            
            self.trainer = RLTrainer(
                train_data=train_data,
                val_data=val_data,
                model_save_path=training_config.get('model_save_path', 'models/')
            )
            
            # Model parametrelerini güncelle
            if model_params:
                self.trainer.model_params.update(model_params)
            
            # Eğitim çalıştır
            use_curriculum = training_config.get('use_curriculum', True)
            model = self.trainer.train_model(use_curriculum=use_curriculum)
            
            if model is None:
                self.logger.error("Model training failed")
                return False
            
            self.results['model_training'] = {
                'status': 'success',
                'model_path': self.trainer.best_model_path,
                'use_curriculum': use_curriculum,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.results['model_training'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_model_evaluation(self) -> bool:
        """Model değerlendirme aşamasını çalıştır"""
        self.logger.info("Starting model evaluation phase...")
        
        try:
            if not self.trainer or not self.trainer.best_model_path:
                self.logger.error("Model training must be run first")
                return False
            
            evaluation_config = self.config.get('model_evaluation', {})
            
            self.evaluator = ModelEvaluator(
                model_path=self.trainer.best_model_path,
                test_data=self.test_data
            )
            
            # Kapsamlı değerlendirme
            n_episodes = evaluation_config.get('n_episodes', 50)
            results = self.evaluator.comprehensive_evaluation(n_episodes=n_episodes)
            
            if results is None:
                self.logger.error("Model evaluation failed")
                return False
            
            # Sonuçları kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"results/evaluation_results_{timestamp}.json"
            self.evaluator.save_results(results_path)
            
            self.results['model_evaluation'] = {
                'status': 'success',
                'results_path': results_path,
                'n_episodes': n_episodes,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Model evaluation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            self.results['model_evaluation'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False
    
    def run_full_pipeline(self, skip_tuning: bool = False) -> bool:
        """Tüm pipeline'ı çalıştır"""
        self.logger.info("Starting full training pipeline...")
        
        pipeline_steps = [
            ("Data Preparation", self.run_data_preparation),
            ("Feature Engineering", self.run_feature_engineering),
        ]
        
        if not skip_tuning:
            pipeline_steps.append(("Hyperparameter Tuning", self.run_hyperparameter_tuning))
        
        pipeline_steps.extend([
            ("Model Training", self.run_model_training),
            ("Model Evaluation", self.run_model_evaluation)
        ])
        
        success = True
        
        for step_name, step_func in pipeline_steps:
            self.logger.info(f"Running {step_name}...")
            
            if not step_func():
                self.logger.error(f"{step_name} failed, stopping pipeline")
                success = False
                break
            
            self.logger.info(f"{step_name} completed successfully")
        
        # Pipeline sonuçlarını kaydet
        self.save_pipeline_results()
        
        if success:
            self.logger.info("Full training pipeline completed successfully!")
        else:
            self.logger.error("Training pipeline failed!")
        
        return success
    
    def save_pipeline_results(self):
        """Pipeline sonuçlarını kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/pipeline_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Pipeline results saved to: {results_path}")
    
    def print_pipeline_summary(self):
        """Pipeline özetini yazdır"""
        print("\n" + "="*80)
        print("TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        for step_name, step_result in self.results.items():
            status = step_result.get('status', 'unknown')
            timestamp = step_result.get('timestamp', 'N/A')
            
            print(f"{step_name.replace('_', ' ').title()}: {status}")
            print(f"  Timestamp: {timestamp}")
            
            if status == 'success':
                # Başarılı adımlar için ek bilgiler
                if step_name == 'data_preparation':
                    data_shape = step_result.get('data_shape', {})
                    print(f"  Data shapes: {data_shape}")
                elif step_name == 'feature_engineering':
                    feature_count = step_result.get('feature_count', 0)
                    print(f"  Feature count: {feature_count}")
                elif step_name == 'hyperparameter_tuning':
                    best_score = step_result.get('best_score', 0)
                    print(f"  Best score: {best_score:.4f}")
                elif step_name == 'model_training':
                    model_path = step_result.get('model_path', 'N/A')
                    print(f"  Model path: {model_path}")
                elif step_name == 'model_evaluation':
                    results_path = step_result.get('results_path', 'N/A')
                    print(f"  Results path: {results_path}")
            else:
                # Başarısız adımlar için hata mesajı
                error = step_result.get('error', 'Unknown error')
                print(f"  Error: {error}")
            
            print()
        
        print("="*80)

# Örnek konfigürasyon
DEFAULT_CONFIG = {
    'data_preparation': {
        'symbols': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        'timeframes': ['1h', '4h', '1d'],
        'start_date': '2023-01-01',
        'end_date': '2024-01-01',
        'data_dir': 'data'
    },
    'feature_engineering': {
        'lookback_window': 50,
        'technical_indicators': True,
        'market_regime_features': True,
        'sentiment_features': False,
        'risk_features': True,
        'feature_selection': True
    },
    'hyperparameter_tuning': {
        'n_trials': 50,
        'study_name': 'ppo_optimization',
        'parallel': True,
        'n_jobs': -1,
        'plot_results': True
    },
    'model_training': {
        'model_save_path': 'models/',
        'use_curriculum': True,
        'model_params': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10
        }
    },
    'model_evaluation': {
        'n_episodes': 50
    }
}

# Kullanım örneği:
# pipeline = TrainingPipeline(DEFAULT_CONFIG)
# success = pipeline.run_full_pipeline(skip_tuning=False)
# pipeline.print_pipeline_summary() 