#!/usr/bin/env python3
"""
Crypto Trading Bot - Ana Uygulama
PPO + LSTM tabanlı çoklu varlık kripto trading botu
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# Proje modüllerini import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.training_pipeline import TrainingPipeline
from utils.risk_management import RiskManager

# Varsayılan konfigürasyon
DEFAULT_CONFIG = {
    'data_preparation': {
        'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        'timeframes': ['1h'],
        'days_back': 210,
        'limit': 5000,
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
    },
    'trading': {
        'max_position_size': 0.1,
        'max_daily_loss': 0.05,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'trading_symbols': ['BTC/USDT', 'ETH/USDT'],
        'trading_timeframes': ['1h', '4h'],
        'update_interval': 60
    }
}

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Logging ayarlarını yapılandır"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"crypto_bot_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def run_training_pipeline(config: Dict[str, Any], skip_tuning: bool = False) -> bool:
    """Training pipeline'ı çalıştır"""
    logger = logging.getLogger(__name__)
    logger.info("Starting training pipeline...")
    
    try:
        pipeline = TrainingPipeline(config)
        success = pipeline.run_full_pipeline(skip_tuning=skip_tuning)
        
        if success:
            pipeline.print_pipeline_summary()
            logger.info("Training pipeline completed successfully!")
        else:
            logger.error("Training pipeline failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Training pipeline error: {e}")
        return False

def run_live_trading(config: Dict[str, Any], model_path: str, paper_trading: bool = True) -> bool:
    """Canlı trading'i başlat (şimdilik placeholder)"""
    logger = logging.getLogger(__name__)
    logger.info("Live trading not implemented yet - placeholder")
    
    try:
        # Risk yöneticisi oluştur
        trading_config = config.get('trading', {})
        risk_manager = RiskManager(
            max_position_size=trading_config.get('max_position_size', 0.1),
            max_daily_loss=trading_config.get('max_daily_loss', 0.05),
            stop_loss=trading_config.get('stop_loss', 0.02),
            take_profit=trading_config.get('take_profit', 0.04)
        )
        
        logger.info(f"Risk manager created with config: {trading_config}")
        logger.info("Live trading module will be implemented in next phase")
        
        return True
        
    except Exception as e:
        logger.error(f"Live trading error: {e}")
        return False

def run_monitoring_dashboard(config: Dict[str, Any]) -> bool:
    """Monitoring dashboard'ı başlat (şimdilik placeholder)"""
    logger = logging.getLogger(__name__)
    logger.info("Monitoring dashboard not implemented yet - placeholder")
    
    try:
        logger.info("Dashboard module will be implemented in next phase")
        return True
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return False

def run_backtesting(config: Dict[str, Any], model_path: str) -> bool:
    """Backtesting çalıştır"""
    logger = logging.getLogger(__name__)
    logger.info("Starting backtesting...")
    
    try:
        from utils.backtesting_engine import BacktestingEngine
        
        backtest_config = config.get('backtesting', {})
        
        engine = BacktestingEngine(
            model_path=model_path,
            data_path=backtest_config.get('data_path', 'data/processed/test_data.pkl'),
            initial_balance=backtest_config.get('initial_balance', 10000),
            commission=backtest_config.get('commission', 0.001)
        )
        
        # Backtest çalıştır
        results = engine.run_backtest()
        
        # Sonuçları kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/backtest_results_{timestamp}.json"
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Backtesting completed! Results saved to: {results_path}")
        return True
        
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        return False

def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument('--mode', choices=['train', 'trade', 'backtest', 'monitor', 'full'], 
                       default='train', help='Çalışma modu')
    parser.add_argument('--config', type=str, default='config/trading_config.json',
                       help='Konfigürasyon dosyası yolu')
    parser.add_argument('--model-path', type=str, default='models/best_model',
                       help='Model dosyası yolu')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Hyperparameter tuning\'ı atla')
    parser.add_argument('--paper-trading', action='store_true', default=True,
                       help='Paper trading modu (gerçek para kullanma)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Log seviyesi')
    
    args = parser.parse_args()
    
    # Logging ayarla
    logger = setup_logging(args.log_level)
    logger.info(f"Starting Crypto Trading Bot in {args.mode} mode")
    
    # Konfigürasyonu yükle
    config = DEFAULT_CONFIG.copy()
    
    # Eğer özel konfigürasyon dosyası varsa yükle
    if os.path.exists(args.config):
        import json
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    try:
        if args.mode == 'train':
            # Training pipeline
            success = run_training_pipeline(config, skip_tuning=args.skip_tuning)
            
        elif args.mode == 'trade':
            # Live trading
            success = run_live_trading(config, args.model_path, args.paper_trading)
            
        elif args.mode == 'backtest':
            # Backtesting
            success = run_backtesting(config, args.model_path)
            
        elif args.mode == 'monitor':
            # Monitoring dashboard
            success = run_monitoring_dashboard(config)
            
        elif args.mode == 'full':
            # Tam pipeline: train -> backtest -> trade
            logger.info("Running full pipeline: train -> backtest -> trade")
            
            # 1. Training
            train_success = run_training_pipeline(config, skip_tuning=args.skip_tuning)
            if not train_success:
                logger.error("Training failed, stopping pipeline")
                return False
            
            # 2. Backtesting
            backtest_success = run_backtesting(config, args.model_path)
            if not backtest_success:
                logger.error("Backtesting failed, stopping pipeline")
                return False
            
            # 3. Live trading
            trade_success = run_live_trading(config, args.model_path, args.paper_trading)
            success = trade_success
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return False
        
        if success:
            logger.info(f"{args.mode} mode completed successfully!")
        else:
            logger.error(f"{args.mode} mode failed!")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return True
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 