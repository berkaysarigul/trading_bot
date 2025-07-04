# Crypto Trading Bot

Bu proje, çoklu kripto varlıklar için PPO + LSTM tabanlı, gelişmiş teknik göstergeler ve risk yönetimi içeren, modüler ve production-ready bir trading bot altyapısıdır.

## 🚀 Özellikler

### 🤖 AI/ML Özellikleri
- **PPO + LSTM** tabanlı Reinforcement Learning
- **Curriculum Learning** ile aşamalı eğitim
- **Hyperparameter Optimization** (Optuna ile)
- **Multi-head Attention** ve **Ensemble** modeller
- **Walk-forward Analysis** ve **Monte Carlo Simulation**

### 📊 Veri İşleme
- **Binance API** ile gerçek zamanlı veri
- **Çoklu coin** ve **çoklu timeframe** desteği
- **Gelişmiş teknik göstergeler** (RSI, MACD, Bollinger Bands, VWAP, ATR, vb.)
- **Market cap ağırlıklandırması**, **Fear & Greed**, sosyal/news sentiment
- **Feature Engineering** ve **Feature Selection**

### 🛡️ Risk Yönetimi
- **Stop-loss**, **take-profit**, **VaR**, **correlation** analizi
- **Slippage** ve **likidite** kontrolü
- **Position sizing** ve **portfolio rebalancing**
- **Real-time risk monitoring**

### 📈 Analiz ve Test
- **Backtest**, **walk-forward**, **Monte Carlo** ve **stress test**
- **Performance metrics** (Sharpe, Sortino, Calmar, Profit Factor)
- **Streamlit dashboard** ve **real-time monitoring**
- **Telegram/Discord alert** sistemi

### 🔧 Production Features
- **Paper trading** ve **gerçek trade** desteği
- **Docker** ile kolay deployment
- **Comprehensive logging** ve **error handling**
- **Unit tests** ve **integration tests**

## 📁 Klasör Yapısı

```
crypto_trading_bot/
├── config/                 # API ve konfigürasyon dosyaları
│   └── api_config.py
├── data/                   # Veri işleme modülleri
│   ├── crypto_data_fetcher.py
│   ├── market_data_processor.py
│   └── technical_indicators.py
├── environments/           # RL ortamları
│   └── crypto_env.py
├── models/                 # AI modelleri
│   ├── ppo_lstm_model.py
│   ├── attention_model.py
│   └── ensemble_model.py
├── training/               # Eğitim pipeline'ı
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── rl_trainer.py
│   ├── model_evaluator.py
│   ├── hyperparameter_tuning.py
│   └── training_pipeline.py
├── trading/                # Canlı trading
│   ├── live_trader.py
│   ├── order_executor.py
│   └── position_manager.py
├── monitoring/             # Monitoring ve alert
│   ├── dashboard.py
│   ├── alert_system.py
│   └── report_generator.py
├── utils/                  # Yardımcı araçlar
│   ├── backtesting_engine.py
│   ├── performance_metrics.py
│   ├── risk_management.py
│   └── hyperparameter_tuning.py
├── main.py                 # Ana uygulama
├── Dockerfile
└── start.sh

tests/                      # Unit testler
data/                       # Veri dosyaları
models/                     # Eğitilmiş modeller
logs/                       # Log dosyaları
results/                    # Sonuçlar
```

## 🛠️ Kurulum

### 1. Gereksinimler
```bash
# Python 3.8+ gerekli
python --version

# Gerekli kütüphaneleri yükle
pip install -r requirements.txt
```

### 2. API Key Ayarları
```bash
# .env dosyası oluştur
cp .env.example .env

# API key'lerinizi girin
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

### 3. Docker ile (Opsiyonel)
```bash
cd crypto_trading_bot
docker build -t crypto-bot .
docker run -e BINANCE_API_KEY=xxx -e BINANCE_API_SECRET=yyy crypto-bot
```

## 🚀 Kullanım

### 1. Training Pipeline
```bash
# Tam training pipeline (veri hazırlama + feature engineering + hyperparameter tuning + eğitim + değerlendirme)
python crypto_trading_bot/main.py --mode train

# Hyperparameter tuning'ı atlayarak hızlı eğitim
python crypto_trading_bot/main.py --mode train --skip-tuning

# Özel konfigürasyon ile
python crypto_trading_bot/main.py --mode train --config config/custom_config.json
```

### 2. Backtesting
```bash
# Eğitilmiş model ile backtest
python crypto_trading_bot/main.py --mode backtest --model-path models/best_model
```

### 3. Live Trading
```bash
# Paper trading (önerilen)
python crypto_trading_bot/main.py --mode trade --paper-trading

# Gerçek trading (dikkatli olun!)
python crypto_trading_bot/main.py --mode trade --model-path models/best_model
```

### 4. Monitoring Dashboard
```bash
# Real-time dashboard
python crypto_trading_bot/main.py --mode monitor
```

### 5. Tam Pipeline
```bash
# Train -> Backtest -> Trade
python crypto_trading_bot/main.py --mode full
```

## ⚙️ Konfigürasyon

### Örnek Konfigürasyon
```json
{
  "data_preparation": {
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "timeframes": ["1h", "4h", "1d"],
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  },
  "feature_engineering": {
    "lookback_window": 50,
    "technical_indicators": true,
    "market_regime_features": true,
    "sentiment_features": false,
    "risk_features": true
  },
  "hyperparameter_tuning": {
    "n_trials": 50,
    "parallel": true,
    "n_jobs": -1
  },
  "model_training": {
    "use_curriculum": true,
    "total_timesteps": 1000000
  },
  "trading": {
    "paper_trading": true,
    "max_position_size": 0.1,
    "stop_loss": 0.02,
    "take_profit": 0.04
  }
}
```

## 📊 Sonuçlar ve Raporlar

### Training Sonuçları
- `models/` - Eğitilmiş modeller
- `results/` - Değerlendirme sonuçları
- `logs/` - Detaylı loglar
- `hyperparameter_results/` - Optimization sonuçları

### Performance Metrikleri
- **Sharpe Ratio** - Risk-ayarlı getiri
- **Sortino Ratio** - Downside risk
- **Calmar Ratio** - Maximum drawdown
- **Profit Factor** - Kazanç/kayıp oranı
- **Win Rate** - Kazanan trade oranı
- **VaR** - Value at Risk

## 🧪 Test

```bash
# Tüm testleri çalıştır
python -m pytest tests/

# Belirli test dosyası
python -m pytest tests/test_model.py

# Coverage ile
python -m pytest tests/ --cov=crypto_trading_bot
```

## 🔒 Güvenlik ve Risk Uyarıları

### ⚠️ Önemli Uyarılar
1. **Gerçek trade öncesi mutlaka paper trading ile uzun süre test yapın!**
2. **API anahtarlarınızı kimseyle paylaşmayın**
3. **Sadece kaybetmeyi göze alabileceğiniz miktarla trade yapın**
4. **Risk yönetimi parametrelerini dikkatli ayarlayın**
5. **Sistem sürekli monitoring altında olmalı**

### 🔧 Risk Yönetimi Önerileri
- Maximum position size: %5-10
- Stop-loss: %2-3
- Daily loss limit: %5
- Portfolio diversification
- Regular rebalancing

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 Destek

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: [bsarigul453@gmail.com]

## 🙏 Teşekkürler

- **Stable-Baselines3** - RL framework
- **Binance** - API ve veri
- **TA-Lib** - Teknik göstergeler
- **Optuna** - Hyperparameter optimization

---

**⚠️ Uyarı**: Bu yazılım eğitim amaçlıdır. Gerçek trading için profesyonel danışmanlık alın ve risk yönetimi kurallarına uyun. 
