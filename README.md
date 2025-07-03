# Crypto Trading Bot

Bu proje, çoklu kripto varlıklar için PPO + LSTM tabanlı, gelişmiş teknik göstergeler ve risk yönetimi içeren, modüler ve production-ready bir trading bot altyapısıdır.

## Özellikler
- Binance API ile gerçek zamanlı veri ve işlem desteği
- Çoklu coin ve çoklu timeframe desteği
- Gelişmiş teknik göstergeler (RSI, MACD, Bollinger Bands, VWAP, ATR, vb.)
- Market cap ağırlıklandırması, Fear & Greed, sosyal/news sentiment
- PPO, LSTM, Multi-head Attention, Ensemble model desteği
- Gelişmiş risk yönetimi (stop-loss, take-profit, VaR, correlation, slippage, likidite)
- Backtest, walk-forward, Monte Carlo ve stress test
- Streamlit dashboard, Telegram/Discord alert, otomatik rapor
- Paper trading ve gerçek trade desteği
- Docker ile kolay deployment
- Otomatik unit testler

## Klasör Yapısı
```
crypto_trading_bot/
  ├── config/
  ├── data/
  ├── environments/
  ├── models/
  ├── monitoring/
  ├── trading/
  ├── utils/
  ├── main.py
  ├── Dockerfile
  └── start.sh

tests/
README.md
requirements.txt
```

## Kurulum
```bash
pip install -r requirements.txt
```
veya Docker ile:
```bash
cd crypto_trading_bot
# API key'lerinizi .env veya start.sh ile girin
docker build -t crypto-bot .
docker run -e BINANCE_API_KEY=xxx -e BINANCE_API_SECRET=yyy crypto-bot
```

## Kullanım
- `main.py` ile veri çekme, feature engineering, backtest ve dashboard otomatik çalışır.
- Kendi modelinizi eğitmek için PPO/LSTM/Attention modellerini ve RL eğitim döngüsünü kullanabilirsiniz.
- Paper trading ve gerçek trade için `LiveTrader` modülünü kullanın.
- Tüm testleri çalıştırmak için:
```bash
python -m unittest discover tests
```

## API Key Ayarları
- `crypto_trading_bot/config/api_config.py` dosyasına veya environment variable olarak girin.

## Uyarı
- Gerçek trade öncesi mutlaka paper trading ile uzun süre test yapın!
- API anahtarlarınızı kimseyle paylaşmayın.

## Lisans
MIT 