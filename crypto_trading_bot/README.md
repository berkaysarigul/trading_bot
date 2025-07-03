# Crypto Trading Bot

Bu proje, PPO + LSTM tabanlı, çoklu kripto varlıklar için optimize edilmiş, gelişmiş bir trading bot sistemidir. Binance API ile gerçek zamanlı veri ve işlem desteği, gelişmiş teknik göstergeler, risk yönetimi, performans metrikleri ve canlı monitoring özellikleri içerir.

## Klasör Yapısı

- config/: API ve model konfigürasyonları
- data/: Veri çekme ve işleme modülleri
- environments/: RL ortamları
- models/: Model mimarileri
- monitoring/: Dashboard ve raporlama
- trading/: Canlı işlem ve emir yönetimi
- utils/: Risk yönetimi, metrikler, backtest

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

Her modül bağımsız olarak geliştirilebilir ve entegre edilebilir. Ayrıntılı kullanım için ilgili modül dosyalarına bakınız. 