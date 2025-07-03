# Gelişmiş Multi-Asset RL Trading Bot

## Özellikler
- Çoklu varlık (N sembol) portföy yönetimi
- Lookback window ile geçmiş teknik veri kullanımı
- Continuous action space (her varlık için -1 ile +1 arası)
- Teknik göstergeler: Close, MA10, EMA20, MACD
- PPO ajanı ile eğitim (Stable-Baselines3)
- VecNormalize, SubprocVecEnv, EvalCallback, StopTrainingOnNoModelImprovement
- Eğitim ve test ayrımı, portföy büyüklüğü loglama ve görselleştirme

## Kurulum
```bash
pip install -r requirements.txt
```

## Eğitim
```bash
python train_multi_asset.py
```

## Test
```bash
python test_multi_asset.py
```

- Test sonrası `data/portfolio_log.csv` ve `data/portfolio_balance.png` dosyaları oluşur.

## Dosya Yapısı
- `envs/multi_asset_env.py`: Gelişmiş RL ortamı
- `utils/data_utils.py`: Veri çekme, teknik indikatör, scaler
- `train_multi_asset.py`: PPO ile eğitim
- `test_multi_asset.py`: Eğitimli ajan ile test ve görselleştirme
- `requirements.txt`: Gereksinimler
- `data/`: Log, model ve çıktı dosyaları

## Notlar
- Matplotlib backend olarak `Agg` kullanılır, GUI gerekmez.
- Tüm semboller aynı tarih aralığında hizalanır.
- Reward fonksiyonu: portföy değer değişimi 