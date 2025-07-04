import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Binance API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "YOUR_API_SECRET")

# Fear & Greed Index API
FEAR_GREED_API = "https://api.alternative.me/fng/"

# Diğer API endpoints
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
REDDIT_API_KEY = os.getenv("REDDIT_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

def load_api_config():
    """API konfigürasyonunu yükle ve doğrula"""
    config = {
        'binance': {
            'api_key': BINANCE_API_KEY,
            'api_secret': BINANCE_API_SECRET
        },
        'fear_greed': {
            'url': FEAR_GREED_API
        },
        'twitter': {
            'api_key': TWITTER_API_KEY
        },
        'reddit': {
            'api_key': REDDIT_API_KEY
        },
        'news': {
            'api_key': NEWS_API_KEY
        }
    }
    
    # API key kontrolü
    if BINANCE_API_KEY == "YOUR_API_KEY":
        print("⚠️  Uyarı: Binance API key'leri ayarlanmamış!")
        print("   .env dosyasına API key'lerinizi ekleyin:")
        print("   BINANCE_API_KEY=your_api_key_here")
        print("   BINANCE_API_SECRET=your_api_secret_here")
    
    return config

def validate_api_keys():
    """API key'lerin geçerli olup olmadığını kontrol et"""
    if BINANCE_API_KEY == "YOUR_API_KEY" or BINANCE_API_SECRET == "YOUR_API_SECRET":
        return False
    return True

# Kullanım örneği:
# config = load_api_config()
# if validate_api_keys():
#     print("API keys are valid")
# else:
#     print("Please set up your API keys") 