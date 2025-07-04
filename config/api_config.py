import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()
 
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "YOUR_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "YOUR_API_SECRET") 