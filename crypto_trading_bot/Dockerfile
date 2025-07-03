FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ortam değişkenleri örneği (güvenlik için production'da .env kullan!)
# ENV BINANCE_API_KEY=your_key
# ENV BINANCE_API_SECRET=your_secret

CMD ["python", "main.py"] 