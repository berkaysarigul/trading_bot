import requests
import pandas as pd

def get_market_caps(symbols):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "ids": ",".join([s.split('/')[0].lower() for s in symbols])}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        market_caps = {item['symbol'].upper(): item['market_cap'] for item in data}
        return market_caps
    except Exception as e:
        print(f"Market cap fetch error: {e}")
        return {s: 1 for s in symbols}

def apply_market_cap_weights(df_dict, market_caps):
    total = sum(market_caps.values())
    weights = {k: v / total for k, v in market_caps.items()}
    for symbol, dfs in df_dict.items():
        for tf, df in dfs.items():
            df['market_cap_weight'] = weights.get(symbol.split('/')[0], 1/len(market_caps))
    return df_dict

def add_fear_greed_feature(df, fear_greed_value):
    df['fear_greed'] = fear_greed_value
    return df

def detect_whale_movements(transactions, threshold=1_000_000):
    # transactions: [{'amount': float, 'from': str, 'to': str, ...}, ...]
    whales = [tx for tx in transactions if tx['amount'] >= threshold]
    return whales

def add_twitter_sentiment_feature(df, sentiment_score):
    df['twitter_sentiment'] = sentiment_score
    return df

def add_reddit_sentiment_feature(df, sentiment_score):
    df['reddit_sentiment'] = sentiment_score
    return df

def add_news_sentiment_feature(df, sentiment_score):
    df['news_sentiment'] = sentiment_score
    return df

# Örnek: Twitter sentiment çekme iskeleti (gerçek uygulamada API anahtarı gerekir)
def fetch_twitter_sentiment(keyword, bearer_token):
    # Burada gerçek bir sentiment API veya kendi modelin kullanılabilir
    # Örnek: https://developer.twitter.com/en/docs/twitter-api
    # Dönüş: [-1, 1] arası bir skor
    return 0.0  # Dummy

# Örnek: Reddit sentiment çekme iskeleti
def fetch_reddit_sentiment(keyword):
    # Pushshift, snscrape veya başka bir API ile veri çekilebilir
    return 0.0  # Dummy

# Örnek: Crypto news sentiment çekme iskeleti
def fetch_news_sentiment(keyword):
    # NewsAPI, cryptopanic, alternative.me gibi kaynaklardan haber çekip sentiment analizi yapılabilir
    return 0.0  # Dummy 