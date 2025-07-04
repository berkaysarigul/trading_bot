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
    total_market_cap = sum(market_caps.values())
    if total_market_cap == 0:
        print("Uyarı: Toplam market cap sıfır, ağırlıklandırma yapılmadı.")
        return df_dict
    weighted_data = {}
    for symbol, dfs in df_dict.items():
        cap = market_caps.get(symbol, 0)
        weight = cap / total_market_cap if total_market_cap != 0 else 0
        weighted_data[symbol] = {}
        for tf, df in dfs.items():
            df = df.copy()
            df['market_cap_weight'] = weight
            weighted_data[symbol][tf] = df
    return weighted_data

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

def fetch_twitter_sentiment(keyword, bearer_token=None):
    """
    Twitter sentiment çekme (opsiyonel)
    Gerçek kullanım için Twitter API v2 gerekli
    """
    if not bearer_token:
        print("Twitter API token gerekli")
        return 0.0
    
    try:
        # Twitter API v2 kullanımı (gerçek implementasyon)
        # import tweepy
        # client = tweepy.Client(bearer_token=bearer_token)
        # tweets = client.search_recent_tweets(query=keyword, max_results=100)
        # sentiment = analyze_sentiment(tweets)
        return 0.0
    except Exception as e:
        print(f"Twitter sentiment error: {e}")
        return 0.0

def fetch_reddit_sentiment(keyword, subreddit="cryptocurrency"):
    """
    Reddit sentiment çekme (opsiyonel)
    """
    try:
        # Reddit API kullanımı (gerçek implementasyon)
        # import praw
        # reddit = praw.Reddit(...)
        # subreddit = reddit.subreddit(subreddit)
        # posts = subreddit.search(keyword, limit=100)
        # sentiment = analyze_sentiment(posts)
        return 0.0
    except Exception as e:
        print(f"Reddit sentiment error: {e}")
        return 0.0

def fetch_news_sentiment(keyword, api_key=None):
    """
    Crypto news sentiment çekme (opsiyonel)
    """
    if not api_key:
        print("News API key gerekli")
        return 0.0
    
    try:
        # News API kullanımı (gerçek implementasyon)
        # url = f"https://newsapi.org/v2/everything?q={keyword}&apiKey={api_key}"
        # response = requests.get(url)
        # articles = response.json()['articles']
        # sentiment = analyze_sentiment(articles)
        return 0.0
    except Exception as e:
        print(f"News sentiment error: {e}")
        return 0.0

def analyze_sentiment(texts):
    """
    Basit sentiment analizi (gerçek kullanımda daha gelişmiş model kullanılabilir)
    """
    # Burada gerçek sentiment analizi yapılabilir
    # Örnek: transformers, textblob, vaderSentiment vb.
    return 0.0 