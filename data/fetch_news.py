import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

TICKERS = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft"
}
OUT_DIR = Path("data/raw")

RSS_FEEDS = {
    "AAPL": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
        "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    ],
    "TSLA": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US",
    ],
    "MSFT": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    ]
}

def fetch_rss(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        root = ET.fromstring(r.content)
        articles = []
        for item in root.iter("item"):
            title   = item.findtext("title", "")
            pubdate = item.findtext("pubDate", "")
            link    = item.findtext("link", "")
            if title:
                try:
                    dt = datetime.strptime(pubdate, "%a, %d %b %Y %H:%M:%S %z")
                except:
                    dt = None
                articles.append({
                    "headline": title,
                    "published_at": dt,
                    "url": link
                })
        return articles
    except Exception as e:
        print(f"  RSS error: {e}")
        return []

def fetch_news():
    all_articles = []

    for ticker, name in TICKERS.items():
        print(f"Fetching news for {ticker}...")
        feeds = RSS_FEEDS.get(ticker, [])
        count = 0
        for url in feeds:
            articles = fetch_rss(url)
            for a in articles:
                a["ticker"] = ticker
                all_articles.append(a)
            count += len(articles)
        print(f"  Got {count} articles")

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["date"] = df["published_at"].dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["headline", "date"])
    df = df.drop_duplicates(subset=["headline"])

    out_path = OUT_DIR / "news.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} articles to {out_path}")
    print(df[["ticker", "headline", "date"]].head(6))
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(df["date"].value_counts().sort_index())

if __name__ == "__main__":
    fetch_news()