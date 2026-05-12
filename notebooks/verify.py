import pandas as pd

prices = pd.read_csv("data/raw/prices.csv")
news   = pd.read_csv("data/raw/news.csv")

print("=== PRICES ===")
print(f"Shape: {prices.shape}")
print(f"Tickers: {prices['ticker'].unique()}")
print(prices.head(3))

print("\n=== NEWS ===")
print(f"Shape: {news.shape}")
print(news[["ticker","headline","date"]].head(5))
print(f"\nArticles per ticker:\n{news['ticker'].value_counts()}")