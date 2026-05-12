import pandas as pd

master = pd.read_parquet("data/processed/master.parquet")

print("=== MASTER DATAFRAME ===")
print(f"Shape: {master.shape}")
print(f"Columns: {master.columns.tolist()}")
print(f"\nTickers: {master['ticker'].unique()}")
print(f"Date range: {master['date'].min()} to {master['date'].max()}")

print("\n--- Rows WITH news ---")
has_news = master[master["headline_count"] > 0]
print(has_news[["date", "ticker", "close", "sentiment_score", "headline_count"]].head(8))

print("\n--- Sentiment breakdown per ticker ---")
print(master.groupby("ticker")["sentiment_score"].describe().round(3))