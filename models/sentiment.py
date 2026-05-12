import pandas as pd
import numpy as np
from pathlib import Path

IN_PRICES = Path("data/raw/prices.csv")
OUT_PATH  = Path("data/processed/master.parquet")

def compute_proxy_sentiment(df):
    """
    Proxy sentiment = normalized daily return.
    Large positive return → bullish sentiment
    Large negative return → bearish sentiment
    """
    df = df.sort_values(["ticker", "date"]).copy()

    # daily % return
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()

    # 5-day rolling return (medium term trend)
    df["return_5d"] = df.groupby("ticker")["close"].pct_change(5)

    # normalize returns to -1 to +1 range using tanh
    df["sentiment_score"] = np.tanh(df["daily_return"] * 10)

    # 3-day rolling average sentiment
    df["sentiment_3d_avg"] = (
        df.groupby("ticker")["sentiment_score"]
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # label: positive / negative / neutral
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-np.inf, -0.1, 0.1, np.inf],
        labels=["negative", "neutral", "positive"]
    )

    # bullish/bearish counts (binary per day)
    df["bullish_count"] = (df["sentiment_score"] > 0.1).astype(int)
    df["bearish_count"] = (df["sentiment_score"] < -0.1).astype(int)
    df["headline_count"] = 1  # placeholder

    return df

def add_technical_indicators(df):
    """Add technical features that will help the LSTM"""
    df = df.sort_values(["ticker", "date"]).copy()

    grp = df.groupby("ticker")["close"]

    # moving averages
    df["ma_5"]  = grp.transform(lambda x: x.rolling(5,  min_periods=1).mean())
    df["ma_20"] = grp.transform(lambda x: x.rolling(20, min_periods=1).mean())

    # RSI (14-day)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df["rsi"] = df.groupby("ticker")["close"].transform(compute_rsi)

    # price momentum
    df["momentum"] = df.groupby("ticker")["close"].pct_change(10)

    # volatility (10-day rolling std of returns)
    df["volatility"] = (
        df.groupby("ticker")["daily_return"]
          .transform(lambda x: x.rolling(10, min_periods=1).std())
    )

    return df

if __name__ == "__main__":
    print("Loading price data...")
    prices_df = pd.read_csv(IN_PRICES)
    prices_df["date"] = pd.to_datetime(prices_df["date"])

    print("Computing proxy sentiment...")
    master = compute_proxy_sentiment(prices_df)

    print("Adding technical indicators...")
    master = add_technical_indicators(master)

    # drop rows with NaN in key columns
    master = master.dropna(subset=["daily_return", "sentiment_score"])
    master = master.reset_index(drop=True)

    print(f"\nMaster shape: {master.shape}")
    print(f"Columns: {master.columns.tolist()}")
    print(f"Date range: {master['date'].min()} to {master['date'].max()}")

    print("\n--- Sample rows ---")
    print(master[["date", "ticker", "close", "daily_return",
                  "sentiment_score", "sentiment_label",
                  "ma_5", "rsi"]].head(10).to_string())

    print("\n--- Sentiment breakdown ---")
    print(master.groupby(["ticker","sentiment_label"]).size().unstack())

    master.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")