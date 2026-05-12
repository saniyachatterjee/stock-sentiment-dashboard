import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = ["AAPL", "TSLA", "MSFT"]
START_DATE = "2025-01-01"
END_DATE   = "2026-05-11"
OUT_DIR    = Path("data/raw")

def fetch_prices():
    all_frames = []

    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)

        if df.empty:
            print(f"  WARNING: No data for {ticker}, skipping.")
            continue

        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df["ticker"] = ticker
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    out_path  = OUT_DIR / "prices.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    print(combined.tail())

if __name__ == "__main__":
    fetch_prices()